//! Pipelines and bind-group layouts, compiled once per Renderer and shared by scenes.
use super::*;
use crate::forward_variant::FeatureMask;

/// One owner for device pipeline slots and the shared installed-scene set.
#[derive(Default)]
pub(crate) struct Registry {
    pub(crate) scene: [OnceLock<std::sync::Arc<CachedPipelines>>; 2],
    pub(crate) environment: OnceLock<wgpu::ComputePipeline>,
    pub(crate) environment_reduce: OnceLock<wgpu::ComputePipeline>,
    pub(crate) display_stats: OnceLock<[wgpu::ComputePipeline; 2]>,
    pub(crate) light_temperature: OnceLock<wgpu::ComputePipeline>,
    pub(crate) camera_indirect_args: OnceLock<wgpu::ComputePipeline>,
    pub(crate) mips: OnceLock<([wgpu::RenderPipeline; 3], wgpu::Sampler)>,
    pub(crate) reconstruction: OnceLock<std::sync::Arc<crate::denoise::spatial::Pipelines>>,
    pub(crate) bc7: OnceLock<crate::gpu_bc7::Codec>,
}

/// Prewarm only a registry this scene can use. Media, denoising and ray
/// refraction own scene-specific layouts even without a material program.
/// The final registry is selected after material and medium installation.
pub(crate) fn prewarm_builtin(scene: &Scene, effects: EffectPlan) -> Result<bool> {
    Ok(scene.render.aovs.is_empty()
        && scene.images.is_empty()
        && crate::attributes::names(scene)?.is_empty()
        && !programs::has_programs(scene)
        && !crate::rt::ray_shadows(scene)
        && !crate::rt::ray_refraction(scene)
        && !effects.denoise
        && !effects.probes
        && environment_node::Settings::from_scene(scene)?.medium.is_none())
}

pub(crate) fn control_formats(denoise: bool) -> &'static [Option<wgpu::TextureFormat>] {
    const FORMATS: [Option<wgpu::TextureFormat>;5] = [
        Some(wgpu::TextureFormat::Rgba32Uint),
        Some(wgpu::TextureFormat::Rgba16Float),Some(wgpu::TextureFormat::Rgba16Float),
        Some(wgpu::TextureFormat::Rgba16Float),Some(wgpu::TextureFormat::Rgba32Float),
    ];
    &FORMATS[..if denoise {5} else {4}]
}

/// All lazy descriptor jobs use the same compile-once and scoped-worker path.
/// A failed build (a scene shader that does not compile) leaves its slot empty
/// and returns the error through the job's handle.
type Job<'scope> = std::thread::ScopedJoinHandle<'scope, Result<()>>;
fn compile<'scope, T: Send + Sync + 'scope, F: FnOnce() -> Result<T> + Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, '_>,
    slot: &'scope OnceLock<T>,
    needed: bool,
    build: F,
) -> Option<Job<'scope>> {
    (needed && slot.get().is_none()).then(|| scope.spawn(move || try_init(slot, build).map(drop)))
}
/// `OnceLock::get_or_init` for a fallible build: concurrent callers still wait for
/// one build; a failure leaves the slot empty and returns its error to that caller.
pub(crate) fn try_init<T>(slot: &OnceLock<T>, build: impl FnOnce() -> Result<T>) -> Result<&T> {
    struct Failed;
    let mut failure = None;
    // OnceLock is not poisoned by an unwinding initializer; resume_unwind skips the panic hook.
    let init = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| slot.get_or_init(|| {
        build().unwrap_or_else(|error| {
            failure = Some(error);
            std::panic::resume_unwind(Box::new(Failed))
        })
    })));
    match (init, failure) {
        (Ok(value), _) => Ok(value),
        (Err(_), Some(error)) => Err(error),
        (Err(panic), None) => std::panic::resume_unwind(panic),
    }
}
/// A scoped worker's result, re-raising its panic.
pub(crate) fn join<T>(job: std::thread::ScopedJoinHandle<'_, T>) -> T {
    job.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}
/// Join lazy descriptor jobs, returning the first failure.
fn joined<'scope>(jobs: impl IntoIterator<Item = Option<Job<'scope>>>) -> Result<()> {
    jobs.into_iter().flatten().try_for_each(join)
}

/// Owned pipeline-creation handle: background variant workers outlive the install
/// call that started them, so they cannot borrow the Renderer.
#[derive(Clone)]
pub(crate) struct Compiler {
    #[cfg(test)]
    pub(crate) hit_selections: std::sync::Arc<Mutex<Vec<(ProgramKernel, ProgramKernel)>>>,
    device: wgpu::Device,
    cache: Option<wgpu::PipelineCache>,
    /// Wall time per driver pipeline creation call; concurrent times overlap. On Metal a call
    /// includes its wait for the process's compile admission (wgpu-hal metal/compile_admission.rs).
    pub(crate) timings: std::sync::Arc<Mutex<std::collections::BTreeMap<String, f64>>>,
    known: std::sync::Arc<Mutex<Known>>,
    pending: std::sync::Arc<(Mutex<Pending>, std::sync::Condvar)>,
    schedule: std::sync::Arc<(Mutex<Schedule>, std::sync::Condvar)>,
    /// Driver creations so far; unchanged since the last save means nothing to persist.
    created: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    persistence: std::sync::Arc<Persistence>,
}
/// How many pipelines the device's driver cache has gained, where the backend counts them. A
/// creation through the cache is a hit or an addition, and only an addition changes what a
/// serialization would write. Metal counts the additions to a device's binary archives, and a
/// device here has the one pipeline cache; Vulkan does not say, so there every save after a
/// creation serializes and compares the bytes.
fn cache_additions(device: &wgpu::Device) -> Option<usize> {
    #[cfg(target_os = "macos")]
    // SAFETY: only reads a counter of the backend device.
    if let Some(metal) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
        return Some(metal.archive_additions());
    }
    let _ = device;
    None
}
/// Where a compiler's finished driver work is saved in the disk store.
#[derive(Clone)]
pub(crate) struct Blob {
    /// The driver's pipeline-cache entry.
    pub(crate) path: std::path::PathBuf,
    /// Without a persistent pipeline cache: the entry recording that this process's driver work reached
    /// the system shader cache.
    pub(crate) marker: Option<std::path::PathBuf>,
    /// Content of the entry the pipeline cache was loaded from: unchanged bytes are not rewritten.
    pub(crate) loaded: Option<blake3::Hash>,
    /// False for CPU-only release preparation, which cannot warm a driver's cache or its ledger.
    pub(crate) save: bool,
}
impl Blob {
    /// A compiler without a driver cache saves only its ledger.
    #[cfg(test)]
    pub(crate) fn ledger_only() -> Self {
        Self { path: Default::default(), marker: None, loaded: None, save: true }
    }
}
/// What of a compiler's driver work is on disk; one state for all its clones.
struct Persistence {
    blob: Blob,
    /// Driver creations counted at the last save, and the blob content on disk. Held for a whole
    /// save, so saves never interleave.
    saved: Mutex<(usize, Option<blake3::Hash>)>,
    /// The driver cache's additions (`cache_additions`) covered by the last serialization.
    /// Written only while `saved` is held.
    serialized_additions: std::sync::atomic::AtomicUsize,
    /// A background save waits for the cache persistence worker. Set until that save holds `saved`,
    /// so a save asked for is never neither queued nor running.
    queued: std::sync::atomic::AtomicBool,
}
/// Background variant workers still compiling, and the first one that failed.
#[derive(Default)]
struct Pending {
    workers: usize,
    failure: Option<String>,
}
/// Driver pipeline creation around critical-path work (the hardware encoder attach).
#[derive(Default)]
struct Schedule {
    critical: bool,
    foreground_preparations: usize,
    active: usize,
}
/// One admitted driver pipeline creation; frees its place when dropped.
struct Permit<'a>(&'a (Mutex<Schedule>, std::sync::Condvar));
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.0.lock().unwrap().active -= 1;
        self.0.1.notify_all();
    }
}
/// Defer optional variants until every foreground preparation has finished.
/// This covers Slang admission as well as driver calls, including error exits.
pub(crate) struct ForegroundPreparation(std::sync::Arc<(Mutex<Schedule>, std::sync::Condvar)>);
impl Drop for ForegroundPreparation {
    fn drop(&mut self) {
        self.0.0.lock().unwrap().foreground_preparations -= 1;
        self.0.1.notify_all();
    }
}
impl<'a> Permit<'a> {
    fn new(schedule: &'a (Mutex<Schedule>, std::sync::Condvar), background: bool) -> Self {
        let (state, signal) = schedule;
        let mut state = signal.wait_while(state.lock().unwrap(), |s| {
            (background && s.foreground_preparations != 0)
                || s.critical && (background || s.active >= crate::constants::CRITICAL_PIPELINE_THREADS)
        }).unwrap();
        state.active += 1;
        Self(schedule)
    }
}
impl Compiler {
    pub(crate) fn new(
        device: &wgpu::Device,
        cache: Option<&wgpu::PipelineCache>,
        known: Known,
        blob: Blob,
    ) -> Self {
        let persistence = std::sync::Arc::new(Persistence {
            saved: Mutex::new((0, blob.loaded)),
            blob,
            // The loaded blob is the cache as it stands: only additions from here on are new.
            serialized_additions: cache_additions(device).unwrap_or(0).into(),
            queued: Default::default(),
        });
        Self {
            #[cfg(test)]
            hit_selections: Default::default(),
            device: device.clone(),
            cache: cache.cloned(),
            timings: Default::default(),
            known: std::sync::Arc::new(Mutex::new(known)),
            pending: Default::default(),
            schedule: Default::default(),
            created: Default::default(),
            persistence,
        }
    }
    pub(crate) fn created(&self) -> usize {
        self.created.load(std::sync::atomic::Ordering::Acquire)
    }
    /// Mark critical-path driver work (encoder attach) in flight or finished. While it
    /// runs, install creations are throttled and background variants wait.
    pub(crate) fn set_critical(&self, critical: bool) {
        self.schedule.0.lock().unwrap().critical = critical;
        self.schedule.1.notify_all();
    }
    /// Admit one creation: at most CRITICAL_PIPELINE_THREADS at once during critical
    /// work; background variants only once it has finished.
    fn permit(&self, background: bool) -> Permit<'_> {
        Permit::new(&self.schedule, background)
    }
    pub(crate) fn foreground_preparation(&self) -> ForegroundPreparation {
        self.schedule.0.lock().unwrap().foreground_preparations += 1;
        ForegroundPreparation(self.schedule.clone())
    }
    #[cfg(test)]
    pub(crate) fn is_idle(&self) -> bool {
        self.pending.0.lock().unwrap().workers == 0
    }
    /// Save the driver's pipeline cache and the known-key ledger to the disk store, on the calling
    /// thread. Setup, a dropped library renderer and `settle_pipelines` need the cache on disk
    /// before they return; renders never do, and call `save_soon`.
    ///
    /// Saving takes no creation permit and waits for none: the driver's cache object is safe to read
    /// while pipelines are created through it (Vulkan synchronizes it internally; the Metal binary
    /// archive is locked around additions and serialization, and an addition that meets a
    /// serialization is deferred instead of waiting). A pipeline that finishes during the save is
    /// in the next one.
    pub(crate) fn save(&self) -> Result<()> {
        let persistence = &*self.persistence;
        if !persistence.blob.save {
            return Ok(());
        }
        // Serialize snapshot and publication together so an older save cannot
        // overwrite a newer ledger or acknowledge its dirty state.
        let mut saved = persistence.saved.lock().unwrap();
        self.save_holding(&mut saved)
    }
    fn save_holding(&self, saved: &mut (usize, Option<blake3::Hash>)) -> Result<()> {
        let persistence = &*self.persistence;
        // Snapshot the keys before the blob: a variant finishing during serialization must not
        // be recorded as warm unless its binary is in the saved blob.
        let known = self.known_snapshot();
        let created = self.created();
        // Serialize only after new driver work, and rewrite only changed bytes: a
        // warm process (all cache hits) leaves the blob untouched.
        if created != saved.0 {
            if let Some(marker) = &persistence.blob.marker {
                if !self.publish("shader-warm", marker, b"warm")? {
                    return Ok(());
                }
            }
            // Read before serializing: the serialization writes every addition this count includes
            // (`metal::Device::archive_additions`), and one made during it is new again next time.
            let gained = cache_additions(&self.device);
            let unchanged = gained.is_some_and(|gained| gained == persistence.serialized_additions.load(std::sync::atomic::Ordering::Acquire));
            // Every creation since the last serialization was a hit: the cache on disk is the cache.
            if let (Some(cache), false) = (&self.cache, unchanged) {
                let step = crate::timing::begin_cache_save(&self.device, "serialize", "pipeline");
                let data = cache.get_data();
                step.finish(data.as_ref().map_or(0, Vec::len), data.is_some());
                let Some(data) = data else { return Ok(()); };
                let hash = blake3::hash(&data);
                if saved.1 != Some(hash) {
                    if !self.publish("pipeline", &persistence.blob.path, &data)? {
                        return Ok(());
                    }
                    saved.1 = Some(hash);
                }
                if let Some(gained) = gained {
                    persistence.serialized_additions.store(gained, std::sync::atomic::Ordering::Release);
                }
            }
            saved.0 = created;
        }
        if let Some(known) = known {
            let (step, bytes) = (crate::timing::begin_cache_save(&self.device, "publish", "pipeline-keys"), known.keys.len() * 32);
            let written = self.save_known(known);
            step.finish(bytes, matches!(written, Ok(true)));
            written?;
        }
        Ok(())
    }
    /// Write one entry of a save; false when the store's budget refused it.
    fn publish(&self, entry: &'static str, path: &std::path::Path, payload: &[u8]) -> Result<bool> {
        let step = crate::timing::begin_cache_save(&self.device, "publish", entry);
        let written = crate::disk_cache::write_entry(path, payload);
        step.finish(payload.len(), matches!(written, Ok(true)));
        Ok(written?)
    }
    /// Save in the background, on the cache persistence worker: the caller never waits for the
    /// driver's serialization or for the filesystem. Requests coalesce, and one made when no driver
    /// work is new does nothing. Ask when the pipelines a piece of work needs all exist: every request
    /// after an addition serializes the whole cache again, 0.02 to 0.1 s on Metal. A process that ends mid-save leaves the previous entries, which stay
    /// valid: entries are published by atomic rename, and the next writer removes an abandoned
    /// temporary file. A failed save is retried by the next request, as its dirty state is kept.
    pub(crate) fn save_soon(&self) {
        let persistence = &*self.persistence;
        if !persistence.blob.save {
            return;
        }
        // A save in flight holds `saved`: work finished since it began needs the save queued behind it.
        if let Ok(saved) = persistence.saved.try_lock() {
            if saved.0 == self.created() && !self.known.lock().unwrap().added {
                return;
            }
        }
        if persistence.queued.swap(true, std::sync::atomic::Ordering::AcqRel) {
            return;
        }
        let compiler = self.clone();
        crate::disk_cache::persist(move || {
            let persistence = &*compiler.persistence;
            if !persistence.blob.save {
                return;
            }
            // Queued until this save holds `saved`: a request is always visible as one or the other
            // (`save_pending`), and one made from here on queues the save after this one.
            let mut saved = persistence.saved.lock().unwrap();
            persistence.queued.store(false, std::sync::atomic::Ordering::Release);
            // Cache availability never controls rendering; the save's receipts say what was written.
            let _ = compiler.save_holding(&mut saved);
        });
    }
    /// Leave on disk whatever this compiler has finished, on the calling thread: wait for a save
    /// that is running, then save what is still new. That includes a background variant that
    /// finished after the last request; one still compiling is not waited for. A process about to
    /// exit calls this once its output is delivered. Nothing new, nothing is done.
    pub(crate) fn finish_save(&self) {
        // Cache availability never controls the exit status.
        let _ = self.save();
    }
    /// Whether a background save is queued or running.
    pub(crate) fn save_pending(&self) -> bool {
        self.persistence.queued.load(std::sync::atomic::Ordering::Acquire) || self.persistence.saved.try_lock().is_err()
    }
    /// Driver creations counted at the last save and the blob content on disk, once any save in
    /// flight has finished.
    pub(crate) fn saved(&self) -> (usize, Option<blake3::Hash>) {
        *self.persistence.saved.lock().unwrap()
    }
    /// This compiler's place in the store, with what is on disk now: a replacement compiler on the
    /// same device keeps saving there.
    #[cfg(test)]
    pub(crate) fn blob(&self) -> Blob {
        Blob { loaded: self.saved().1, ..self.persistence.blob.clone() }
    }
    /// Block until every background variant has finished; a failed worker is an error.
    pub(crate) fn settle(&self) -> Result<()> {
        let (pending, done) = &*self.pending;
        let pending = done.wait_while(pending.lock().unwrap(), |p| p.workers > 0).unwrap();
        match &pending.failure {
            Some(failure) => Err(anyhow::anyhow!("pipeline variant: {failure}")),
            None => Ok(()),
        }
    }
    /// Run one optional variant compilation after foreground preparation. A failure
    /// or panic still retires the worker (its drawn key keeps the superset fallback),
    /// so settling reports it instead of waiting forever.
    pub(crate) fn spawn_variant(&self, mode: VariantCompile, build: impl FnOnce() -> Result<()> + Send + 'static) {
        self.spawn_worker(mode, true, build)
    }
    /// One background compilation, counted until it retires. `after_foreground`
    /// holds it back until foreground preparation has finished.
    fn spawn_worker(&self, mode: VariantCompile, after_foreground: bool, build: impl FnOnce() -> Result<()> + Send + 'static) {
        self.pending.0.lock().unwrap().workers += 1;
        let c = self.clone();
        std::thread::spawn(move || {
            if let VariantCompile::Gated(gate) = &mode {
                let (open, signal) = &**gate;
                let _open = signal.wait_while(open.lock().unwrap(), |open| !*open);
            }
            let failure = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if after_foreground { drop(c.permit(true)); }
                build()
            })) {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(format!("{error:#}")),
                Err(_) => Some("worker panicked".into()),
            };
            let (pending, done) = &*c.pending;
            let mut pending = pending.lock().unwrap();
            pending.workers -= 1;
            if pending.failure.is_none() { pending.failure = failure; }
            done.notify_all();
        });
    }
    fn time(&self, label: Option<&str>, start: std::time::Instant) {
        let ms=start.elapsed().as_secs_f64()*1000.;
        crate::timing::creation(&self.device,label.unwrap_or("pipeline"),"pipeline",0.,ms);
        *self
            .timings
            .lock()
            .unwrap()
            .entry(label.unwrap_or("pipeline").into())
            .or_default() += ms;
    }
    pub(crate) fn render(&self, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
        let _permit = self.permit(false);
        let start = std::time::Instant::now();
        let _progress = vork_slang::progress::Job::start(true).named(descriptor.label.unwrap_or("pipeline"));
        let pipeline = crate::gpu_memory::render_pipeline(&self.device, descriptor);
        self.created.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.time(descriptor.label, start);
        pipeline
    }
    pub(crate) fn compute(&self, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline {
        self.compute_admitted(descriptor, false)
    }
    /// One compute pipeline creation, admitted as foreground work or as
    /// background work: the latter enters the driver only while no foreground
    /// preparation and no critical-path work is in flight.
    fn compute_admitted(&self, descriptor: &wgpu::ComputePipelineDescriptor<'_>, background: bool) -> wgpu::ComputePipeline {
        let _permit = self.permit(background);
        let start = std::time::Instant::now();
        let _progress = vork_slang::progress::Job::start(true).named(descriptor.label.unwrap_or("pipeline"));
        let pipeline = crate::gpu_memory::compute_pipeline(&self.device, descriptor);
        self.created.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.time(descriptor.label, start);
        pipeline
    }
    /// Publish an optional pipeline through the same retained slot for every install.
    pub(crate) fn install_variant<P:Clone+Send+Sync+'static>(&self,mode:&VariantCompile,slot:std::sync::Arc<OnceLock<P>>,fallback:Option<P>,build:impl FnOnce(&Compiler)->Result<P>+Send+'static)->Result<Drawn<P>> {
        if let (true,Some(pipeline))=(mode.fallback(),&fallback) {return Ok(Drawn::inline(pipeline.clone()));}
        let deferred=fallback.is_some();
        self.publish_variant(mode,slot,fallback,deferred,build)
    }
    /// Install a kernel behind the prebuilt engine. Longer renders can promote
    /// it at an exposure boundary; delivery never waits for optional compilation.
    /// Completed pipelines are saved normally; explicit settling joins workers.
    pub(crate) fn install_behind_engine<P:Clone+Send+Sync+'static>(&self,mode:&VariantCompile,slot:std::sync::Arc<OnceLock<P>>,engine:P,build:impl FnOnce(&Compiler)->Result<P>+Send+'static)->Drawn<P> {
        let (worker,target)=(self.clone(),slot.clone());
        self.spawn_worker(mode.clone(),false,move || try_init(&target,||build(&worker)).map(drop));
        Drawn {variant:slot,fallback:Some(engine),deferred:true}
    }
    /// The caller keeps its existing pass sequence until this optional kernel is ready.
    pub(crate) fn install_optional<P:Clone+Send+Sync+'static>(&self,mode:&VariantCompile,slot:std::sync::Arc<OnceLock<P>>,warm:bool,build:impl FnOnce(&Compiler)->Result<P>+Send+'static)->Result<Drawn<P>> {
        let deferred=mode.defer(slot.get().is_some() || warm,true);
        self.publish_variant(mode,slot,None,deferred,build)
    }
    fn publish_variant<P:Clone+Send+Sync+'static>(&self,mode:&VariantCompile,slot:std::sync::Arc<OnceLock<P>>,fallback:Option<P>,deferred:bool,build:impl FnOnce(&Compiler)->Result<P>+Send+'static)->Result<Drawn<P>> {
        if deferred {
            let (worker,target)=(self.clone(),slot.clone());
            self.spawn_variant(mode.clone(),move || try_init(&target,||build(&worker)).map(drop));
        } else {try_init(&slot,||build(self))?;}
        Ok(Drawn {variant:slot,fallback,deferred})
    }
    /// A render pipeline whose descriptor names no cache, created through the persistent one.
    pub(crate) fn render_cached(&self, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
        self.render(&wgpu::RenderPipelineDescriptor { cache: self.cache.as_ref(), ..descriptor.clone() })
    }
    /// A compute pipeline whose descriptor names no cache, created through the persistent one.
    pub(crate) fn compute_cached(&self, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline {
        self.compute(&wgpu::ComputePipelineDescriptor { cache: self.cache.as_ref(), ..descriptor.clone() })
    }
    pub(crate) fn compute_entry(&self, layout:&wgpu::PipelineLayout, module:&wgpu::ShaderModule, entry:&str, label:&str)->wgpu::ComputePipeline {
        self.compute(&wgpu::ComputePipelineDescriptor {label:Some(label),layout:Some(layout),module,entry_point:Some(entry),compilation_options:Default::default(),cache:self.cache.as_ref()})
    }
    pub(crate) fn is_known(&self, key: &blake3::Hash) -> bool {
        self.known.lock().unwrap().keys.contains(key.as_bytes())
    }
    pub(crate) fn is_known_input(&self,key:Option<blake3::Hash>)->bool {
        key.is_some_and(|key|self.is_known(&key))
    }
    pub(crate) fn remember(&self, key: impl Into<Option<blake3::Hash>>) {
        let Some(key)=key.into() else {return;};
        let mut known = self.known.lock().unwrap();
        known.added |= known.keys.insert(*key.as_bytes());
    }
    /// Snapshot without consuming dirty state: every unsuccessful save can retry.
    /// The renderer serializes saves before taking this snapshot.
    pub(crate) fn known_snapshot(&self) -> Option<Known> {
        let known = self.known.lock().unwrap();
        known.added.then(|| Known {
            path: known.path.clone(),
            keys: known.keys.clone(),
            added: true,
        })
    }
    /// Acknowledge only a published ledger, after its blob or system-cache marker.
    /// False when the store's budget refused it.
    pub(crate) fn save_known(&self, snapshot: Known) -> Result<bool> {
        let written = snapshot.save()?;
        if written {
            let mut known = self.known.lock().unwrap();
            // Keys only accumulate. A later insertion still needs another save;
            // remembering an existing key does not dirty a committed snapshot.
            known.added = known.keys.len() != snapshot.keys.len();
        }
        Ok(written)
    }
}
/// Moda's 2D pipelines are created like vork's own: through the persistent pipeline cache (a Metal binary archive
/// on macOS), admitted, timed and counted, so a save after new canvas pipelines writes them to disk.
impl moda_text_draw::device::PipelineHost for Compiler {
    fn render_pipeline(&self, descriptor: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
        self.render_cached(descriptor)
    }
    fn compute_pipeline(&self, descriptor: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline {
        self.compute_cached(descriptor)
    }
}
/// Content keys of geometry pipelines the persistent pipeline cache holds: a known
/// variant is a warm hit and compiles inline; an unknown one compiles in the background.
#[derive(Default)]
pub(crate) struct Known {
    path: std::path::PathBuf,
    keys: std::collections::HashSet<[u8; 32]>,
    added: bool,
}
impl Known {
    pub(crate) fn load(path: std::path::PathBuf, blob_accepted: bool) -> Self {
        let keys = blob_accepted
            .then(|| crate::disk_cache::read(&path))
            .flatten()
            .map(|bytes| {
                bytes
                    .chunks_exact(32)
                    .map(|key| key.try_into().unwrap())
                    .collect()
            })
            .unwrap_or_default();
        Self {
            path,
            keys,
            added: false,
        }
    }
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
    fn save(&self) -> Result<bool> {
        let mut keys: Vec<_> = self.keys.iter().collect();
        keys.sort_unstable();
        Ok(crate::disk_cache::write_entry(
            &self.path,
            &keys.into_iter().flatten().copied().collect::<Vec<u8>>(),
        )?)
    }
}
#[cfg(test)]
mod release_case_tests {
    use super::*;
    use crate::forward_variant::*;

    fn path(features: FeatureMask, boundaries: u32) -> ProgramKernel {
        ProgramKernel::Path { adaptive: true, catchers: false, random_walk: false, features, boundaries, emitter_profiles: false, media: None }
    }
    fn features(kernel: ProgramKernel) -> FeatureMask {
        let ProgramKernel::Path { features, .. } = kernel else { unreachable!() };
        features
    }
    const SUN: FeatureMask = 1 << FEATURE_LIGHT_SHIFT;
    const POINT: FeatureMask = 1 << (FEATURE_LIGHT_SHIFT + 1);
    const AREA: FeatureMask = 1 << (FEATURE_LIGHT_SHIFT + 3);

    /// Whatever lights and camera the prepared scene has, the bundle holds the
    /// same cases: its materials decide everything else in the key.
    #[test]
    fn a_prepared_path_scene_covers_the_same_light_and_camera_cases_whatever_its_own() {
        let materials = FEATURE_SPECIALIZED | FEATURE_INSTALLED | FEATURE_COAT | FEATURE_SUBSURFACE;
        let lights = [SUN, AREA, FEATURE_LIGHT_TYPES];
        // No solids: three light cases, the camera fact as installed.
        for camera in [0, FEATURE_CAMERA_EXTERIOR] {
            for own in [SUN, POINT, AREA, FEATURE_LIGHT_TYPES] {
                let cases = path(materials | own | camera, 0).release_cases();
                assert_eq!(cases.iter().map(|&case| features(case)).collect::<Vec<_>>(),
                    lights.iter().map(|light| materials | light | camera).collect::<Vec<_>>());
            }
        }
        // Solids: both camera cases for each light case, from either starting point.
        let expected: Vec<_> = lights.iter().flat_map(|light| [FEATURE_CAMERA_EXTERIOR, 0].map(|camera| materials | light | camera)).collect();
        for camera in [0, FEATURE_CAMERA_EXTERIOR] {
            let cases = path(materials | folded_light_kinds(POINT) | camera, 2).release_cases();
            assert_eq!(cases.iter().map(|&case| features(case)).collect::<Vec<_>>(), expected);
            // Only the two facts vary: every other field of the kernel is the scene's.
            for case in cases {
                assert_eq!(path(features(case), 2), case);
                assert_eq!(features(case) & !(FEATURE_LIGHT_TYPES | FEATURE_CAMERA_EXTERIOR), materials);
            }
        }
    }

    /// The cases are keys a real scene installs: every light case is one
    /// `folded_light_kinds` produces, so a prepared kernel is found by its key.
    #[test]
    fn release_light_cases_are_installable_keys() {
        for case in path(FEATURE_SPECIALIZED | FEATURE_INSTALLED | POINT, 0).release_cases() {
            let lights = features(case) & FEATURE_LIGHT_TYPES;
            assert_eq!(folded_light_kinds(lights), lights);
        }
        assert_eq!(folded_light_kinds(0), FEATURE_LIGHT_TYPES, "no analytic light is the every-kind case");
        assert_eq!(folded_light_kinds(SUN | AREA), FEATURE_LIGHT_TYPES, "a mix is the every-kind case");
    }

    /// Engine and test keys are not scene unions; other kernels have no cases.
    #[test]
    fn only_scene_keys_have_release_cases() {
        let engine = path((SHADING_LAYERS | SHADING_METAL) as FeatureMask, 1);
        assert_eq!(engine.release_cases(), vec![engine]);
        let gi = ProgramKernel::Gi { features: FEATURE_SPECIALIZED | SUN };
        assert_eq!(gi.release_cases(), vec![gi]);
    }
}
#[cfg(test)]
mod known_persistence_tests {
    use super::*;

    #[test]
    fn committed_snapshot_preserves_later_keys() -> Result<()> {
        let (device, _) = wgpu::Device::noop(&Default::default());
        let root = std::env::temp_dir().join(format!("vork-key-snapshot-{}", std::process::id()));
        let path = root.join("keys.entry");
        let compiler = Compiler::new(&device, None, Known::load(path.clone(), false), Blob::ledger_only());
        let first = blake3::hash(b"first pipeline");
        let later = blake3::hash(b"later pipeline");
        compiler.remember(first);
        let snapshot = compiler.known_snapshot().unwrap();
        compiler.remember(later);
        compiler.save_known(snapshot)?;
        let loaded = Known::load(path.clone(), true);
        assert!(loaded.keys.contains(first.as_bytes()));
        assert!(!loaded.keys.contains(later.as_bytes()));
        let snapshot = compiler.known_snapshot().expect("later key must remain dirty");
        compiler.remember(first); // Duplicate insertion needs no further save.
        compiler.save_known(snapshot)?;
        assert!(compiler.known_snapshot().is_none());
        let loaded = Known::load(path.clone(), true);
        assert_eq!(loaded.keys.len(), 2);
        assert!(loaded.keys.contains(later.as_bytes()));
        assert!(!loaded.added);
        assert!(Known::load(path, false).keys.is_empty());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn background_save_publishes_off_the_calling_thread_and_only_new_work() -> Result<()> {
        let (device, _) = wgpu::Device::noop(&Default::default());
        let root = std::env::temp_dir().join(format!("vork-background-save-{}", std::process::id()));
        let path = root.join("keys.entry");
        let compiler = Compiler::new(&device, None, Known::load(path.clone(), false), Blob::ledger_only());
        let key = blake3::hash(b"pipeline");
        compiler.remember(key);
        compiler.save_soon();
        compiler.save_soon(); // Coalesces with the queued save, or finds nothing left to do.
        crate::disk_cache::wait_persistence()?;
        assert_eq!(crate::disk_cache::read(&path), Some(key.as_bytes().to_vec()));
        assert!(compiler.known_snapshot().is_none());
        // Nothing is new: no save is queued.
        compiler.save_soon();
        assert!(!compiler.save_pending());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    /// A process about to exit must not take a save for finished while the worker is between
    /// taking it off the queue and running it.
    #[test]
    fn a_requested_save_stays_pending_until_it_runs_and_exit_finishes_it() -> Result<()> {
        let (device, _) = wgpu::Device::noop(&Default::default());
        let root = std::env::temp_dir().join(format!("vork-finish-save-{}", std::process::id()));
        let path = root.join("keys.entry");
        let compiler = Compiler::new(&device, None, Known::load(path.clone(), false), Blob::ledger_only());
        let key = blake3::hash(b"pipeline");
        compiler.remember(key);
        // The worker is busy, so the save stays queued: finishing runs it on this thread.
        let (release, gate) = std::sync::mpsc::channel::<()>();
        crate::disk_cache::persist(move || { let _ = gate.recv(); });
        compiler.save_soon();
        assert!(compiler.save_pending());
        compiler.finish_save();
        assert_eq!(crate::disk_cache::read(&path), Some(key.as_bytes().to_vec()));
        // The worker now takes the save off the queue but cannot run it while this thread holds the
        // saved state: wherever the worker is, the save still counts as pending.
        let later = blake3::hash(b"later pipeline");
        compiler.remember(later);
        let saved = compiler.persistence.saved.lock().unwrap();
        compiler.save_soon();
        release.send(()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(compiler.persistence.queued.load(std::sync::atomic::Ordering::Acquire));
        drop(saved);
        compiler.finish_save();
        assert!(Known::load(path.clone(), true).keys.contains(later.as_bytes()));
        crate::disk_cache::wait_persistence()?;
        assert!(!compiler.save_pending());
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn abandoned_snapshot_keeps_keys_dirty() {
        let (device, _) = wgpu::Device::noop(&Default::default());
        let compiler = Compiler::new(&device, None, Known::default(), Blob::ledger_only());
        let key = blake3::hash(b"pipeline");
        compiler.remember(key);
        drop(compiler.known_snapshot().unwrap());
        assert!(compiler.known_snapshot().unwrap().keys.contains(key.as_bytes()));
    }
}

/// How install obtains feature variants that are not yet in the persistent cache.
#[derive(Clone, Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) enum VariantCompile {
    /// Draw with the shared superset pipeline until the variant compiles in the background.
    #[default]
    Background,
    /// Draw only the generic; one-frame shots have no use for a later variant.
    /// The path kernel behind its prebuilt engine is the exception: it compiles
    /// for the next render of the scene (Compiler::install_behind_engine).
    Fallback,
    /// `vork setup`: Fallback's choices, and the hit kernels compile the superset
    /// key a later Fallback install draws with (CachedPipelines::drawn_kernel).
    Setup,
    /// Exhaustive setup's ordinary tables: retain material proofs while
    /// sharing hit evaluators across light mixes. Other kernels use Setup.
    SetupPlain,
    /// Compile every variant before install returns.
    Inline,
    /// Tests: background workers start only once the gate opens, regardless of warmth.
    Gated(std::sync::Arc<(Mutex<bool>, std::sync::Condvar)>),
}
impl VariantCompile {
    /// The caller admits a generic only when it is suitable for foreground use.
    /// Geometry generics are cheap shipped controls; camera generics need a warm
    /// driver cache, otherwise their larger JIT loses to direct specialization.
    pub(crate) fn defer(&self,warm:bool,generic_ready:bool)->bool {
        match self {
            Self::Inline=>false,
            Self::Gated(_)=>true,
            Self::Background|Self::Fallback|Self::Setup|Self::SetupPlain=>!warm && generic_ready,
        }
    }
    /// One-frame choices: draw generics and compile no later variant.
    pub(crate) fn fallback(&self)->bool {matches!(self,Self::Fallback|Self::Setup|Self::SetupPlain)}
}
/// A retained variant; the caller supplies a pipeline fallback or keeps its existing passes.
#[derive(Clone)]
pub(crate) struct Drawn<P = wgpu::RenderPipeline> {
    variant: std::sync::Arc<OnceLock<P>>,
    fallback: Option<P>,
    deferred: bool,
}
impl<P> Drawn<P> {
    pub(crate) fn inline(pipeline:P)->Self {
        Self {variant:std::sync::Arc::new(OnceLock::from(pipeline)),fallback:None,deferred:false}
    }
    pub(crate) fn current(&self, prefer_variant:bool) -> &P {
        if prefer_variant {self.variant.get().or(self.fallback.as_ref())} else {self.fallback.as_ref().or(self.variant.get())}.expect("installed variant or fallback")
    }
    pub(crate) fn deferred(&self)->bool {self.deferred}
    pub(crate) fn ready(&self) -> bool {
        self.variant.get().is_some()
    }
}

impl crate::InstalledScene {
    /// Whether every drawn pipeline is its final variant (no superset fallback in use).
    pub(crate) fn variants_ready(&self) -> bool {
        self.geometry_pipelines.values().all(Drawn::ready) && self.decoupled.as_ref().is_none_or(|d|d.variants_ready())
            && self.gi.as_ref().is_none_or(Drawn::ready) && self.reflections.as_ref().is_none_or(Drawn::ready)
            && self.path.as_ref().is_none_or(|path|path.drawn().ready())
    }
    /// Drawn pipelines installed behind a superset fallback (cold variants).
    pub(crate) fn deferred_variants(&self) -> usize {
        self.geometry_pipelines
            .values()
            .filter(|d| d.deferred())
            .count() + self.decoupled.as_ref().map_or(0,|d|d.deferred_variants())
            + usize::from(self.gi.as_ref().is_some_and(Drawn::deferred))
            + usize::from(self.reflections.as_ref().is_some_and(Drawn::deferred))
            + usize::from(self.path.as_ref().is_some_and(|path|path.drawn().deferred()))
    }
}

/// The shader one drawn key uses: module plus fragment override constants. Variant
/// backends implement this; the fallback key must cover every drawn feature and each
/// feature gate must stay `FEATURE && weight > 0`, so fallback pixels match.
pub(crate) trait VariantShader: Send + Sync {
    /// The vertex and fragment modules (one module may serve both stages).
    fn modules(&self, key: GeometryKey) -> Result<[wgpu::ShaderModule; 2]>;
    /// Whether the generic control shader ships with the renderer.
    fn generic_shipped(&self) -> bool;
    fn constants(&self, key: GeometryKey) -> Vec<(&'static str, f64)>;
    /// Shipped table evaluator, when it can serve the scene's transport.
    fn camera_generic(&self, kernel: ProgramKernel) -> Option<&'static crate::native::Compiled>;
    fn geometry_input_key(&self, key: GeometryKey,lookup_only:bool) -> Result<blake3::Hash>;
    /// Captured compiler inputs (or selected shipped bytes), before pipeline layout/state.
    /// A missing resolver receipt is cold and must not invoke Slang during scheduling.
    fn kernel_input_key(&self, kernel: ProgramKernel,lookup_only:bool) -> Result<blake3::Hash>;
    /// A compute kernel with the scene's programs at hits, and its entry.
    fn program_kernel(&self, kernel: ProgramKernel) -> Result<(wgpu::ShaderModule, &'static str)>;
    /// Whether `program_kernel` would compile nothing: shipped, or its code is stored.
    fn kernel_stored(&self, kernel: ProgramKernel) -> Result<bool>;
    fn detached_request(&self, kernel: ProgramKernel) -> Result<crate::specialization::Source>;
    /// The prebuilt path engine key that draws `kernel`'s scene, or what in the
    /// scene the engine cannot draw.
    fn path_engine(&self, kernel: ProgramKernel) -> std::result::Result<ProgramKernel, &'static str>;
}
/// Compute kernels that evaluate surfaces at ray hits. GI, reflections and path
/// transport specialize over installed capabilities; other kernels compile per
/// scene when programs or attributes require it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ProgramKernel {
    /// Path transport with adaptive or fixed sampling.
    Path { adaptive: bool, catchers: bool, random_walk: bool, features: FeatureMask, boundaries: u32, emitter_profiles: bool, media: Option<u32> },
    /// Environment radiance at the exact straight path-camera sample.
    PathEnvironment { adaptive: bool },
    /// Ray-traced diffuse indirect.
    Gi { features: FeatureMask },
    /// Ray-traced reflections that shade unseen hits.
    Reflections { features: FeatureMask },
    /// The irradiance volume update (probes.rs).
    Probes { features:FeatureMask, directional:bool },
    /// Fixed camera stages and the scene's authored-program evaluation.
    Decoupled { stage: ShadeStage, features: FeatureMask, subsurface: bool, emitter_profiles: bool, boundaries: u32 },
}
impl ProgramKernel {
    /// The prebuilt path engine's key for this path kernel: the same family
    /// (sampling, boundary capacity) with every capability, or why the engine has
    /// no such family. The engine draws the scene's picture with the scene's
    /// tables; the kernel's own key is the same engine compiled for what the
    /// scene installs. Whether a scene's resources fit the engine is the
    /// shader's question (VariantShader::path_engine).
    pub(crate) fn path_engine(self) -> std::result::Result<Self, &'static str> {
        let Self::Path { adaptive, catchers, random_walk, features, boundaries, media, .. } = self else {
            return Err("not path transport");
        };
        if random_walk { return Err("random-walk subsurface is installed"); }
        if catchers { return Err("a shadow catcher is installed"); }
        if media.is_some() { return Err("a participating medium is installed"); }
        if boundaries > crate::surface_native::PathEngine::BOUNDARIES { return Err("dielectric solids nest or overlap"); }
        // An engine key has no specialised bit, so it carries no geometry union.
        if features & crate::forward_variant::FEATURE_SPECIALIZED != 0
            && features & crate::forward_variant::FEATURE_IMPLICIT_GEOMETRY != 0 { return Err("implicit geometry is installed"); }
        let all = (crate::forward_variant::SHADING_LAYERS | crate::forward_variant::SHADING_METAL) as FeatureMask;
        Ok(Self::Path { adaptive, catchers, random_walk, features: all, boundaries, emitter_profiles: true, media })
    }
    /// The keys a release bundle holds for one prepared path scene: the scene's
    /// materials, programs and resources at each covered case of the two key
    /// facts that come from elsewhere in a scene.
    /// - Lights: a single sun, area lights only, and every kind (which is also the
    ///   key of a scene with no analytic light or with several kinds). A scene lit
    ///   by points or spots alone compiles its kernel on first use.
    /// - Camera: with solids installed, both the proof that no camera starts
    ///   inside one and its absence. Without solids the proof always holds
    ///   (passes::path::camera_exterior), so the scene's own value is the only one.
    /// A key that is not a scene union has neither fact and is its only case.
    pub(crate) fn release_cases(self) -> Vec<Self> {
        use crate::forward_variant::{FEATURE_CAMERA_EXTERIOR, FEATURE_INSTALLED, FEATURE_LIGHT_SHIFT, FEATURE_LIGHT_TYPES};
        let Self::Path { features, boundaries, .. } = self else { return vec![self] };
        if features & FEATURE_INSTALLED == 0 { return vec![self]; }
        let sun: FeatureMask = 1 << FEATURE_LIGHT_SHIFT;
        let area: FeatureMask = 1 << (FEATURE_LIGHT_SHIFT + 3);
        let cameras = if boundaries > 0 { vec![FEATURE_CAMERA_EXTERIOR, 0] } else { vec![features & FEATURE_CAMERA_EXTERIOR] };
        let mut cases = Vec::new();
        for lights in [sun, area, FEATURE_LIGHT_TYPES] {
            for &camera in &cameras {
                let mut case = self;
                if let Self::Path { features, .. } = &mut case {
                    *features = *features & !(FEATURE_LIGHT_TYPES | FEATURE_CAMERA_EXTERIOR) | lights | camera;
                }
                cases.push(case);
            }
        }
        cases
    }
    /// Untextured, unlayered table materials share one hit evaluator across light
    /// mixes. Keep runtime metalness, sidedness and every analytic light kind;
    /// do not turn on unrelated texture/layer/transport capabilities.
    pub(crate) fn plain_hit(self) -> Option<Self> {
        use crate::forward_variant::*;
        let lighting = FEATURE_LIGHT_TYPES | FEATURE_PUNCTUAL | FEATURE_FINITE
            | FEATURE_RECTANGLE | FEATURE_ELLIPSE | FEATURE_SHADOW_RAYS;
        let allowed = lighting | FEATURE_SPECIALIZED | FEATURE_LIT
            | FeatureMask::from(SHADING_METAL);
        match self {
            Self::Gi { features } if features & FEATURE_SPECIALIZED != 0 && features & !allowed == 0 => Some(Self::Gi { features: allowed }),
            Self::Reflections { features } if features & FEATURE_SPECIALIZED != 0 && features & !allowed == 0 => Some(Self::Reflections { features: allowed }),
            _ => None,
        }
    }
    /// The same kernel over the generic feature key (variant.slang Unspecialized:
    /// layered and metallic shading kept), for kernels that otherwise specialize to
    /// the scene and have no shipped generic: ray GI, reflection and probe hits,
    /// camera indirect work, and transmissive table/occupancy stages with boundaries.
    pub(crate) fn superset(self) -> Option<Self> {
        let all = (crate::forward_variant::SHADING_LAYERS | crate::forward_variant::SHADING_METAL) as FeatureMask;
        match self {
            Self::Gi {..} => Some(Self::Gi { features: all }),
            Self::Reflections {..} => Some(Self::Reflections { features: all }),
            Self::Probes {directional,..} => Some(Self::Probes { features: all, directional }),
            // These stages store evaluations, whose record size belongs to the scene.
            Self::Decoupled { stage: ShadeStage::Indirect | ShadeStage::Evaluate | ShadeStage::EvaluateOpaque, .. } => {
                use crate::forward_variant::{FEATURE_EXACT_RECORD,FEATURE_RECORD_WORDS};
                let mut superset = self;
                if let Self::Decoupled { features, .. } = &mut superset { *features = all | FEATURE_EXACT_RECORD | *features & FEATURE_RECORD_WORDS; }
                Some(superset)
            }
            Self::Decoupled { stage: ShadeStage::Occupancy, .. } => {
                let mut superset = self;
                if let Self::Decoupled { features, .. } = &mut superset { *features = all; }
                Some(superset)
            }
            _ => None,
        }
    }
    pub(crate) fn camera_label(self)->Option<String> {
        let Self::Decoupled {stage,..}=self else {return None;};
        Some(match stage {
            ShadeStage::EvaluateOpaque=>"camera material evaluate opaque table".into(),
            ShadeStage::Evaluate=>"camera material evaluate table".into(),
            ShadeStage::EvaluateProgram(id)=>format!("camera material evaluate program {id}"),
            ShadeStage::Compose=>"camera result composition".into(),
            ShadeStage::Fused=>"camera fused table shading".into(),
            ShadeStage::Indirect=>"camera unsupported indirect work".into(),
            ShadeStage::Occupancy=>"camera dielectric occupancy".into(),
            ShadeStage::Index=>"camera occupied slots".into(),
            ShadeStage::Classify=>"camera material classify".into(),
            ShadeStage::Prefix=>"camera material prefix".into(),
            ShadeStage::Scatter=>"camera material scatter".into(),
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ShadeStage {
    Indirect, Fused, Occupancy, Index, Classify, Prefix, Scatter, EvaluateOpaque, Evaluate, EvaluateProgram(u32), Compose }
impl ShadeStage {
    pub(crate) fn composes(self)->bool {matches!(self,Self::Compose|Self::Fused)}
    /// Stages that evaluate the built-in material table (Evaluate, EvaluateOpaque, Fused);
    /// each program evaluates its own bin.
    pub(crate) fn table(self)->bool {matches!(self,Self::Evaluate|Self::EvaluateOpaque|Self::Fused)}
}
#[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
pub(crate) enum CoverageStage {Depth,Visibility}
/// One drawn geometry pipeline: its pass role and, for shading roles, the material
/// feature mask it specializes. Install creates exactly the roles its draws record.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) enum Role {
    ProgramCoverage {program:u32, stage:CoverageStage},
    /// Alpha-tested shadow caster.
    Fibre(crate::fibre_transport::Stage),
    /// Double-sided opaque depth, position only.
    OpaqueDepth,
    OpaqueSingleDepth,
    /// Alpha-tested depth; `hashed` for dithered coverage (its entry reads SV_PrimitiveID).
    MaskedDepth {hashed:bool},
    /// Decoupled visibility keys, double- and single-sided and alpha-tested (passes/decoupled.rs).
    Visibility,
    VisibilitySingle,
    VisibilityMasked,
    /// Transparent draws' fragment lists in front of the visibility passes' depth.
    TransparentFragments,
    Controls(FeatureMask),
    ProgramControls { program:u32, features:FeatureMask, transparent:bool },
    TransparentControls(FeatureMask),
}
impl Role {
    /// The single selection rule shared by install (what to compile) and bundles (what to bind).
    pub(crate) fn of(
        kind: crate::bundles::Kind,
        flags: &crate::plan::DrawFlags,
        material: &crate::materials::Material,
    ) -> Self {
        // Program entries always read the primitive (ctx.primitive_id).
        let hashed=material.dithered && material.program==0;
        let role=Self::of_capabilities(kind, flags, material.double_sided, hashed, material.features);
        if material.program!=0 {
            match role {
                Self::MaskedDepth {..}=>return Self::ProgramCoverage {program:material.program,stage:CoverageStage::Depth},
                Self::VisibilityMasked=>return Self::ProgramCoverage {program:material.program,stage:CoverageStage::Visibility},
                Self::Controls(features)=>return Self::ProgramControls {program:material.program,features,transparent:false},
                Self::TransparentControls(features)=>return Self::ProgramControls {program:material.program,features,transparent:true},
                _=>{}
            }
        }
        role
    }
    /// `hashed`: table entries that declare SV_PrimitiveID (forward_variant::FEATURE_DITHERED).
    fn of_capabilities(kind: crate::bundles::Kind, flags: &crate::plan::DrawFlags,
        double_sided: bool, hashed: bool, features: FeatureMask) -> Self {
        use crate::bundles::Kind;
        let features = features | if hashed {crate::forward_variant::FEATURE_DITHERED} else {0};
        match kind {
            Kind::Fibre(stage,_) => Self::Fibre(stage),
            Kind::Depth if flags.alpha_mask => Self::MaskedDepth {hashed},
            Kind::Depth if double_sided => Self::OpaqueDepth,
            Kind::Depth => Self::OpaqueSingleDepth,
            Kind::Visibility(_) if flags.alpha_mask => Self::VisibilityMasked,
            Kind::Visibility(_) if double_sided => Self::Visibility,
            Kind::Visibility(_) => Self::VisibilitySingle,
            Kind::TransparentFragments(_) => Self::TransparentFragments,
            Kind::Controls => Self::Controls(
                features
                    & !(crate::materials::FEATURE_LIGHT_TYPES
                        | crate::materials::FEATURE_PUNCTUAL
                        | crate::materials::FEATURE_FINITE
                        | crate::materials::FEATURE_RECTANGLE
                        | crate::materials::FEATURE_ELLIPSE),
            ),
            // Controls evaluate the surface/environment response, never finite lights.
            Kind::TransparentControls => Self::TransparentControls(
                features
                    & !(crate::materials::FEATURE_LIGHT_TYPES
                        | crate::materials::FEATURE_PUNCTUAL
                        | crate::materials::FEATURE_FINITE
                        | crate::materials::FEATURE_RECTANGLE
                        | crate::materials::FEATURE_ELLIPSE),
            ),
        }
    }
    /// The material feature mask this role specializes (0 for depth and shadow roles).
    pub(crate) fn shading(self) -> FeatureMask {
        match self {
            Self::Controls(m) | Self::TransparentControls(m) | Self::ProgramControls {features:m,..} => m,
            _ => 0,
        }
    }
    /// The shared pipeline evaluating every material feature, when this role specializes.
    /// The superset keeps the key's hashed coverage: the generic is shipped with and without it.
    fn superset(self) -> Option<Self> {
        let (Self::Controls(m) | Self::TransparentControls(m)) = self else {return None};
        let all = FeatureMask::from(crate::materials::SHADING_LAYERS | crate::materials::SHADING_METAL)
            | m & crate::forward_variant::FEATURE_DITHERED;
        if m == all {return None;}
        Some(if matches!(self, Self::Controls(_)) {Self::Controls(all)} else {Self::TransparentControls(all)})
    }
    /// Bundle kinds a draw is recorded in; mirrors the recorder's draw filters.
    /// Raster colour is decoupled: opaque draws record visibility keys and
    /// controls, transparent ones fragment lists and controls, and the shade
    /// pass produces colour. Path frames record controls.
    pub(crate) fn kinds(
        flags: &crate::plan::DrawFlags,
        decoupled: bool,
    ) -> &'static [crate::bundles::Kind] {
        use crate::bundles::Kind;
        match (flags.transparent, decoupled) {
            (false, true) => &[Kind::Depth, Kind::Visibility(0), Kind::Controls],
            (false, false) => &[Kind::Depth, Kind::Controls],
            (true, true) => &[Kind::TransparentFragments(0), Kind::TransparentControls],
            (true, false) => &[Kind::TransparentControls],
        }
    }
}
/// A compute pipeline's identity: its shader inputs and the entries of its bind
/// groups. Compute pipeline state is fixed: no overrides or immediate data.
/// Installs and scene-free warming (passes::path::warm_engines) both use it.
pub(crate) fn compute_pipeline_key<E:std::fmt::Debug>(input:blake3::Hash,groups:&[Option<Vec<E>>])->blake3::Hash {
    let mut hash=blake3::Hasher::new();
    hash.update(input.as_bytes());
    hash.update(format!("{groups:?}").as_bytes());
    hash.finalize()
}
/// The bind groups of a path pipeline, as its identity states them: the path
/// kernel's own entries, the material table, no group 2, the hit buffers.
/// Greenlight's cache probe compiles this and compute_pipeline_key as written.
pub(crate) fn path_groups<E>(path:Vec<E>,material:Vec<E>,hits:E)->Vec<Option<Vec<E>>> {
    vec![Some(path),Some(material),None,Some(vec![hits])]
}
fn layout_identity(groups:&[Option<Vec<wgpu::BindGroupLayoutEntry>>])->blake3::Hash {
    blake3::hash(format!("{groups:?} immediate_size=0").as_bytes())
}

/// Installed pipelines per (vertex layout, role), shared by every scene on this registry.
pub(crate) type GeometryKey = (crate::geometry::VertexLayout, Role);
pub(crate) struct CachedPipelines {
    pub(crate) raster_samples: u32,
    pub(crate) denoise: bool,
    /// Fingerprints of the descriptor entries used to construct these layouts.
    depth_identity: blake3::Hash,
    fragments_identity: blake3::Hash,
    fibre_identity: blake3::Hash,
    shadow_identity: blake3::Hash,
    resolve_identity: blake3::Hash,
    material_entries: Vec<wgpu::BindGroupLayoutEntry>,
    probes: bool,
    /// Camera ray transport: shadows, refraction/thickness, or probes.
    pub(crate) ray_shadows: bool,
    /// Material programs provide procedural coverage to ray transport.
    geometry: Mutex<
        std::collections::HashMap<GeometryKey, std::sync::Arc<OnceLock<wgpu::RenderPipeline>>>,
    >,
    pub(crate) camera_evaluations: Mutex<std::collections::HashMap<ProgramKernel,std::sync::Arc<OnceLock<wgpu::ComputePipeline>>>>,
    /// Module and override constants per drawn key (the variant backend seam).
    pub(crate) shader: Box<dyn VariantShader>,
    /// Decoupled slot merge (passes/decoupled.rs), built on first use.
    pub(crate) decoupled_merge: OnceLock<wgpu::ComputePipeline>,
    cluster_module: OnceLock<wgpu::ShaderModule>,
    deform_module: OnceLock<wgpu::ShaderModule>,
    depth_layout: wgpu::PipelineLayout,
    /// Decoupled transparent fragments' group 2 (passes/decoupled.rs Fragments) and layout.
    pub(crate) fragments_layout: wgpu::BindGroupLayout,
    pub(crate) fragments_geometry_layout: wgpu::BindGroupLayout,
    pub(crate) field_controls: bool,
    pub(crate) software_raster: bool,
    pub(crate) program_trace: bool,
    fragments_pipeline_layout: wgpu::PipelineLayout,
    fibre_layout: wgpu::PipelineLayout,
    pub(crate) fibre_capture_layout: wgpu::BindGroupLayout,
    control_targets: Vec<Option<wgpu::ColorTargetState>>,

    pub(crate) resolve_module: String,
    control_module: OnceLock<wgpu::ShaderModule>,
    resolve_layout: wgpu::PipelineLayout,
    pub(crate) deform_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) cluster_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) camera_cull_pipeline: OnceLock<[wgpu::ComputePipeline; 3]>,
    pub(crate) global_layout: wgpu::BindGroupLayout,
    pub(crate) mat_layout: wgpu::BindGroupLayout,
    pub(crate) camera_pages: u32,
    pub(crate) shadow_layout: wgpu::BindGroupLayout,
    pub(crate) resolve_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) fused_resolve_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) post_source_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) control_resolve_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) gi_demand_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) ao_blur_horizontal_pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) ao_blur_vertical_pipeline: OnceLock<wgpu::ComputePipeline>,
}
impl CachedPipelines {
    /// Identity of the compilation requests, layouts and state this install builds.
    /// Source planning happens before registry lookup so lazy kernels cannot inherit
    /// a previous scene's unconsumed policy or omit a newly consumed policy.
    pub(crate) fn identity(&self,resolve:&str,geometry:&std::collections::HashSet<GeometryKey>,kernels:&[ProgramKernel])->Result<blake3::Hash> {
        enum Input {Geometry(GeometryKey),Compute(ProgramKernel)}
        let inputs:Vec<_>=geometry.iter().copied().map(Input::Geometry)
            .chain(kernels.iter().copied().map(Input::Compute)).collect();
        // Use the existing bounded compiler worker helper for independent
        // captures; serial frontends would delay all installation workers.
        let mut entries=crate::programs::parallel_map(&inputs,|input| {
            let (program,key)=match *input {
                Input::Geometry(key)=> {
                    let program=match key.1 {Role::ProgramControls {program,..}|Role::ProgramCoverage {program,..}=>program,_=>0};
                    (program,self.geometry_content_key(key,false)?)
                },
                Input::Compute(key)=> {
                    let program=if let ProgramKernel::Decoupled {stage:ShadeStage::EvaluateProgram(id),..}=key {id} else {0};
                    (program,self.compute_content_key(key,false)?)
                },
            };
            Ok((program,key))
        })?;
        entries.sort_by_key(|(program,key)|(*program,*key.as_bytes()));
        entries.dedup();
        let mut hash=blake3::Hasher::new();
        for (program,entry) in entries {hash.update(&program.to_le_bytes());hash.update(entry.as_bytes());}
        hash.update(resolve.as_bytes());
        // These layouts/state are constructed for the registry and consumed by
        // fixed passes and host bindings independently of material entries.
        for layout in [self.depth_identity,self.fragments_identity,self.fibre_identity,self.shadow_identity,self.resolve_identity] {hash.update(layout.as_bytes());}
        hash.update(format!("{:?}",self.control_targets).as_bytes());
        Ok(hash.finalize())
    }

    /// The pipeline for one drawn role, compiled once per registry. Different keys
    /// compile concurrently; the registry lock is held only to find the slot.
    pub(crate) fn geometry(&self, c: &Compiler, key: GeometryKey) -> Result<wgpu::RenderPipeline> {
        try_init(&self.slot(key), || self.build_geometry(c, key)).cloned()
    }
    fn slot(&self, key: GeometryKey) -> std::sync::Arc<OnceLock<wgpu::RenderPipeline>> {
        std::sync::Arc::clone(self.geometry.lock().unwrap().entry(key).or_default())
    }
    pub(crate) fn camera_slot(&self,key:ProgramKernel)->std::sync::Arc<OnceLock<wgpu::ComputePipeline>> {
        self.camera_evaluations.lock().unwrap().entry(key).or_default().clone()
    }
    pub(crate) fn content_key(&self,key:GeometryKey)->Option<blake3::Hash> {self.geometry_content_key(key,true).ok()}
    fn geometry_content_key(&self,key:GeometryKey,lookup_only:bool)->Result<blake3::Hash> {
        let mut hash=blake3::Hasher::new();
        hash.update(self.shader.geometry_input_key(key,lookup_only)?.as_bytes());
        hash.update(format!("{:?}",key.0).as_bytes());
        Ok(self.with_geometry_state(key.1,|_,layout,_,vs,fs,samples,write,targets| {
            hash.update(layout.as_bytes());
            hash.update(format!("{vs} {fs} {samples} {write} {targets:?} {:?}",self.shader.constants(key)).as_bytes());
            hash.finalize()
        }))
    }

    pub(crate) fn kernel_content_key(&self,kernel:ProgramKernel)->Option<blake3::Hash> {self.compute_content_key(kernel,true).ok()}
    fn compute_content_key(&self,kernel:ProgramKernel,lookup_only:bool)->Result<blake3::Hash> {
        let input=self.shader.kernel_input_key(kernel,lookup_only)?;
        let hits=||vec![crate::rt::hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE)];
        let probes=||crate::probes::read_entries(0,wgpu::ShaderStages::COMPUTE).to_vec();
        let groups=match kernel {
            ProgramKernel::Path {adaptive,..}|ProgramKernel::PathEnvironment {adaptive}=>path_groups(passes::path::layout_entries(adaptive),self.material_entries.clone(),crate::rt::hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE)),
            ProgramKernel::Gi {..}=>vec![Some(crate::rt::gi_entries()),Some(self.material_entries.clone()),self.probes.then(probes),Some(crate::rt::gi_read_entries())],
            ProgramKernel::Reflections {..}=>vec![Some(crate::rt::reflections_hits_entries()),Some(self.material_entries.clone()),self.probes.then(probes),Some(crate::rt::reflection_read_entries())],
            ProgramKernel::Probes {..}=>vec![Some(crate::probes::update_entries()),Some(self.material_entries.clone()),Some(probes()),Some(hits())],
            ProgramKernel::Decoupled {..}=>vec![Some(raster_global_entries()),Some(self.material_entries.clone()),Some(camera_shadow_entries_for(self.ray_shadows,self.program_trace)),Some(passes::decoupled::camera_geometry_entries_for(self.camera_pages)),Some(passes::decoupled::camera_entries(self.raster_samples))],
        };
        Ok(compute_pipeline_key(input,&groups))
    }

    /// Install the drawn keys. Warm variants (in the persistent cache) compile inline;
    /// cold ones draw with their shared superset until a background worker lands them.
    pub(crate) fn install_geometry(
        self: &std::sync::Arc<Self>,
        r: &Renderer,
        keys: std::collections::HashSet<GeometryKey>,
        mode: &VariantCompile,
    ) -> Result<std::collections::HashMap<GeometryKey, Drawn>> {
        let c = &r.compiler;
        let warm =
            |key: GeometryKey| self.slot(key).get().is_some() || c.is_known_input(self.content_key(key));
        // The shipped generic needs no Slang compilation. Even one cold key
        // starts behind it; compiling a scene-union fallback would block frame 1.
        let plan: Vec<_> = keys.iter().map(|&key| {
            let fallback = key.1.superset().and_then(|family| {
                let defer=mode.defer(warm(key),self.shader.generic_shipped() || warm((key.0,family)));
                defer.then_some((key.0, family))
            });
            (key, fallback)
        }).collect();
        let inline: std::collections::HashSet<_> = plan
            .iter()
            .map(|&(key, fallback)| fallback.unwrap_or(key))
            .collect();
        let inline:Vec<_>=inline.into_iter().collect();
        let cursor=std::sync::atomic::AtomicUsize::new(0);
        let compiled: std::collections::HashMap<_, _> = std::thread::scope(|scope| {
            (0..inline.len().min(std::thread::available_parallelism().map_or(1,usize::from)))
                .map(|_|scope.spawn(|| {
                    let mut pipelines=Vec::new();
                    loop {
                        let index=cursor.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
                        let Some(&key)=inline.get(index) else {break;};
                        pipelines.push((key,self.geometry(c,key)?));
                    }
                    Ok(pipelines)
                })).collect::<Vec<_>>().into_iter()
                .map(join).collect::<Result<Vec<_>>>()
        })?.into_iter().flatten().collect();
        plan.into_iter()
            .map(|(key, fallback)| {
                let this=self.clone();
                let drawn=c.install_variant(mode,self.slot(key),fallback.map(|f|compiled[&f].clone()),move |worker|this.build_geometry(worker,key))?;
                Ok((key, drawn))
            })
            .collect()
    }

    fn build_geometry(&self, c: &Compiler, key: GeometryKey) -> Result<wgpu::RenderPipeline> {
        let pipeline = self.build_role(c, key)?;
        c.remember(self.content_key(key));
        Ok(pipeline)
    }

    fn with_geometry_state<T>(&self,role:Role,consume:impl FnOnce(&wgpu::PipelineLayout,blake3::Hash,&str,&str,&str,u32,bool,&[Option<wgpu::ColorTargetState>])->T)->T {
        let fibre_near_targets=[Some(wgpu::ColorTargetState {format:wgpu::TextureFormat::Rg32Uint,blend:None,write_mask:wgpu::ColorWrites::ALL})];
        let fibre_targets=vec![Some(wgpu::ColorTargetState {format:wgpu::TextureFormat::Rgba16Float,
            blend:Some(wgpu::BlendState {color:wgpu::BlendComponent {src_factor:wgpu::BlendFactor::One,dst_factor:wgpu::BlendFactor::One,operation:wgpu::BlendOperation::Add},alpha:wgpu::BlendComponent {src_factor:wgpu::BlendFactor::One,dst_factor:wgpu::BlendFactor::One,operation:wgpu::BlendOperation::Add}}),write_mask:wgpu::ColorWrites::ALL});1];
        let fragment_targets = [Some(wgpu::ColorTargetState {
            format: crate::passes::decoupled::KEY_FORMAT, blend: None, write_mask: wgpu::ColorWrites::empty(),
        })];
        let visibility_targets = [Some(wgpu::ColorTargetState {
            format: crate::passes::decoupled::KEY_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL,
        })];
        let depth = if self.software_raster {(&self.fragments_pipeline_layout,self.fragments_identity)} else {(&self.depth_layout,self.depth_identity)};
        let samples = self.raster_samples;
        let (layout, label, vs, fs, samples, write, targets): (
            _,
            _,
            _,
            _,
            _,
            _,
            &[Option<wgpu::ColorTargetState>],
        ) = match role {
            Role::Fibre(stage) => (
                if stage.additive() {(&self.fibre_layout,self.fibre_identity)} else {(&self.depth_layout,self.depth_identity)},
                "fibre transport capture","shadow_vs",stage.entry(),1,
                !stage.additive(),
                if stage.additive() {&fibre_targets} else if stage==crate::fibre_transport::Stage::Near {&fibre_near_targets} else {&[]}
            ),
            Role::OpaqueDepth => (
                depth,
                "packed opaque depth",
                if self.software_raster {"software_depth_vs"} else {"opaque_depth_vs"},
                if self.software_raster {"software_depth_fs"} else {""},
                samples,
                true,
                &[],
            ),
            Role::OpaqueSingleDepth => (
                depth,
                "packed single-sided opaque depth",
                if self.software_raster {"software_depth_vs"} else {"opaque_single_depth_vs"},
                if self.software_raster {"software_single_depth_fs"} else {"opaque_single_depth_fs"},
                samples,
                true,
                &[],
            ),
            Role::MaskedDepth {..} | Role::ProgramCoverage {stage:CoverageStage::Depth,..} => (
                depth,
                "packed depth",
                "main_vs",
                if role==(Role::MaskedDepth {hashed:true}) {"hashed_depth_fs"} else {"depth_fs"},
                samples,
                true,
                &[],
            ),
            Role::Visibility | Role::VisibilitySingle | Role::VisibilityMasked | Role::ProgramCoverage {stage:CoverageStage::Visibility,..} => (
                (&self.fragments_pipeline_layout,self.fragments_identity),
                "decoupled visibility",
                if matches!(role,Role::VisibilityMasked | Role::ProgramCoverage {stage:CoverageStage::Visibility,..}) { "masked_vs" } else { "vs" },
                match role { Role::Visibility => "fs", Role::VisibilitySingle => "single_fs", _ => "masked_fs" },
                1,
                true,
                &visibility_targets,
            ),
            // In the visibility pass after its opaque keys, writing no key.
            Role::TransparentFragments => (
                (&self.fragments_pipeline_layout,self.fragments_identity),
                "decoupled transparent fragments",
                "masked_vs",
                if self.field_controls || self.software_raster {"transparent_fragments_fs"} else {"transparent_triangles_fs"},
                1,
                false,
                &fragment_targets,
            ),
            Role::Controls(_) | Role::ProgramControls {transparent:false,..} => (
                if self.field_controls || self.software_raster {(&self.fragments_pipeline_layout,self.fragments_identity)} else {depth},
                "packed opaque controls",
                "main_vs",
                "controls_fs",
                samples,
                self.field_controls,
                &self.control_targets,
            ),
            Role::TransparentControls(_) | Role::ProgramControls {transparent:true,..} => (
                if self.field_controls || self.software_raster {(&self.fragments_pipeline_layout,self.fragments_identity)} else {depth},
                "packed controls",
                "main_vs",
                "transparent_controls_fs",
                samples,
                true,
                &self.control_targets,
            ),
        };
        consume(layout.0,layout.1,label,vs,fs,samples,write,targets)
    }

    fn build_role(&self,c:&Compiler,(packed,role):GeometryKey)->Result<wgpu::RenderPipeline> {
        self.with_geometry_state(role,|layout,_,label,vs,fs,samples,write,targets| {
        let shading = role.shading();
        let variant_label = if shading & crate::materials::FEATURE_SPECIALIZED != 0 {
            format!("{label} key={shading:016x}")
        } else {
            label.to_owned()
        };
        Ok(packed_geometry_pipeline(
            c,
            &self.shader.modules((packed, role))?,
            &self.shader.constants((packed, role)),
            layout,
            &variant_label,
            vs,
            fs,
            samples,
            write,
            targets,
            packed,
        ))
        })
    }

    // Slang registration: ("clusters", "main", "compute"). Per-file discovery
    // registers shaders/clusters.slang; no shared build registry edit is needed.
    fn build_cluster_pipeline(&self, r: &Renderer) -> wgpu::ComputePipeline {
        let entries = cluster_entries();
        let bind_layout = r
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("cluster light culling"),
                entries: &entries,
            });
        let layout = r
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cluster light culling"),
                bind_group_layouts: &[Some(&bind_layout)],
                immediate_size: 0,
            });
        r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cluster light culling"),
            layout: Some(&layout),
            module: self
                .cluster_module
                .get_or_init(|| native::shader_module(&r.device, &native::CLUSTERS_CULL)),
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: r.cache.as_ref(),
        })
    }
    fn build_deform_pipeline(&self, r: &Renderer) -> wgpu::ComputePipeline {
        passes::deform::pipeline(r, &self.deform_module)
    }
    pub(crate) fn ensure_assets(&self, r: &Renderer, plan: &crate::plan::ScenePlan) -> Result<()> {
        std::thread::scope(|scope| {
            let compute_jobs: [(_, _, fn(&Self, &Renderer) -> wgpu::ComputePipeline); 2] = [
                (
                    &self.cluster_pipeline,
                    plan.cluster,
                    Self::build_cluster_pipeline,
                ),
                (
                    &self.deform_pipeline,
                    plan.deform,
                    Self::build_deform_pipeline,
                ),
            ];
            joined(compute_jobs.map(|(slot, needed, build)| compile(scope, slot, needed, move || Ok(build(self, r)))))
        })
    }

    /// The path transport module and entry, with the scene's programs at hits.
    pub(crate) fn program_kernel(&self, kernel: ProgramKernel) -> Result<(wgpu::ShaderModule, &'static str)> {
        self.shader.program_kernel(kernel)
    }
    /// The kernel an install draws for `kernel`. Kernels specialized to the scene's
    /// material features have a superset key that shades every material the same way
    /// and stores the scene's evaluation record;
    /// a one-frame install draws with the superset when `vork setup` stored it and the
    /// specialization is not stored, instead of compiling (a still has no later frame
    /// to use the specialization). Setup compiles the superset. Every other install
    /// specializes: a cold specialization compiles faster than the superset. The
    /// lookups are content-addressed, so a superset of a scene setup never saw
    /// (other programs or inputs) is simply absent.
    /// Shipped late-specialization artifacts exist for every capability key;
    /// prefer a remembered device superset before consulting artifact presence.
    /// Hit consumers retain a Drawn slot so multi-frame installs can promote
    /// this fallback after their exact pipeline compiles in the background.
    /// Path transport has a prebuilt engine instead of a superset to compile: it
    /// needs no Slang, and its driver pipeline is the same for every scene, so
    /// setup and earlier renders leave it warm. A path kernel whose own pipeline
    /// is not warm draws the engine when the engine's is, and compiles behind it
    /// (install_behind_engine); a longer render takes its kernel at an exposure
    /// boundary. A cold engine is never waited for: building its driver pipeline
    /// costs as much as the scene's own kernel, so the scene compiles that, as it
    /// always did. Setup is what builds the engine.
    pub(crate) fn drawn_kernel(&self, kernel: ProgramKernel, mode: &VariantCompile, compiler: &Compiler) -> ProgramKernel {
        if let Ok(engine) = self.shader.path_engine(kernel) {
            let warm = |kernel| compiler.is_known_input(self.kernel_content_key(kernel));
            return match mode {
                VariantCompile::Inline => kernel,
                VariantCompile::Setup | VariantCompile::SetupPlain | VariantCompile::Gated(_) => engine,
                VariantCompile::Background | VariantCompile::Fallback =>
                    if !warm(kernel) && warm(engine) { engine } else { kernel },
            };
        }
        if matches!(mode, VariantCompile::Background | VariantCompile::Fallback | VariantCompile::Gated(_))
            && !compiler.is_known_input(self.kernel_content_key(kernel))
        {
            if let Some(plain) = kernel.plain_hit().filter(|plain| compiler.is_known_input(self.kernel_content_key(*plain))) {
                return plain;
            }
        }
        let Some(superset) = kernel.superset() else { return kernel };
        let stored = |kernel| self.shader.kernel_stored(kernel).unwrap_or(false);
        match mode {
            VariantCompile::Setup => superset,
            VariantCompile::SetupPlain => kernel.plain_hit().unwrap_or(superset),
            VariantCompile::Fallback if !compiler.is_known_input(self.kernel_content_key(kernel))
                && compiler.is_known_input(self.kernel_content_key(superset)) => superset,
            VariantCompile::Fallback if !stored(kernel) && stored(superset) => superset,
            _ => kernel,
        }
    }
    /// Hit keys, path transport included, use the existing deferred compiler and
    /// retained compute slots.
    /// A warm setup fallback must not permanently select the movie's evaluator.
    pub(crate) fn install_hit(
        self: &std::sync::Arc<Self>, r: &Renderer, requested: ProgramKernel,
        mode: &VariantCompile, layout: wgpu::PipelineLayout, label: &'static str,
    ) -> Result<Drawn<wgpu::ComputePipeline>> {
        Ok(self.install_hit_selected(r, requested, mode, layout, label)?.0)
    }
    /// `install_hit`, with the kernel it chose to draw first. The choice reads the
    /// known-pipeline ledger, which background workers write, so it is made once
    /// here: what is installed and what a caller reports cannot disagree.
    pub(crate) fn install_hit_selected(
        self: &std::sync::Arc<Self>, r: &Renderer, requested: ProgramKernel,
        mode: &VariantCompile, layout: wgpu::PipelineLayout, label: &'static str,
    ) -> Result<(Drawn<wgpu::ComputePipeline>, ProgramKernel)> {
        if crate::specialization::enabled() && r.cache.is_some()
            && matches!(mode, VariantCompile::Background | VariantCompile::Fallback)
            && self.shader.path_engine(requested).is_ok()
        {
            let key = self.compute_content_key(requested, false)?;
            let path = r.disk_cache.path("specialization", 1, key);
            if let Some(cache) = crate::specialization::load(&r.device, &path) {
                let (module, entry) = self.program_kernel(requested)?;
                let pipeline = r.compiler.compute(&wgpu::ComputePipelineDescriptor {
                    label: Some(label), layout: Some(&layout), module: &module, entry_point: Some(entry),
                    compilation_options: Default::default(), cache: Some(&cache),
                });
                return Ok((Drawn::inline(pipeline), requested));
            }
            let selected = self.drawn_kernel(requested, mode, &r.compiler);
            if selected != requested {
                let ProgramKernel::Path {adaptive,..} = requested else {unreachable!()};
                let source = self.shader.detached_request(requested)?;
                let groups = path_groups(passes::path::layout_entries(adaptive), self.material_entries.clone(), crate::rt::hit_buffers_entry(0,wgpu::ShaderStages::COMPUTE));
                if let Some(ended) = crate::specialization::start(key, path.clone(), source, groups)? {
                    let (module, entry) = self.program_kernel(selected)?;
                    let engine = r.compiler.compute(&wgpu::ComputePipelineDescriptor {
                        label: Some(label), layout: Some(&layout), module: &module, entry_point: Some(entry),
                        compilation_options: Default::default(), cache: r.compiler.cache.as_ref(),
                    });
                    let pipelines = self.clone();
                    let drawn = r.compiler.install_behind_engine(mode, self.camera_slot(requested), engine, move |compiler| {
                        let cache = crate::specialization::wait(&compiler.device, &path, &ended)?;
                        let (module, entry) = pipelines.program_kernel(requested)?;
                        Ok(compiler.compute(&wgpu::ComputePipelineDescriptor {
                            label: Some(label), layout: Some(&layout), module: &module, entry_point: Some(entry),
                            compilation_options: Default::default(), cache: Some(&cache),
                        }))
                    });
                    return Ok((drawn, selected));
                }
            }
        }
        let selected = self.drawn_kernel(requested, mode, &r.compiler);
        #[cfg(test)]
        r.compiler.hit_selections.lock().unwrap().push((requested, selected));
        let build = |compiler: &Compiler, kernel| -> Result<wgpu::ComputePipeline> {
            let (module, entry) = self.program_kernel(kernel)?;
            let pipeline = compiler.compute(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&layout), module: &module,
                entry_point: Some(entry), compilation_options: Default::default(),
                cache: compiler.cache.as_ref(),
            });
            compiler.remember(self.kernel_content_key(kernel));
            Ok(pipeline)
        };
        let fallback = (selected != requested).then(|| build(&r.compiler, selected)).transpose()?;
        let pipelines = self.clone();
        let own = move |compiler: &Compiler| -> Result<wgpu::ComputePipeline> {
            let (module, entry) = pipelines.program_kernel(requested)?;
            let pipeline = compiler.compute(&wgpu::ComputePipelineDescriptor {
                label: Some(label), layout: Some(&layout), module: &module,
                entry_point: Some(entry), compilation_options: Default::default(),
                cache: compiler.cache.as_ref(),
            });
            compiler.remember(pipelines.kernel_content_key(requested));
            Ok(pipeline)
        };
        // A render's kernel behind its prebuilt engine always compiles, a still's
        // too: otherwise repeats of the scene would stay at the engine's speed.
        // Setup keeps the engine alone; it has no scene to come back to.
        let behind_engine = selected != requested && !matches!(mode, VariantCompile::Setup | VariantCompile::SetupPlain)
            && self.shader.path_engine(requested).ok() == Some(selected);
        let drawn = match fallback {
            Some(engine) if behind_engine => r.compiler.install_behind_engine(mode, self.camera_slot(requested), engine, own),
            fallback => r.compiler.install_variant(mode, self.camera_slot(requested), fallback, own)?,
        };
        Ok((drawn, selected))
    }
    /// The output layout and dispatch must use the same planned resolve, including
    /// encoder surfaces attached after frame-slot creation.
    pub(crate) fn output_resolve(&self, effects: EffectPlan) -> &OnceLock<wgpu::ComputePipeline> {
        if effects.fuse_controls() {
            &self.fused_resolve_pipeline
        } else {
            &self.resolve_pipeline
        }
    }

    pub(crate) fn ensure_output(&self, r: &Renderer, effects: EffectPlan) -> Result<()> {
        let entry = if effects.fuse_controls() {
            if self.raster_samples > 1 { "resolve_fused" } else { "resolve_fused_single" }
        } else { "main" };
        try_init(self.output_resolve(effects), || Ok(passes::resolve::pipeline(
            r, &self.resolve_module, entry, Some(&self.resolve_layout),
        )))?;
        Ok(())
    }

    pub(crate) fn ensure_effects(&self, r: &Renderer, features: EffectPlan) -> Result<()> {
        let ao_blur_layout = crate::passes::ao::slang_layout(&r.device);
        let gi_demand_layout = crate::passes::ao::demand_layout(&r.device);
        enum Module {
            Slang(&'static crate::native::Compiled),
            Resolve,
        }
        let jobs = [
            (
                &self.gi_demand_pipeline,
                features.indirect,
                Module::Slang(&crate::native::GI_DEMAND_BUILD),
                Some(&gi_demand_layout),
                "build",
                "GI exact consumer demand",
            ),
            (
                &self.fused_resolve_pipeline,
                features.fuse_controls() && features.outputs.is_none_or(|o|o.beauty),
                Module::Resolve,
                Some(&self.resolve_layout),
                "main",
                "fused control and output resolve",
            ),
            (
                &self.resolve_pipeline,
                !features.fuse_controls() && features.outputs.is_none_or(|o|o.beauty),
                Module::Resolve,
                Some(&self.resolve_layout),
                "main",
                "resolve",
            ),
            (
                &self.post_source_pipeline,
                (features.compositor || features.denoise) && features.outputs.is_none_or(|o|o.beauty),
                Module::Resolve,
                Some(&self.resolve_layout),
                "post_source",
                "scene-linear post source",
            ),
            (
                &self.ao_blur_horizontal_pipeline,
                features.indirect,
                Module::Slang(&crate::native::AO_BLUR_HORIZONTAL),
                Some(&ao_blur_layout),
                "horizontal",
                "AO per-sample horizontal reconstruction",
            ),
            (
                &self.ao_blur_vertical_pipeline,
                features.indirect,
                Module::Slang(&crate::native::AO_BLUR_VERTICAL),
                Some(&ao_blur_layout),
                "vertical",
                "AO per-sample vertical reconstruction",
            ),
        ];
        std::thread::scope(|scope| {
            scope.spawn(|| {
                if features.outputs.is_none_or(|o|o.beauty || o.controls() || o.aovs || o.catcher) {
                    environment::Environment::pipeline(r);
                }
            });
            let fixed = [
                // Implicit path writes controls itself; implicit raster still
                // schedules Controls, just like graph::Graph's output pruning.
                compile(scope, &self.control_resolve_pipeline, features.standalone_controls(features.outputs), || {
                let shader = control_resolve_shader(self.raster_samples, self.denoise);
                let module = self
                    .control_module
                    .get_or_init(|| crate::native::shader_module(&r.device, shader));
                let entries = control_resolve_entries(self.raster_samples, self.denoise);
                let group = r
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("control resolve"),
                        entries: &entries,
                    });
                let layout = r
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("control resolve"),
                        bind_group_layouts: &[Some(&group)],
                        immediate_size: 0,
                    });
                Ok(r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("single-sample controls"),
                    layout: Some(&layout),
                    module,
                    entry_point: Some(shader.entry),
                    compilation_options: Default::default(),
                    cache: r.cache.as_ref(),
                }))
                }),
            ];
            let descriptors = jobs.map(|(slot, needed, module, layout, entry, label)| {
                compile(scope, slot, needed, move || {
                    if matches!(&module, Module::Resolve) {
                        let entry=if label == "fused control and output resolve" {
                            if self.raster_samples > 1 {"resolve_fused"} else {"resolve_fused_single"}
                        } else {entry};
                        return Ok(passes::resolve::pipeline(r,&self.resolve_module,entry,layout));
                    }
                    let native_module;
                    let module = match module {
                        Module::Resolve => unreachable!(),
                        Module::Slang(shader) => {
                            native_module = crate::native::shader_module(&r.device, shader);
                            &native_module
                        }
                    };
                    Ok(r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout,
                        module,
                        entry_point: Some(entry),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            constants: if label == "fused control and output resolve" { &[("FUSED_CONTROLS", 1.)] } else { &[] },
                            ..Default::default()
                        },
                        cache: r.cache.as_ref(),
                    }))
                })
            });
            joined(fixed.into_iter().chain(descriptors))
        })
    }
    pub(crate) fn with_sources(
        r: &Renderer,
        resolve: &str,
        raster_samples: u32,
        ray_shadows: bool,
        slang: crate::slang_forward::SlangScene,
    ) -> Self {
        let d = &r.device;
        let globals=raster_global_entries();
        let material_entries=material_entries_for(slang.textures,slang.samplers);
        let capture_entries=crate::fibre_transport::capture_entries().to_vec();
        let fragments_entries=crate::passes::decoupled::fragments_entries();
        let fragments_geometry_entries=crate::passes::decoupled::fragments_geometry_entries();
        let global_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame + objects"),
            entries: &globals,
        });
        let mat_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &material_entries,
        });
        let shadow_entries = camera_shadow_entries_for(ray_shadows,slang.program_trace);
        let shadow_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow read"),
            entries: &shadow_entries,
        });
        let fibre_capture_layout=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("fibre capture inputs"),entries:&capture_entries});
        let fibre_layout=d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("fibre transport capture"),bind_group_layouts:&[Some(&global_layout),Some(&mat_layout),Some(&fibre_capture_layout)],immediate_size:0});
        // Depth and controls read only frame/object + material tables.
        let depth_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("packed depth layout"),
            bind_group_layouts: &if slang.program_trace {vec![Some(&global_layout),Some(&mat_layout),Some(&shadow_layout)]} else {vec![Some(&global_layout),Some(&mat_layout)]},
            immediate_size: 0,
        });
        let fragments_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transparent fragments"),
            entries: &fragments_entries,
        });
        let fragments_geometry_layout=d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("transparent opacity geometry"),entries:&fragments_geometry_entries});
        let fragments_pipeline_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("decoupled transparent fragments"),
            bind_group_layouts: &if slang.program_trace {vec![Some(&global_layout),Some(&mat_layout),Some(&fragments_layout),Some(&fragments_geometry_layout),Some(&shadow_layout)]} else {vec![Some(&global_layout),Some(&mat_layout),Some(&fragments_layout),Some(&fragments_geometry_layout)]},
            immediate_size: 0,
        });

        let denoise=slang.denoise;
        let camera_pages=crate::passes::decoupled::evaluation_pages(&d.limits(),slang.aov_count);
        // The controls pass writes ambient, IDs, normals, motion (and denoising albedo);
        // colour comes from shading.
        let control_targets: Vec<_> = control_formats(denoise).iter().map(|format|
            format.map(wgpu::ColorTargetState::from)).collect();
        let resolve_module = resolve.to_owned();
        let resolve_entries = passes::resolve::layout_entries(
            if raster_samples > 1 { "resolve_color" } else { "resolve_fused_single" },
        );
        let output_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: r.output_format(),
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let output_entries=vec![output_entry];
        let output_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("encoder surface"),
            entries: &output_entries,
        });
        let resolve_bind_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resolve and motion resources"),
            entries: &resolve_entries,
        });
        let resolve_layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resolve and motion layout"),
            bind_group_layouts: &[Some(&resolve_bind_layout), Some(&output_layout)],
            immediate_size: 0,
        });

        let scene_groups=vec![Some(globals),Some(material_entries.clone())];
        let mut depth_groups=scene_groups.clone();
        if slang.program_trace {depth_groups.push(Some(shadow_entries.clone()));}
        let mut fragment_groups=scene_groups.clone();
        fragment_groups.extend([Some(fragments_entries),Some(fragments_geometry_entries)]);
        if slang.program_trace {fragment_groups.push(Some(shadow_entries.clone()));}
        let mut fibre_groups=scene_groups;
        fibre_groups.push(Some(capture_entries));
        Self {
            material_entries,
            probes: slang.probes,
            decoupled_merge: OnceLock::new(),
            raster_samples,
            denoise,
            depth_identity: layout_identity(&depth_groups),
            fragments_identity: layout_identity(&fragment_groups),
            fibre_identity: layout_identity(&fibre_groups),
            shadow_identity: layout_identity(&[Some(shadow_entries)]),
            resolve_identity: layout_identity(&[Some(resolve_entries),Some(output_entries)]),
            ray_shadows,
            field_controls: slang.field_controls,
            software_raster: slang.software_raster,
            program_trace: slang.program_trace,
            // Every raster role uses native Slang.
            shader: Box::new(crate::slang_forward::SlangForward::new(
                r,
                ray_shadows,
                raster_samples > 1,
                slang,
            )),
            cluster_module: OnceLock::new(),
            deform_module: OnceLock::new(),
            depth_layout,
            fragments_layout,
            fragments_geometry_layout,
            fragments_pipeline_layout,
            fibre_layout,
            fibre_capture_layout,
            geometry: Mutex::default(),
            camera_evaluations: Mutex::default(),
            control_targets,
            resolve_module,
            control_module: OnceLock::new(),
            resolve_layout,
            deform_pipeline: OnceLock::new(),
            cluster_pipeline: OnceLock::new(),
            camera_cull_pipeline: OnceLock::new(),
            global_layout,
            mat_layout,
            camera_pages,
            shadow_layout,
            resolve_pipeline: OnceLock::new(),
            fused_resolve_pipeline: OnceLock::new(),
            post_source_pipeline: OnceLock::new(),
            control_resolve_pipeline: OnceLock::new(),
            gi_demand_pipeline: OnceLock::new(),
            ao_blur_horizontal_pipeline: OnceLock::new(),
            ao_blur_vertical_pipeline: OnceLock::new(),
        }
    }
}
/// Group 0 of every geometry pass (shaders/frame.slang): frame, objects, lights,
/// environment, LTC tables and clusters.
/// Scene globals (frame.slang), read by raster stages and the compute kernels that shade surfaces.
pub(crate) fn raster_global_entries()->Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries=global_entries();
    entries.push(buffer_entry(11,wgpu::ShaderStages::VERTEX,wgpu::BufferBindingType::Storage {read_only:true}));
    entries
}
pub(crate) fn global_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = vec![
                buffer_entry(
                    0,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                ),
                buffer_entry(
                    1,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                buffer_entry(
                    2,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                buffer_entry(
                    7,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Uniform,
                ),
                buffer_entry(
                    8,
                    wgpu::ShaderStages::FRAGMENT,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                texture_entry(
                    3,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
                sampler_entry(4, false),
                texture_entry(
                    5,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
                texture_entry(
                    6,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
                texture_entry(
                    9,
                    wgpu::TextureSampleType::Float { filterable: true },
                    false,
                ),
            ];
    entries.push(buffer_entry(10, wgpu::ShaderStages::FRAGMENT, wgpu::BufferBindingType::Storage {read_only:true}));
    for e in &mut entries {e.visibility |= wgpu::ShaderStages::COMPUTE;}
    entries
}
/// Group 1: records plus scene-sized material texture and sampler tables. Ray-traced AO also reads it from compute for alpha-tested hits.
pub(crate) fn material_entries(_rays: bool) -> Vec<wgpu::BindGroupLayoutEntry> {material_entries_for(1,1)}
/// Per-stage resource slots a pipeline layout binds, as Metal counts them
/// (binding arrays use one buffer slot for their argument table).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(crate) struct StageResources {pub buffers:u32,pub textures:u32,pub samplers:u32}
/// The tightest per-stage limits of the backends we ship: Metal.
pub(crate) const METAL_STAGE_LIMITS:StageResources=StageResources {buffers:31,textures:128,samplers:16};
pub(crate) fn stage_resources(groups:&[&[wgpu::BindGroupLayoutEntry]],stage:wgpu::ShaderStages)->StageResources {
    let mut r=StageResources {buffers:0,textures:0,samplers:0};
    let mut sizes = stage == wgpu::ShaderStages::VERTEX;
    for e in groups.iter().flat_map(|g|g.iter()).filter(|e|e.visibility.contains(stage)) {
        sizes |= matches!(e.ty,wgpu::BindingType::Buffer {ty:wgpu::BufferBindingType::Storage {..},..});
        let [buffers,textures,samplers]=e.ty.metal_resource_slots(e.count);
        r.buffers+=u32::from(buffers);
        r.textures+=u32::from(textures);
        r.samplers+=u32::from(samplers);
    }
    r.buffers += u32::from(sizes);
    r
}
/// Fails when a pipeline layout would exceed a shipped backend's per-stage limits, on any host.
pub(crate) fn check_stage_limits(label:&str,groups:&[&[wgpu::BindGroupLayoutEntry]],stage:wgpu::ShaderStages)->anyhow::Result<()> {
    let r=stage_resources(groups,stage);let m=METAL_STAGE_LIMITS;
    anyhow::ensure!(r.buffers<=m.buffers && r.textures<=m.textures && r.samplers<=m.samplers,
        crate::gpu_memory::DeviceLimit(format!("pipeline layout {label} binds {} buffers, {} textures and {} samplers to one stage; Metal allows {}, {} and {}",r.buffers,r.textures,r.samplers,m.buffers,m.textures,m.samplers)));
    Ok(())
}
/// Scene-sized scalar storage layouts must leave room for uniforms and Metal's
/// implicit sizes buffer, in addition to the device's storage-only limit.
pub(crate) fn scalar_storage_limit(limits:&wgpu::Limits,metal:bool,uniforms:u32)->u32 {
    let native=if metal {METAL_STAGE_LIMITS.buffers.saturating_sub(uniforms+1)} else {u32::MAX};
    limits.max_storage_buffers_per_shader_stage.min(native)
}
pub(crate) fn material_entries_for(textures:u32,samplers:u32) -> Vec<wgpu::BindGroupLayoutEntry> {
    let textures=textures.max(vork_slang::registry::MATERIAL_TEXTURE_CAPACITY);
    let samplers=samplers.max(vork_slang::registry::MATERIAL_SAMPLER_CAPACITY);
    use crate::binding_abi::{MATERIAL, Resource};
    let visibility = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    MATERIAL.iter().enumerate().map(|(binding, description)| {
        let binding=binding as u32;
        let (ty,count)=match description.resource {
            Resource::ReadBuffer(_) | Resource::WriteBuffer(_) => (wgpu::BindingType::Buffer {
                ty:wgpu::BufferBindingType::Storage {read_only:matches!(description.resource,Resource::ReadBuffer(_))},
                has_dynamic_offset:false,min_binding_size:None,
            },None),
            Resource::Textures=>(texture_entry(binding,wgpu::TextureSampleType::Float {filterable:true},false).ty,std::num::NonZeroU32::new(textures)),
            Resource::Samplers=>(sampler_entry(binding,false).ty,std::num::NonZeroU32::new(samplers)),
        };
        wgpu::BindGroupLayoutEntry {binding,visibility: if binding == 3 {visibility | wgpu::ShaderStages::VERTEX} else {visibility},ty,count}
    }).collect()
}
/// Group 2: immutable emitter-sampling rows.
pub(crate) fn shadow_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    vec![crate::shadows::sequence_entry()]
}
/// Stable camera resource table. Optional consumers bind empty records.
pub(crate) fn camera_shadow_entries(rays:bool)->Vec<wgpu::BindGroupLayoutEntry> {
    camera_shadow_entries_for(rays,false)
}
fn camera_shadow_entries_for(rays:bool,trace:bool)->Vec<wgpu::BindGroupLayoutEntry> {
    let stages=if trace {wgpu::ShaderStages::COMPUTE|wgpu::ShaderStages::FRAGMENT} else {wgpu::ShaderStages::COMPUTE};
    let mut entries=shadow_entries();

    if rays {
        entries.extend(crate::rt::layout_entries(1,stages));
        entries.push(crate::rt::hit_buffers_entry(3,stages));
    } else {
        entries.extend((1..4).map(|i|buffer_entry(i,stages,wgpu::BufferBindingType::Storage {read_only:true})));
    }
    entries.extend(crate::fibre_transport::read_entries(4));
    entries.extend(crate::probes::read_entries(9,stages));

    for e in &mut entries {e.visibility=stages;}
    entries
}
pub(crate) fn buffer_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    ty: wgpu::BufferBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
pub(crate) fn environment_texture_entry(binding:u32)->wgpu::BindGroupLayoutEntry {
    let mut entry=texture_entry(binding,wgpu::TextureSampleType::Uint,false);
    if let wgpu::BindingType::Texture {view_dimension,..}=&mut entry.ty {*view_dimension=wgpu::TextureViewDimension::D2Array;}
    entry
}

pub(crate) fn texture_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    multisampled: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled,
        },
        count: None,
    }
}
pub(crate) fn sampler_entry(binding: u32, comparison: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(if comparison {
            wgpu::SamplerBindingType::Comparison
        } else {
            wgpu::SamplerBindingType::Filtering
        }),
        count: None,
    }
}
fn packed_geometry_pipeline(
    c: &Compiler,
    [vertex, fragment]: &[wgpu::ShaderModule; 2],
    constants: &[(&str, f64)],
    layout: &wgpu::PipelineLayout,
    label: &str,
    vs: &str,
    fs: &str,
    samples: u32,
    write: bool,
    targets: &[Option<wgpu::ColorTargetState>],
    packed: crate::geometry::VertexLayout,
) -> wgpu::RenderPipeline {
    // Every vertex stage but the fibre captures' is a camera stage (camera-instance remap).
    let camera = vs != "shadow_vs";
    let mut attributes = packed.attributes().to_vec();

    // wgpu compacts empty vertex layouts. Put the instance stream before the
    // optional constant stream, keeping every active binding contiguous.
    if camera {attributes.insert(3,vec![wgpu::VertexAttribute {shader_location:vork_slang::registry::CAMERA_INSTANCE_LOCATION,offset:0,format:wgpu::VertexFormat::Uint32}]);}
    // Decoupled visibility reads its pass's parameter block as a constant stream after it.
    let visibility = matches!(fs, "fs" | "single_fs" | "masked_fs" | "transparent_fragments_fs" | "transparent_triangles_fs");
    if visibility {attributes.insert(4,vec![wgpu::VertexAttribute {shader_location:vork_slang::registry::VISIBILITY_PASS_LOCATION,offset:0,format:wgpu::VertexFormat::Uint32x4},wgpu::VertexAttribute {shader_location:vork_slang::registry::VISIBILITY_LENS_LOCATION,offset:16,format:wgpu::VertexFormat::Float32x2}]);}
    let buffers: Vec<_> = attributes
        .iter()
        .enumerate()
        .map(|(slot, attributes)| wgpu::VertexBufferLayout {
            array_stride: if packed.points && slot<3 {0} else {match slot {
                0 => packed.vertex_stride(),
                1 => packed.vertex_stride() * if packed.deform {2} else {1},
                2 => {
                    if packed.constant_attributes() {
                        0
                    } else {
                        packed.attribute_stride()
                    }
                }
                3 if camera => 4,
                _ => 0,
            }},
            step_mode: if camera && slot==3 {wgpu::VertexStepMode::Instance} else {wgpu::VertexStepMode::Vertex},
            attributes,
        })
        .collect();
    let buffer_count = attributes.iter().rposition(|a| !a.is_empty()).unwrap() + 1;
    c.render(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vertex,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers: &buffers[..buffer_count],
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(write),
            depth_compare: Some(if fs=="fibre_far_fs" {
                wgpu::CompareFunction::Greater
            } else if fs=="fibre_accumulate_fs" {
                wgpu::CompareFunction::Always
            } else if write && fs=="controls_fs" {
                wgpu::CompareFunction::LessEqual
            } else if write {
                wgpu::CompareFunction::Less
            } else if matches!(fs,"transparent_fs"|"transparent_receiver_fs"|"transparent_fragments_fs"|"transparent_triangles_fs") {
                wgpu::CompareFunction::LessEqual
            } else {
                wgpu::CompareFunction::Equal
            }),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: samples,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: (!fs.is_empty()).then_some(
            wgpu::FragmentState {
                module: fragment,
                entry_point: Some(fs),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                targets,
            },
        ),
        multiview_mask: None,
        cache: c.cache.as_ref(),
    })
}

#[cfg(test)]
mod control_resolve_tests {
    use super::*;

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn composited_previs_subsurface_installs_control_resolve() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let features = EffectPlan {
            compositor: true, subsurface: true,
            outputs: Some(crate::compositor::Dependencies {
                color: true, beauty: true, ..Default::default()
            }),
            ..Default::default()
        };
        let pipelines = renderer.cached_pipelines(1);
        pipelines.ensure_effects(&renderer, features)?;
        assert!(pipelines.control_resolve_pipeline.get().is_some(),
            "subsurface composition consumes controls even without GI, reflections or denoising");
        Ok(())
    }
}

#[cfg(all(test, vork_private_tests))]
include!("pipelines_tests_private.rs");

/// Control resolve for the raster coverage sample count (msaa4 or single-sample jitter).

/// Native single-sample/MSAA entries, with optional denoising guide output.
pub(crate) fn control_resolve_shader(samples: u32, denoise: bool) -> &'static crate::native::Compiled {
    match (samples, denoise) {
        (1, false) => &crate::native::CONTROL_RESOLVE_SINGLE_RESOLVE,
        (1, true) => &crate::native::CONTROL_RESOLVE_SINGLE_RESOLVE_DENOISE,
        (crate::constants::RASTER_MSAA_SAMPLES, false) => &crate::native::CONTROL_RESOLVE_MSAA_RESOLVE,
        (crate::constants::RASTER_MSAA_SAMPLES, true) => &crate::native::CONTROL_RESOLVE_MSAA_RESOLVE_DENOISE,
        _ => panic!("unsupported control resolve sample count: {samples}"),
    }
}

/// Single-sample controls copy an RGBA16F texel; MSAA RGB averages retain f32.
pub(crate) fn resolved_ambient_format(samples: u32) -> wgpu::TextureFormat {
    if samples == 1 { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba32Float }
}

pub(crate) fn control_resolve_entries(samples: u32, denoise: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    use wgpu::{BindingType, TextureFormat as Format, TextureSampleType as Sample};
    let mut entries = Vec::new();
    for (binding, sample_type) in [
        Sample::Depth,
        Sample::Uint,
        Sample::Float { filterable: false },
        Sample::Float { filterable: false },
        Sample::Float { filterable: false },
    ]
    .into_iter()
    .enumerate()
    {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: samples > 1,
            },
        });
    }
    for (index, format) in [
        Format::Rg32Float,
        Format::R32Uint,
        Format::Rgba16Float,
        resolved_ambient_format(samples),
        Format::Rgba16Float,
    ]
    .into_iter()
    .enumerate()
    {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5 + index as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            count: None,
            ty: BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
        });
    }
    entries.push(buffer_entry(10, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Uniform));
    if denoise {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 11, visibility: wgpu::ShaderStages::COMPUTE, count: None,
            ty: BindingType::Texture { sample_type: Sample::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: samples > 1 },
        });
        entries.push(buffer_entry(12, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: false }));
    }
    entries
}

#[cfg(test)]
mod raster_control_tests {
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn raster_coverage_counts_have_distinct_shared_registry_entries() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let single = r.cached_pipelines(1);
        let msaa = r.cached_pipelines(crate::constants::RASTER_MSAA_SAMPLES);
        assert!(!std::sync::Arc::ptr_eq(&single, &msaa));
        assert!(std::sync::Arc::ptr_eq(&single, &r.cached_pipelines(1)));
        assert!(std::sync::Arc::ptr_eq(&msaa, &r.cached_pipelines(crate::constants::RASTER_MSAA_SAMPLES)));
        Ok(())
    }
}

pub(crate) fn cluster_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    [
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::BufferBindingType::Uniform,
            wgpu::BufferBindingType::Storage { read_only: false },
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| buffer_entry(binding as u32, wgpu::ShaderStages::COMPUTE, ty))
    .collect()
}

/// The same scene globals for ray kernels, independent of their image bindings.
pub(crate) fn ray_globals<'a>(s: &'a InstalledScene, uniform: &'a wgpu::Buffer, objects: &'a wgpu::Buffer, lights: &'a wgpu::Buffer, clusters: &'a wgpu::Buffer, cluster_lights: &'a wgpu::Buffer, light_tree: &'a wgpu::Buffer) -> Vec<wgpu::BindGroupEntry<'a>> {
    vec![entry(10,light_tree.as_entire_binding()),entry(0,uniform.as_entire_binding()),entry(1,objects.as_entire_binding()),entry(2,lights.as_entire_binding()),
        entry(3,wgpu::BindingResource::TextureView(&s.environment.specular)),entry(4,wgpu::BindingResource::Sampler(&s.environment.sampler)),
        entry(5,wgpu::BindingResource::TextureView(&s.environment.diffuse)),entry(6,wgpu::BindingResource::TextureView(&s.environment.brdf)),
        entry(7,clusters.as_entire_binding()),entry(8,cluster_lights.as_entire_binding()),entry(9,wgpu::BindingResource::TextureView(&s.environment.ltc))]
}

#[cfg(test)]
mod foreground_admission_tests {
    use super::*;
    #[test]
    fn deferred_compilation_waits_for_foreground_preparation() {
        let schedule = std::sync::Arc::new((Mutex::new(Schedule { foreground_preparations: 1, ..Default::default() }), std::sync::Condvar::new()));
        let foreground = ForegroundPreparation(schedule.clone());
        std::thread::scope(|scope| {
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (admitted_tx, admitted_rx) = std::sync::mpsc::channel();
            let schedule = &schedule;
            let worker = scope.spawn(move || {
                started_tx.send(()).unwrap();
                let _permit = Permit::new(schedule, true);
                admitted_tx.send(()).unwrap();
            });
            started_rx.recv().unwrap();
            // Foreground work remains admitted; only optional compilation waits.
            drop(Permit::new(schedule, false));
            assert!(admitted_rx.recv_timeout(std::time::Duration::from_millis(20)).is_err());
            drop(foreground);
            admitted_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
            worker.join().unwrap();
        });
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn panicked_variant_worker_fails_settle_instead_of_hanging() -> Result<()> {
        let r = crate::test_renderer()?;
        // A standalone compiler: the renderer's own drop also settles its workers.
        let compiler = Compiler::new(&r.device, None, Known::default(), Blob::ledger_only());
        compiler.spawn_variant(VariantCompile::Background, || panic!("injected variant failure"));
        compiler.spawn_variant(VariantCompile::Background, || Err(anyhow::anyhow!("injected compile failure")));
        let (settled_tx, settled) = std::sync::mpsc::channel();
        let settling = compiler.clone();
        std::thread::spawn(move || settled_tx.send(settling.settle()).unwrap());
        let result = settled.recv_timeout(std::time::Duration::from_secs(10)).expect("settle hung after a variant worker panic");
        assert!(result.is_err());
        Ok(())
    }
}

/// Populate shipped pipeline descriptors through the same constructors used at
/// install and frame-slot creation. No scene-keyed material variants are needed.
pub(crate) fn warm_builtin(r: &Renderer) -> Result<()> {
    materials::mip_resources(r);
    if crate::gpu_bc7::supported(r, 4, 4, 64) {
        crate::gpu_bc7::Context::new(r)?;
    }
    environment::Environment::pipeline(r);
    environment::reduce_pipeline(r);
    render_images::publication_pipeline(r);
    for samples in [1, crate::constants::RASTER_MSAA_SAMPLES] {
        passes::subsurface::pipelines(r, samples);
        // Geometry layouts do not depend on device ray-query support. Camera
        // shadow resources are warmed separately with their actual ray axis.
        for attributes in 0..=crate::constants::PRECOMPILED_ATTRIBUTE_SLOTS as u32 {
            for denoise in [false, true] {
                let p = if denoise || attributes>0 {
                    std::sync::Arc::new(CachedPipelines::with_sources(r, RESOLVE,
                        samples, false, crate::slang_forward::SlangScene { denoise, attributes, ..Default::default() }))
                } else { r.cached_pipelines(samples) };
                if attributes == 0 {
                    p.ensure_effects(r, EffectPlan::default())?;
                    let effects = EffectPlan { indirect:true, reflections:true,
                        compositor:true, ..Default::default() };
                    p.ensure_effects(r, effects)?;
                    passes::camera_cull::pipelines(r, &p);
                    passes::decoupled::merge_pipeline(&p, r);
                    p.deform_pipeline.get_or_init(|| p.build_deform_pipeline(r));
                    p.cluster_pipeline.get_or_init(|| p.build_cluster_pipeline(r));
                    if r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) && !denoise && samples == 1 {
                        crate::probes::pipelines(r, &p, 0, &VariantCompile::Inline)?;
                    }
                }
                let all = crate::forward_variant::ForwardKey::generic(crate::forward_variant::Stage::Controls,false).features;
                let mut roles = std::collections::HashSet::new();
                for transparent in [false,true] { for alpha_mask in [false,true] {
                    for double_sided in [false,true] { for dithered in [false,true] {
                        let flags = crate::plan::DrawFlags { transparent, alpha_mask,
                            casts_shadows:true, hair:false, moving:false };
                        for &kind in Role::kinds(&flags, true) {
                            roles.insert(Role::of_capabilities(kind, &flags, double_sided, dithered, all));
                        }
                    }}
                }}
                let mut keys = Vec::new();
                for layout in crate::geometry::VertexLayout::builtin_layouts(attributes as u8) {
                    for &role in &roles {
                        if denoise && !matches!(role, Role::Controls(_) | Role::TransparentControls(_) | Role::ProgramControls {..}) { continue; }
                        if samples > 1 && matches!(role,
                            Role::Visibility | Role::VisibilitySingle | Role::VisibilityMasked | Role::TransparentFragments) { continue; }
                        keys.push((layout,role));
                    }
                }
                warm_geometry(r,&p,keys)?;
            }
        }
    }
    r.errors.check()
}

/// Setup bounds driver concurrency independently of the finite descriptor count.
fn warm_geometry(r:&Renderer, p:&CachedPipelines, keys:Vec<GeometryKey>) -> Result<()> {
    // Exhaustive vertex-fetch/role descriptors only help when another process
    // can reuse them. Without a pipeline cache, Vulkan drivers such as lavapipe
    // repeat this entire matrix at startup; library programs are still compiled
    // by their frame-slot constructors. Metal retains pipelines in its system cache.
    if !r.release_preparation && r.cache.is_none() && r.adapter_info.backend != wgpu::Backend::Metal {
        return Ok(());
    }
    let next=std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        (0..std::thread::available_parallelism().map_or(1,usize::from).min(8)).map(|_| {
            let (keys,next)=(&keys,&next);
            scope.spawn(move || {
                while let Some(&key)=keys.get(next.fetch_add(1,std::sync::atomic::Ordering::Relaxed)) {
                    p.geometry(&r.compiler,key)?;
                }
                Ok(())
            })
        }).collect::<Vec<_>>().into_iter()
            .try_for_each(join)
    })
}

#[cfg(test)]
mod registry_identity_tests {
    use super::*;

    #[test]
    fn plain_hit_preserves_capabilities_and_rejects_complex_materials() {
        use crate::forward_variant::*;
        for features in [FEATURE_SPECIALIZED, FEATURE_SPECIALIZED | FeatureMask::from(SHADING_METAL)
            | FEATURE_LIGHT_TYPES | FEATURE_FINITE | FEATURE_RECTANGLE] {
            let ProgramKernel::Gi { features: plain } = (ProgramKernel::Gi { features }).plain_hit().unwrap() else { unreachable!() };
            assert_eq!(features & !plain, 0, "a plain hit key may only add capabilities");
            assert_eq!(plain & ((1 << (TEXTURE_FEATURES + 2)) - 4), 0, "no texture lookups");
            assert_eq!(plain & FeatureMask::from(SHADING_LAYERS), 0, "no layered material evaluation");
            assert_eq!((ProgramKernel::Reflections { features }).plain_hit(), Some(ProgramKernel::Reflections { features: plain }));
        }
        for features in [0, FEATURE_SPECIALIZED | FEATURE_COAT, FEATURE_SPECIALIZED | FEATURE_TRANSMISSION,
            FEATURE_SPECIALIZED | FEATURE_LOBES, FEATURE_SPECIALIZED | (1 << 2)] {
            assert!((ProgramKernel::Gi { features }).plain_hit().is_none());
        }
    }
    #[test]
    fn a_path_kernel_names_one_prebuilt_engine_per_family() {
        use crate::forward_variant::*;
        let path = |features, boundaries| ProgramKernel::Path { adaptive: true, catchers: false, random_walk: false,
            features, boundaries, emitter_profiles: false, media: None };
        let engine = path(FEATURE_SPECIALIZED | FEATURE_COAT | FEATURE_TRANSMISSION, 1).path_engine().unwrap();
        assert_eq!(path(FEATURE_SPECIALIZED, 1).path_engine(), Ok(engine), "the material mix does not select an engine");
        assert_eq!(engine.path_engine(), Ok(engine), "an engine key is its own engine");
        assert_ne!(path(FEATURE_SPECIALIZED, 0).path_engine(), Ok(engine), "boundary capacity is a family");
        let ProgramKernel::Path { features, emitter_profiles, .. } = engine else { unreachable!() };
        assert!(features & FEATURE_SPECIALIZED == 0 && emitter_profiles, "every capability, as Unspecialized states them");
        assert!(crate::surface_native::PathEngine::ALL.contains(&crate::surface_native::PathEngine { adaptive: true, boundaries: 1 }));
        // What a capability cannot express has no family: the scene compiles its own kernel.
        assert!(path(FEATURE_SPECIALIZED, crate::surface_native::PathEngine::BOUNDARIES + 1).path_engine().is_err());
        assert!(path(FEATURE_SPECIALIZED | FEATURE_IMPLICIT_GEOMETRY, 0).path_engine().is_err());
        for kernel in [
            ProgramKernel::Path { adaptive: true, catchers: true, random_walk: false, features: FEATURE_SPECIALIZED, boundaries: 0, emitter_profiles: false, media: None },
            ProgramKernel::Path { adaptive: true, catchers: false, random_walk: true, features: FEATURE_SPECIALIZED, boundaries: 0, emitter_profiles: false, media: None },
            ProgramKernel::Path { adaptive: true, catchers: false, random_walk: false, features: FEATURE_SPECIALIZED, boundaries: 0, emitter_profiles: false, media: Some(1) },
            ProgramKernel::Gi { features: FEATURE_SPECIALIZED },
        ] {
            assert!(kernel.path_engine().is_err(), "{kernel:?}");
        }
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn mesh_edit_reuses_pipelines_across_field_policy() -> Result<()> {
        let r = crate::test_renderer()?;
        let root=std::path::Path::new(".");
        let document=serde_json::json!({"version":2,"nodes":[
            {"type":"mesh","geometry":{
                "attributes":{"POSITION":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[0,1,0]]}},
                "indices":{"type":"u32","data":[0,1,2]}},
                "materials":[{"inputs":{"base_color":[0.8,0.2,0.1]}}]},
            {"type":"camera","translation":[0,0,4],"target":[0,0,0]},
            {"type":"light","shape":"environment","color":[0.2,0.2,0.2]}
        ]});
        let mut scene=crate::test_scene(document,root)?;
        scene.render.resolution=[64,36];
        scene.render.background=vork_scene::v2::Background::Color([0.1,0.1,0.1]);
        let intent=serde_json::from_value(serde_json::json!({"samples":1,"denoise":false}))?;
        scene.render.settings=vork_scene::Profile::load("std/draft")?.resolve(&intent)?;
        scene.render.settings.approximate_field_boundaries=false;
        let options=||InstallOptions {variants:VariantCompile::Inline,..Default::default()};
        let first=r.install_with_options(scene.clone(),&root,options())?;
        r.compiler.settle()?;
        let known=r.compiler.known.lock().unwrap().keys.clone();
        let geometry_slots=first.pipelines.geometry.lock().unwrap().len();
        let creation_start=crate::timing::creations(&r.device).len();
        // A numeric edit changes scene data, never its shader specialization.
        scene.render.background=vork_scene::v2::Background::Color([0.2,0.3,0.4]);
        scene.render.settings.approximate_field_boundaries=true;
        let edited=r.install_with_options(scene,&root,options())?;
        r.compiler.settle()?;
        assert!(std::sync::Arc::ptr_eq(&first.pipelines,&edited.pipelines),
            "mesh edits must reuse the registry across the field policy");
        assert_eq!(r.compiler.known.lock().unwrap().keys,known,"mesh edit admitted new variant keys");
        assert_eq!(edited.pipelines.geometry.lock().unwrap().len(),geometry_slots,
            "mesh edit allocated new geometry pipeline slots");
        let creations=crate::timing::creations(&r.device);
        let new=&creations[creation_start..];
        // Display/instance pipelines can create fresh objects using the driver
        // cache; they must not trigger fresh scene shader compilation.
        eprintln!("mesh edit creation receipt: {:?}",new.iter().map(|c|(&c.kind,&c.label)).collect::<Vec<_>>());
        assert!(new.iter().all(|c|c.kind!="Slang"),"mesh edit recompiled shader code");
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn first_opaque_gltf_insertion_reuses_scene_shaders() -> Result<()> {
        let r = crate::test_renderer()?;
        let root=std::env::temp_dir().join(format!("vork-gltf-insertion-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let positions=[[-1f32,-1.,0.],[1.,-1.,0.],[0.,1.,0.]];
        let normals=[[0f32,0.,1.];3];
        let bytes:Vec<u8>=positions.iter().chain(&normals).flatten().flat_map(|v|v.to_le_bytes()).collect();
        std::fs::write(root.join("triangle.bin"),bytes)?;
        std::fs::write(root.join("triangle.gltf"),serde_json::to_vec(&serde_json::json!({
            "asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0,"NORMAL":1}}]}],
            "buffers":[{"uri":"triangle.bin","byteLength":72}],
            "bufferViews":[{"buffer":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":36}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-1,-1,0],"max":[1,1,0]},
                {"bufferView":1,"componentType":5126,"count":3,"type":"VEC3"}]
        }))?)?;
        let geometry=serde_json::json!({"attributes":{
            "POSITION":{"type":"vec3","data":positions},"NORMAL":{"type":"vec3","data":normals}},
            "indices":{"type":"u32","data":[0,1,2]}});
        // Match the imported default's response and vertex layout: insertion adds
        // geometry, not a new material capability. Also retain an authored program
        // to catch invalidation of unrelated program shaders.
        for programmed in [false,true] {
            let mut document=serde_json::json!({"version":2,"nodes":[
                {"type":"mesh","geometry":geometry,"materials":[{"double_sided":false,
                    "inputs":{"base_color":[1,1,1],"base_metalness":1,"specular_roughness":1}}]},
                {"type":"camera","translation":[0,0,4],"target":[0,0,0]},
                {"type":"light","shape":"environment","color":[0.2,0.2,0.2]}]});
            if programmed {
                document["programs"]=serde_json::json!({"paint":{"source":"import vork; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs i) { Closure c={}; c.base_color=float3(.4,.2,.1); return c; }"}});
                document["materials"]=serde_json::json!({"paint":{"program":"paint"}});
                document["nodes"].as_array_mut().unwrap().push(serde_json::json!({
                    "type":"mesh","geometry":geometry,"materials":["paint"],"translation":[-2,0,0]}));
            }
            let mut scene=crate::test_scene(document,&root)?;
            scene.render.resolution=[64,36];
            let intent=serde_json::from_value(serde_json::json!({"samples":1,"denoise":false}))?;
            scene.render.settings=vork_scene::Profile::load("std/draft")?.resolve(&intent)?;
            let options=||InstallOptions {variants:VariantCompile::Inline,..Default::default()};
            let first=r.install_with_options(scene.clone(),&root,options())?;
            r.compiler.settle()?;
            let known=r.compiler.known.lock().unwrap().keys.clone();
            let creation_start=crate::timing::creations(&r.device).len();
            scene.nodes.push(serde_json::from_value(serde_json::json!({
                "type":"asset","file":"triangle.gltf","translation":[2,0,0]}))?);
            let edited=r.install_with_options(scene,&root,options())?;
            r.compiler.settle()?;
            assert!(std::sync::Arc::ptr_eq(&first.pipelines,&edited.pipelines),
                "first opaque glTF invalidated scene shaders (programmed={programmed})");
            assert_eq!(r.compiler.known.lock().unwrap().keys,known,
                "first opaque glTF admitted new shader keys (programmed={programmed})");
            let creations=crate::timing::creations(&r.device);
            assert!(creations[creation_start..].iter().all(|c|c.kind!="Slang"),
                "first opaque glTF recompiled shaders: {:?}",creations[creation_start..].iter().map(|c|(&c.kind,&c.label)).collect::<Vec<_>>());
        }
        std::fs::remove_dir_all(root)?;
        r.errors.check()?;
        Ok(())
    }

    fn identity_store_entries(root:&std::path::Path)->std::collections::BTreeSet<std::ffi::OsString> {
        std::fs::read_dir(root).unwrap().map(|entry|entry.unwrap().file_name()).filter(|name| {
            let name=name.to_string_lossy();name.contains(".slang-variant-") || name.contains(".slang-input-")
        }).collect()
    }
    fn identity_geometry(attributes:u32)->GeometryKey {
        (crate::geometry::VertexLayout {points:false,deform:false,tangent:false,uv:[crate::constants::VERTEX_DEFAULT_UV;2],color:false,custom:attributes as u8},Role::Controls(0))
    }
    #[test]
    fn release_inline_preparation_keeps_exact_movie_variants() -> Result<()> {
        use crate::forward_variant::FEATURE_SPECIALIZED;
        let r=pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let registry=std::sync::Arc::new(CachedPipelines::with_sources(
            &r,RESOLVE,1,false,crate::slang_forward::SlangScene {no_shadow_containment:true,..Default::default()}));
        for kernel in [ProgramKernel::Gi {features:FEATURE_SPECIALIZED},ProgramKernel::Reflections {features:FEATURE_SPECIALIZED}] {
            let install=|mode:&VariantCompile|match kernel {
                ProgramKernel::Gi {features}=>passes::ao::ray_pipeline(&r,&registry,features,false,mode),
                ProgramKernel::Reflections {features}=>passes::reflections::pipeline(&r,&registry,features,false,mode),
                _=>unreachable!(),
            };
            assert_eq!(registry.drawn_kernel(kernel,&VariantCompile::Setup,&r.compiler),kernel.superset().unwrap());
            install(&VariantCompile::Setup)?;
            assert!(registry.camera_slot(kernel).get().is_none(),"setup must not masquerade as exact coverage");
            assert_eq!(registry.drawn_kernel(kernel,&VariantCompile::Inline,&r.compiler),kernel);
            let exact=install(&VariantCompile::Inline)?;
            assert!(exact.ready() && !exact.deferred());
            assert!(registry.camera_slot(kernel).get().is_some());
            assert_eq!(registry.drawn_kernel(kernel,&VariantCompile::Background,&r.compiler),kernel);
            let created=r.compiler.created();
            let movie=install(&VariantCompile::Background)?;
            assert!(movie.ready() && !movie.deferred());
            assert_eq!(r.compiler.created(),created,"movie recreated its prepared exact pipeline");
        }
        r.finish_shader_warmup()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn background_compilation_reuses_requested_registry() -> Result<()> {
        use crate::forward_variant::FEATURE_SPECIALIZED;
        use crate::slang_forward::SlangScene;
        let mut r = crate::test_renderer()?;
        let root=std::env::temp_dir().join(format!("vork-background-identity-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&root)?;
        r.disk_cache=crate::disk_cache::Store::for_test(root.clone());
        r.compiler=Compiler::new(&r.device,r.cache.as_ref(),Known::load(root.join("known.entry"),false),r.compiler.blob());
        let scene=SlangScene {no_shadow_containment:true,..Default::default()};
        let kernels=[ProgramKernel::Gi {features:FEATURE_SPECIALIZED},ProgramKernel::Reflections {features:FEATURE_SPECIALIZED}];
        let geometry=std::collections::HashSet::new();
        let install=|registry:&std::sync::Arc<CachedPipelines>,kernel,mode|match kernel {
            ProgramKernel::Gi {features}=>passes::ao::ray_pipeline(&r,registry,features,false,mode),
            ProgramKernel::Reflections {features}=>passes::reflections::pipeline(&r,registry,features,false,mode),
            _=>unreachable!(),
        };
        // Setup has remembered only the plain hit families. Neither exact
        // specialization is warm when the first scene registry is requested.
        let setup=std::sync::Arc::new(CachedPipelines::with_sources(&r,RESOLVE,1,false,scene.clone()));
        for kernel in kernels {install(&setup,kernel.plain_hit().unwrap(),&VariantCompile::Inline)?;}
        let first=r.program_pipelines(RESOLVE,1,false,scene.clone(),&geometry,&kernels)?;
        let foreground=r.compiler.foreground_preparation();
        let mut drawn=Vec::new();
        for kernel in kernels {
            assert_eq!(first.drawn_kernel(kernel,&VariantCompile::Background,&r.compiler),kernel.plain_hit().unwrap());
            let pipeline=install(&first,kernel,&VariantCompile::Background)?;
            assert!(pipeline.deferred());
            assert!(!pipeline.ready());
            drawn.push(pipeline);
        }
        drop(foreground);
        r.compiler.settle()?;
        assert!(drawn.iter().all(Drawn::ready));
        for kernel in kernels {
            assert_eq!(first.drawn_kernel(kernel,&VariantCompile::Background,&r.compiler),kernel);
        }
        let created=r.compiler.created();
        let second=r.program_pipelines(RESOLVE,1,false,scene,&geometry,&kernels)?;
        assert!(std::sync::Arc::ptr_eq(&first,&second),"background warmth split the requested registry");
        assert_eq!(r.program_pipelines.lock().unwrap().len(),1);
        for kernel in kernels {
            assert!(std::sync::Arc::ptr_eq(&first.camera_slot(kernel),&second.camera_slot(kernel)));
            let pipeline=install(&second,kernel,&VariantCompile::Background)?;
            assert!(pipeline.ready());
            assert!(!pipeline.deferred());
        }
        r.compiler.settle()?;
        assert_eq!(r.compiler.created(),created,"second install recreated an exact pipeline");
        r.errors.check()?;
        drop(r);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    fn check_only_does_not_capture_native_pipeline_inputs() -> Result<()> {
        use crate::slang_forward::SlangScene;
        let mut r=pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        r.check_only=true;
        let root=std::env::temp_dir().join(format!("vork-check-identity-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&root)?;
        r.disk_cache=crate::disk_cache::Store::for_test(root.clone());
        let attributes=crate::constants::PRECOMPILED_ATTRIBUTE_SLOTS as u32+1;
        let key=identity_geometry(attributes);
        let keys=std::collections::HashSet::from([key]);
        let kernels=[ProgramKernel::Gi {features:0}];
        let registry=r.program_pipelines(RESOLVE,1,false,SlangScene {attributes,field_controls:true,..Default::default()},&keys,&kernels)?;
        registry.geometry(&r.compiler,key)?;
        passes::ao::ray_pipeline(&r,&registry,0,false,&VariantCompile::Setup)?;
        r.compiler.settle()?;
        assert!(r.program_pipelines.lock().unwrap().is_empty(),"checks do not retain executable registries");
        assert!(identity_store_entries(&root).is_empty(),"check captured native inputs or compiled an artifact");
        r.errors.check()?;
        drop(r);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn unconsumed_flags_reuse_registry_and_pipeline_slots() -> Result<()> {
        use crate::slang_forward::SlangScene;
        let mut r = crate::test_renderer()?;
        let root=std::env::temp_dir().join(format!("vork-identity-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&root)?;
        r.disk_cache=crate::disk_cache::Store::for_test(root.clone());
        r.compiler=Compiler::new(&r.device,r.cache.as_ref(),Known::load(root.join("known.entry"),false),r.compiler.blob());
        let mut expected=Vec::new();
        for attributes in [0,crate::constants::PRECOMPILED_ATTRIBUTE_SLOTS as u32+1] {
            let base=SlangScene {attributes,..Default::default()};
            let key=identity_geometry(attributes);
            let keys=std::collections::HashSet::from([key]);
            let registry=r.program_pipelines(RESOLVE,1,false,base.clone(),&keys,&[])?;
            registry.geometry(&r.compiler,key)?;
            let remembered=registry.content_key(key).expect("compiled entry has a captured input receipt");
            expected.push((attributes,remembered.to_hex().to_string()));
            assert!(r.compiler.is_known(&remembered));
            let start=crate::timing::creations(&r.device).len();
            let pipeline_start=r.compiler.created();
            for (boundary,probes) in [(true,false),(false,true),(true,true)] {
                let edited=r.program_pipelines(RESOLVE,1,false,SlangScene {no_boundary_transactions:boundary,probes,..base.clone()},&keys,&[])?;
                assert!(std::sync::Arc::ptr_eq(&registry,&edited),"irrelevant flags split the planned registry");
                edited.geometry(&r.compiler,key)?;
                assert_eq!(edited.content_key(key),Some(remembered));
            }
            let creations=crate::timing::creations(&r.device);
            assert_eq!(r.compiler.created(),pipeline_start,"irrelevant flags created a pipeline");
            assert!(creations[start..].iter().all(|c|c.kind=="shader module"),"irrelevant flags compiled code");
            eprintln!("identity proof: attributes={attributes}, three non-consumed edits: zero pipelines and zero recompiles; remembered={remembered}");
            let consumed=r.program_pipelines(RESOLVE,1,false,SlangScene {field_controls:true,..base},&keys,&[])?;
            assert!(!std::sync::Arc::ptr_eq(&registry,&consumed),"a consumed define/layout must split the registry");
            consumed.geometry(&r.compiler,key)?;
            assert_ne!(consumed.content_key(key),Some(remembered));
            assert!(r.compiler.created()>pipeline_start);
        }
        r.settle_pipelines()?;
        std::fs::write(root.join("expected.json"),serde_json::to_vec(&expected)?)?;
        r.errors.check()?;
        drop(r);
        let status=std::process::Command::new(std::env::current_exe()?)
            .args(["--ignored","--exact","pipelines::registry_identity_tests::persistent_keys_reuse_unconsumed_flags","--nocapture","--test-threads=1"])
            .env("VORK_IDENTITY_TEST_STORE",&root).status()?;
        assert!(status.success(),"fresh-process reuse failed");
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
    #[test]
    #[ignore="spawned by the registry test with its isolated persisted cache"]
    fn persistent_keys_reuse_unconsumed_flags() -> Result<()> {
        use crate::slang_forward::SlangScene;
        let root=std::path::PathBuf::from(std::env::var_os("VORK_IDENTITY_TEST_STORE").expect("parent supplies persisted test store"));
        let expected:Vec<(u32,String)>=serde_json::from_slice(&std::fs::read(root.join("expected.json"))?)?;
        let mut r = crate::test_renderer()?;
        r.disk_cache=crate::disk_cache::Store::for_test(root.clone());
        r.compiler=Compiler::new(&r.device,r.cache.as_ref(),Known::load(root.join("known.entry"),true),r.compiler.blob());
        let entries=identity_store_entries(&root);
        let start=crate::timing::creations(&r.device).len();
        for (attributes,expected) in expected {
            let key=identity_geometry(attributes);
            let keys=std::collections::HashSet::from([key]);
            let registry=r.program_pipelines(RESOLVE,1,false,SlangScene {attributes,no_boundary_transactions:true,probes:true,..Default::default()},&keys,&[])?;
            let remembered=registry.content_key(key).expect("persisted compiler receipt");
            assert_eq!(remembered.to_hex().as_str(),expected);
            assert!(r.compiler.is_known(&remembered),"fresh process missed remembered key");
            registry.geometry(&r.compiler,key)?;
        }
        assert!(crate::timing::creations(&r.device)[start..].iter().all(|c|c.kind!="Slang" && c.kind!="SPIR-V optimization"));
        assert_eq!(identity_store_entries(&root),entries,"non-consumed flags created cache entries");
        eprintln!("identity proof: fresh process: zero recompiles, identical remembered keys and native cache entries");
        r.errors.check()
    }

    #[test]
    fn remembered_keys_cover_shader_inputs_and_descriptor_capacities() {
        use crate::slang_forward::SlangScene;
        let base=SlangScene::default();
        let request=|scene:&SlangScene|scene.test_entry_inputs("scene_controls",("controls_fs","fragment"),None);
        let (args,key)=request(&base);
        assert!(args.iter().any(|arg|arg=="-DVORK_NO_IMPLICIT"));
        assert!(args.iter().any(|arg|arg=="-DVORK_NO_CONNECTION_TRANSMISSION"));
        assert!(args.iter().any(|arg|arg=="-DVORK_NO_GAUSSIANS"));
        for (changed,define,present) in [
            (SlangScene {aov_count:1,..base.clone()},"-DVORK_AOV_COUNT=1",true),
            (SlangScene {field_controls:true,..base.clone()},"-DVORK_NO_IMPLICIT",false),
            (SlangScene {software_raster:true,..base.clone()},"-DVORK_SOFTWARE_RASTER",true),
            (SlangScene {connection_transmission:true,..base.clone()},"-DVORK_NO_CONNECTION_TRANSMISSION",false),
            (SlangScene {no_points:true,..base.clone()},"-DVORK_NO_POINTS",true),
            (SlangScene {gaussians:true,..base.clone()},"-DVORK_NO_GAUSSIANS",false),
            (SlangScene {no_shadow_containment:true,..base.clone()},"-DVORK_NO_SHADOW_CONTAINMENT",true),
            (SlangScene {program_trace:true,..base.clone()},"-DVORK_PROGRAM_TRACE",true),
            (SlangScene {field_controls:true,approximate_field_boundaries:true,..base.clone()},"-DVORK_APPROXIMATE_FIELD_ADMISSION",true),
        ] {
            let (emitted,changed_key)=request(&changed);
            assert_eq!(emitted.iter().any(|arg|arg==define),present,"{define}");
            assert_ne!(key,changed_key,"emitted compiler input {define} must change identity");
        }
        assert_eq!(material_entries_for(0,0),material_entries_for(1,1));
        assert_eq!(material_entries_for(1,1),material_entries_for(vork_slang::registry::MATERIAL_TEXTURE_CAPACITY,vork_slang::registry::MATERIAL_SAMPLER_CAPACITY));
        for entries in [material_entries_for(vork_slang::registry::MATERIAL_TEXTURE_CAPACITY+1,1),material_entries_for(1,vork_slang::registry::MATERIAL_SAMPLER_CAPACITY+1)] {
            assert_ne!(layout_identity(&[Some(material_entries_for(1,1))]),layout_identity(&[Some(entries)]));
        }
    }

}

#[cfg(test)]
mod stage_limit_tests {
    use super::*;
    #[test]
    fn metal_reserves_sizes_once_per_stage_and_for_vertex_pulling() {
        let storage=buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true});
        let uniform=buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Uniform);
        assert_eq!(stage_resources(&[&[uniform]],wgpu::ShaderStages::COMPUTE).buffers,1);
        assert_eq!(stage_resources(&[&[storage],&[storage,uniform]],wgpu::ShaderStages::COMPUTE).buffers,4);
        assert_eq!(stage_resources(&[&[storage]],wgpu::ShaderStages::FRAGMENT).buffers,0);
        assert_eq!(stage_resources(&[],wgpu::ShaderStages::VERTEX).buffers,1);
    }

    #[test]
    fn scalar_admission_includes_uniforms_and_native_sizes() {
        let limits=wgpu::Limits {max_storage_buffers_per_shader_stage:31,..Default::default()};
        for uniforms in [0,1,2] {
            let allowed=scalar_storage_limit(&limits,true,uniforms);
            let mut entries=(0..uniforms).map(|i|buffer_entry(i,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Uniform)).collect::<Vec<_>>();
            entries.extend((0..allowed).map(|i|buffer_entry(i+uniforms,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true})));
            check_stage_limits("dynamic scalar layout",&[&entries],wgpu::ShaderStages::COMPUTE).unwrap();
            entries.push(buffer_entry(allowed+uniforms,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}));
            assert!(check_stage_limits("dynamic scalar overflow",&[&entries],wgpu::ShaderStages::COMPUTE).is_err());
            assert_eq!(scalar_storage_limit(&limits,false,uniforms),31);
        }
    }

    #[test]
    fn scene_sized_camera_layouts_fit_metal() {
        // Array capacities vary with allowed scene content. Each table takes
        // one Metal argument-buffer slot, including arrays of textures/samplers.
        for (textures,samplers,pages) in [(1,1,1),(4096,128,1024)] {
            for rays in [false,true] {for trace in [false,true] {for samples in [1,4,32] {
                let owned=[raster_global_entries(),material_entries_for(textures,samplers),
                    camera_shadow_entries_for(rays,trace),passes::decoupled::camera_geometry_entries_for(pages),
                    passes::decoupled::camera_entries(samples)];
                let groups=owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
                for stage in [wgpu::ShaderStages::VERTEX,wgpu::ShaderStages::FRAGMENT,wgpu::ShaderStages::COMPUTE] {
                    check_stage_limits("decoupled shade",&groups,stage).unwrap();
                }
            }}}
        }
    }

    #[test]
    fn stage_limits_use_executor_slots_for_mixed_resource_arrays() {
        let stage=wgpu::ShaderStages::COMPUTE;
        let scalar=buffer_entry(0,stage,wgpu::BufferBindingType::Storage {read_only:true});
        let texture=wgpu::BindGroupLayoutEntry {visibility:stage,..texture_entry(1,wgpu::TextureSampleType::Float {filterable:true},false)};
        let sampler=wgpu::BindGroupLayoutEntry {visibility:stage,..sampler_entry(2,false)};
        let array=|entry:wgpu::BindGroupLayoutEntry|wgpu::BindGroupLayoutEntry {count:std::num::NonZeroU32::new(64),..entry};
        let scalars=(0..30).map(|binding|wgpu::BindGroupLayoutEntry {binding,..scalar}).collect::<Vec<_>>();
        let texture_array=[array(texture)];
        assert!(check_stage_limits("scalar buffers plus texture array",&[&scalars,&texture_array],stage)
            .unwrap_err().to_string().contains("32 buffers"));

        // Arrays of every resource class occupy argument-buffer slots, even
        // when their capacities exceed the scalar texture/sampler limits.
        let storage=wgpu::BindGroupLayoutEntry {binding:4,visibility:stage,ty:wgpu::BindingType::StorageTexture {access:wgpu::StorageTextureAccess::WriteOnly,format:wgpu::TextureFormat::Rgba8Unorm,view_dimension:wgpu::TextureViewDimension::D2},count:None};
        let mixed=[array(scalar),array(texture),array(sampler),array(storage),texture,sampler,storage,
            wgpu::BindGroupLayoutEntry {binding:3,visibility:stage,ty:wgpu::BindingType::AccelerationStructure {vertex_return:false},count:None},
            wgpu::BindGroupLayoutEntry {visibility:wgpu::ShaderStages::FRAGMENT,..array(texture)}];
        assert_eq!(stage_resources(&[&scalars[..25],&mixed],stage),
            StageResources {buffers:31,textures:2,samplers:1});
        check_stage_limits("mixed arrays at limit",&[&scalars[..25],&mixed],stage).unwrap();
        assert!(check_stage_limits("mixed arrays overflow",&[&scalars[..26],&mixed],stage).is_err());

        // Composite resource kinds must also retain every executor slot.
        let external=[wgpu::BindGroupLayoutEntry {binding:0,visibility:stage,ty:wgpu::BindingType::ExternalTexture,count:None}];
        assert_eq!(stage_resources(&[&external],stage),StageResources {buffers:1,textures:3,samplers:0});
    }

    #[test]
    fn stage_limits_count_slots_filter_visibility_and_reject_overflow() {
        let buffer = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding, visibility,
            ty: wgpu::BindingType::Buffer {ty: wgpu::BufferBindingType::Storage {read_only:true}, has_dynamic_offset:false, min_binding_size:None},
            count: std::num::NonZeroU32::new(8),
        };
        let mut entries = (0..30).map(|i| buffer(i, wgpu::ShaderStages::COMPUTE)).collect::<Vec<_>>();
        entries.push(buffer(30, wgpu::ShaderStages::FRAGMENT));
        assert_eq!(stage_resources(&[&entries], wgpu::ShaderStages::COMPUTE).buffers, 31);
        check_stage_limits("at limit", &[&entries], wgpu::ShaderStages::COMPUTE).unwrap();
        entries[30].visibility = wgpu::ShaderStages::COMPUTE;
        assert!(check_stage_limits("overflow", &[&entries], wgpu::ShaderStages::COMPUTE).unwrap_err().to_string().contains("32 buffers"));
        for (ty, limit) in [(wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), 16),
            (wgpu::BindingType::Texture {sample_type:wgpu::TextureSampleType::Float {filterable:true}, view_dimension:wgpu::TextureViewDimension::D2, multisampled:false},128)] {
            let entries = (0..=limit).map(|binding| wgpu::BindGroupLayoutEntry {binding, visibility:wgpu::ShaderStages::COMPUTE, ty, count:None}).collect::<Vec<_>>();
            check_stage_limits("at limit", &[&entries[..limit as usize]], wgpu::ShaderStages::COMPUTE).unwrap();
            assert!(check_stage_limits("overflow", &[&entries], wgpu::ShaderStages::COMPUTE).is_err());
        }
    }
}
