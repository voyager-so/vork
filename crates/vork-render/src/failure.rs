//! Stable CLI failure categories carried as typed anyhow context.
// The categories live in a file without renderer dependencies; see failure/kind.rs.
mod kind;
pub use kind::FailureKind;

/// The scene category on an error whose words stay the author's diagnostic: it displays as
/// the error it wraps and continues that error's chain, so the message at every level is the
/// one written where the scene was refused.
#[derive(Debug)]
struct Authored(anyhow::Error);
impl std::fmt::Display for Authored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}
impl std::error::Error for Authored {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> { self.0.source() }
}
/// Wrap reading, validating and lowering what the author wrote. An error already
/// categorised, or with a cause that has its own category (what the device hands back,
/// a missing compiler), is returned as it is; a filesystem error stays uncategorised, so
/// it keeps its own exit code; everything else becomes the scene's.
pub fn scene(error: anyhow::Error) -> anyhow::Error {
    if kind(&error).is_some() || filesystem(&error) { return error; }
    anyhow::Error::new(Authored(error))
}
/// A typed cause, also when the scene category wraps it.
pub(crate) fn cause<T: std::fmt::Display + std::fmt::Debug + Send + Sync + 'static>(error: &anyhow::Error) -> Option<&T> {
    error.downcast_ref::<T>().or_else(|| error.downcast_ref::<Authored>().and_then(|authored| authored.0.downcast_ref::<T>()))
}
/// Wrap operations that also evaluate scene inputs, retaining their precise category.
/// Leaf GPU failures can attach Device directly.
pub(crate) fn device(error: anyhow::Error) -> anyhow::Error {
    categorize(error, FailureKind::Device)
}
/// Wrap creating the GPU device, an encoder or the denoiser, retaining a precise category.
pub(crate) fn setup(error: anyhow::Error) -> anyhow::Error {
    categorize(error, FailureKind::Setup)
}
fn categorize(error: anyhow::Error, default: FailureKind) -> anyhow::Error {
    if error.downcast_ref::<FailureKind>().is_some() || error.is::<Authored>() { return error; }
    let category = kind(&error).unwrap_or(default);
    error.context(category)
}
/// A worker thread that unwound. Nothing an author writes is meant to panic the
/// renderer, so this is never the scene's; the payload keeps the cause.
pub(crate) fn panicked(what: &str, payload: Box<dyn std::any::Any + Send>) -> anyhow::Error {
    let cause = payload.downcast_ref::<String>().map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied()).unwrap_or("no message");
    anyhow::anyhow!("{what} panicked: {cause}").context(FailureKind::Device)
}
/// An attached category, else a cause whose category does not depend on where
/// it surfaces: a missing or wrong Slang compiler is a setup failure (a lazily
/// compiled variant meets it mid-render), an encoder's device limit is a
/// device failure of a working device, though met while opening the encoder,
/// and a buffer the device would not map, a wait it failed, a completion that
/// never arrived or a native allocation it refused is a device failure, though
/// met while installing a scene.
/// The causes are for crates below vork-render and for results the device hands
/// back; vork-render code attaches a FailureKind where it knows one.
pub(crate) fn kind(error: &anyhow::Error) -> Option<FailureKind> {
    error.downcast_ref::<FailureKind>().copied()
        .or_else(|| error.is::<Authored>().then_some(FailureKind::Scene))
        .or_else(|| error.downcast_ref::<vork_slang::CompilerUnavailable>().map(|_| FailureKind::Setup))
        .or_else(|| error.downcast_ref::<vork_encode::DeviceLimit>().map(|_| FailureKind::Device))
        .or_else(|| device_cause(error).then_some(FailureKind::Device))
}
fn device_cause(error: &anyhow::Error) -> bool {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if error.is::<ash::vk::Result>() { return true; }
    error.is::<wgpu::BufferAsyncError>() || error.is::<wgpu::PollError>()
        || error.is::<std::sync::mpsc::RecvError>() || error.is::<crate::denoise::OutOfMemory>()
        || error.is::<crate::probes::placement::MemoryPressure>()
}
fn filesystem(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<std::io::Error>())
}
/// Only setup failures point at `vork setup`; a failed render on a working
/// device (a scene beyond a renderer limit, say) is not one.
pub fn suggests_setup(error: &anyhow::Error) -> bool {
    matches!(kind(error), Some(FailureKind::Setup))
}

/// Categories survive contextual wrapping; filesystem errors retain their source.
/// Exit 3 says the scene or program is invalid and the machine is not in question,
/// so only an error categorised Scene gets it. An error nothing categorised is not
/// known to be the scene's: it exits 4 with the device, encoder and setup failures.
pub fn exit_code(error: &anyhow::Error) -> u8 {
    match kind(error) {
        Some(FailureKind::Usage) => 2,
        Some(FailureKind::Scene) => 3,
        Some(FailureKind::Device | FailureKind::Setup) => 4,
        None if filesystem(error) => 5,
        None => 4,
    }
}

/// Run upstream parser code on user files. A panic there
/// is a bug in that crate for malformed input; return it as an error naming the
/// crate source line instead of unwinding through the renderer. Contained panics
/// are not printed; every other panic reaches the previous hook unchanged.
pub(crate) fn contain<T>(what: &str, run: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    use std::cell::{Cell, RefCell};
    thread_local! {
        static DEPTH: Cell<u32> = const { Cell::new(0) };
        static PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
    }
    static HOOK: std::sync::Once = std::sync::Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if DEPTH.get() == 0 {
                return previous(info);
            }
            let at = info.location().map_or_else(String::new, |at| {
                // Name the crate file, not the registry checkout path.
                let file = at.file().replace('\\', "/");
                let file = file.split_once("/registry/src/")
                    .and_then(|(_, rest)| rest.split_once('/'))
                    .map_or(file.as_str(), |(_, file)| file);
                format!(" at {file}:{}", at.line())
            });
            let message = info.payload_as_str().unwrap_or("panic");
            PANIC.set(Some(format!("{at}: {message}")));
        }));
    });
    DEPTH.set(DEPTH.get() + 1);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
    DEPTH.set(DEPTH.get() - 1);
    result.unwrap_or_else(|_| {
        let panic = PANIC.take().unwrap_or_default();
        Err(anyhow::anyhow!("{what} failed on malformed input (upstream panic{panic})"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn contained_panics_become_errors_with_their_location() {
        let error = contain("parser", || -> anyhow::Result<()> { panic!("bad index") }).unwrap_err();
        let text = format!("{error:#}");
        assert!(text.starts_with("parser failed on malformed input (upstream panic at "), "{text}");
        assert!(text.contains("failure.rs:") && text.ends_with(": bad index)"), "{text}");
        assert_eq!(contain("parser", || Ok(7)).unwrap(), 7);
    }
    #[test]
    fn categories_survive_context_without_matching_error_text() {
        assert_eq!(exit_code(&anyhow::anyhow!("bad field").context(FailureKind::Scene).context("load")), 3);
        assert_eq!(exit_code(&device(anyhow::anyhow!("adapter"))), 4);
        assert_eq!(exit_code(&device(anyhow::anyhow!("focus").context(FailureKind::Scene))), 3);
        assert_eq!(exit_code(&anyhow::anyhow!("destination").context(FailureKind::Usage)), 2);
        assert_eq!(exit_code(&anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied)).context("write")), 5);
    }
    #[test]
    fn only_a_scene_category_exits_three() {
        // Nothing categorised this: it is not known to be the scene's.
        assert_eq!(exit_code(&anyhow::anyhow!("unexpected").context("render")), 4);
        assert!(!suggests_setup(&anyhow::anyhow!("unexpected")));
        // Reading and evaluating the document: its refusals are the scene's, its files' errors stay filesystem errors.
        let refused = scene(anyhow::anyhow!("nodes.a: unknown field").context("load scene")).context("render");
        assert_eq!(exit_code(&refused), 3);
        // The category adds no words: the message is the diagnostic, at every level.
        assert_eq!(format!("{refused:#}"), "render: load scene: nodes.a: unknown field");
        assert_eq!(scene(anyhow::anyhow!("nodes.a: unknown field")).to_string(), "nodes.a: unknown field");
        // A boundary outside it keeps it.
        assert_eq!(exit_code(&device(scene(anyhow::anyhow!("bad")))), 3);
        assert_eq!(exit_code(&setup(scene(anyhow::anyhow!("bad")).context("frame 0"))), 3);
        let missing = || anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound)).context("read texture");
        assert_eq!(exit_code(&scene(missing())), 5);
        assert_eq!(exit_code(&scene(missing()).context("install scene")), 5);
        // A category attached below the boundary is kept, whichever it is.
        assert_eq!(exit_code(&scene(anyhow::anyhow!("--at").context(FailureKind::Usage))), 2);
        assert_eq!(exit_code(&scene(device(anyhow::anyhow!("lost")))), 4);
        assert_eq!(exit_code(&scene(missing().context(FailureKind::Device))), 4);
        assert!(suggests_setup(&scene(setup(anyhow::anyhow!("no adapter")))));
    }
    #[test]
    fn what_the_device_hands_back_is_a_device_failure_inside_a_scene_boundary() {
        let causes: Vec<(&str, anyhow::Error)> = vec![
            ("map", wgpu::BufferAsyncError.into()),
            ("completion", { let (sender, receiver) = std::sync::mpsc::channel::<()>(); drop(sender); receiver.recv().unwrap_err().into() }),
            ("device limit", crate::gpu_memory::device_limit("pool", "2 bytes", "1 byte", "reduce it").into()),
        ];
        for (what, cause) in causes {
            let error = scene(cause.context("read environment").context("install scene"));
            assert_eq!(exit_code(&error), 4, "{what}: {error:#}");
            assert!(matches!(kind(&error), Some(FailureKind::Device)), "{what}");
            assert!(!suggests_setup(&error), "{what}");
        }
    }
    #[test]
    fn a_worker_panic_is_never_the_scenes_and_keeps_its_cause() {
        let join = |run: fn()| std::thread::spawn(run).join().unwrap_err();
        let error = scene(panicked("scene install worker", join(|| panic!("index {} out of range", 7))));
        assert_eq!(exit_code(&error), 4);
        assert_eq!(format!("{error:#}"), "device or encoder failure: scene install worker panicked: index 7 out of range");
        let error = panicked("device creation worker", join(|| panic!("backend fault")));
        assert!(format!("{error:#}").ends_with("device creation worker panicked: backend fault"), "{error:#}");
        let error = panicked("encoder preparation worker", join(|| std::panic::panic_any(3_u8)));
        assert_eq!(exit_code(&error), 4);
        assert!(format!("{error:#}").ends_with("panicked: no message"), "{error:#}");
    }
    /// The real construction paths that need no device: each refusal as the renderer builds it.
    #[test]
    fn device_class_refusals_exit_four_as_built() {
        // Capacity: a binding beyond the device's limits, met while installing.
        let limits = wgpu::Limits { max_storage_buffer_binding_size: 127, max_buffer_size: 256, ..Default::default() };
        let error = crate::gpu_memory::storage_binding(&limits, "test pool", 128, "reduce pool size").unwrap_err();
        assert_eq!(exit_code(&scene(error.context("install scene"))), 4);
        let error = crate::gpu_memory::frame_bindings(&wgpu::Limits { max_storage_buffer_binding_size: 128, max_buffer_size: 256, ..Default::default() },
            [8, 8], 4, |[w, h], samples| [("sample records", u64::from(w) * u64::from(h) * u64::from(samples))]).unwrap_err();
        assert_eq!(exit_code(&scene(error)), 4);
        let entries: Vec<_> = (0..64).map(|binding| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }).collect();
        let error = crate::pipelines::check_stage_limits("overflow", &[&entries], wgpu::ShaderStages::COMPUTE).unwrap_err();
        assert_eq!(exit_code(&scene(error)), 4);
        // What the device recorded.
        let errors = crate::gpu_memory::Errors::default();
        errors.frame_complete(Err(anyhow::anyhow!("timestamps unwritten"))).unwrap_err();
        assert_eq!(exit_code(&scene(errors.check().unwrap_err())), 4);
    }
    /// The real construction paths of scene refusals: each still exits 3, and a missing file 5.
    #[test]
    fn scene_refusals_exit_three_as_built() {
        let path = std::path::Path::new("scene.json");
        for document in [&b"{"[..], b"not a scene"] {
            let error = crate::decode_scene_document(document, path).err().expect("a malformed document decoded");
            assert_eq!(exit_code(&error), 3, "{error:#}");
        }
        let error = crate::load(std::path::Path::new("no/such/scene.json")).unwrap_err();
        assert_eq!(exit_code(&error), 5, "{error:#}");
    }
    /// A buffer the device will not map, as a readback meets it: destroyed before its map completes.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn a_failed_readback_map_exits_four() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let buffer = r.device.create_buffer(&wgpu::BufferDescriptor { label: Some("readback"), size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
        buffer.destroy();
        let read = || -> anyhow::Result<()> { r.wait(None)?; rx.recv()??; Ok(()) };
        let error = read().expect_err("a destroyed buffer mapped");
        assert!(error.is::<wgpu::BufferAsyncError>(), "{error:#}");
        assert_eq!(exit_code(&error), 4);
        // Met inside an installation, it is still the device's.
        assert_eq!(exit_code(&scene(error.context("install scene"))), 4);
        Ok(())
    }
    #[test]
    fn only_setup_failures_suggest_setup() {
        // A render beyond a renderer limit on a working device (here the path
        // occupancy backstop, an untagged frame error) is a device failure.
        let limit = device(anyhow::anyhow!("more glass solids overlap at a point").context("frame 0"));
        assert_eq!(exit_code(&limit), 4);
        assert!(!suggests_setup(&limit));
        let adapter = setup(anyhow::anyhow!("no adapter").context("create device"));
        assert_eq!(exit_code(&adapter), 4);
        assert!(suggests_setup(&adapter) && suggests_setup(&adapter.context("render")));
        // The first category attached is kept.
        assert!(!suggests_setup(&setup(anyhow::anyhow!("lost").context(FailureKind::Device))));
        assert!(!suggests_setup(&setup(anyhow::anyhow!("bad").context(FailureKind::Scene))));
    }
    #[test]
    fn slang_is_setup_and_encoder_limits_are_device_wherever_they_surface() {
        let slang = || anyhow::Error::new(vork_slang::CompilerUnavailable("run Slang compiler".into())).context("compile path variant");
        for error in [slang(), device(slang())] {
            assert_eq!(exit_code(&error), 4);
            assert!(suggests_setup(&error), "{error:#}");
        }
        assert!(matches!(device(slang()).downcast_ref::<FailureKind>(), Some(FailureKind::Setup)));
        let limit = || anyhow::Error::from(vork_encode::DeviceLimit("test device refusal".into())).context("open encoder");
        for error in [limit(), setup(limit())] {
            assert_eq!(exit_code(&error), 4);
            assert!(!suggests_setup(&error), "{error:#}");
        }
    }
}
