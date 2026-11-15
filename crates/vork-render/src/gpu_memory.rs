//! Resource context and normal-error delivery for synchronous and asynchronous GPU failures.
use anyhow::Result;
use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

/// Shared diagnostic for resource and dispatch admission against a known device limit.
pub(crate) use vork_encode::gpu_memory::device_limit;
/// A refusal naming a feature or capacity this device lacks: a device failure wherever it surfaces.
pub(crate) use vork_encode::DeviceLimit;

/// Check the complete binding, with recovery advice specific to its allocation.
/// Allocation capacity alone does not imply that a buffer can be bound whole.
pub(crate) fn storage_binding(limits: &wgpu::Limits, label: &str, bytes: u64, remedy: &str) -> Result<()> {
    anyhow::ensure!(
        bytes <= limits.max_storage_buffer_binding_size && bytes <= limits.max_buffer_size,
        device_limit(label, format_args!("{bytes} bytes in one storage binding"),
            format_args!("max_storage_buffer_binding_size={} bytes, max_buffer_size={} bytes", limits.max_storage_buffer_binding_size, limits.max_buffer_size), remedy)
    );
    Ok(())
}

/// Check whole-frame buffers whose sizes grow with pixels and samples. When one
/// does not fit, report the internal camera-budget ceiling and a same-aspect
/// resolution at which every buffer fits this device. `bindings(extent, samples)` gives
/// (label, bytes) pairs that grow with each argument; returns their sizes.
pub(crate) fn frame_bindings<const N: usize>(
    limits: &wgpu::Limits,
    [width, height]: [u32; 2],
    samples: u32,
    bindings: impl Fn([u32; 2], u32) -> [(&'static str, u64); N],
) -> Result<[u64; N]> {
    let limit = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
    let fits = |extent, samples| bindings(extent, samples).iter().all(|(_, bytes)| *bytes <= limit);
    let sizes = bindings([width, height], samples);
    let Some(&(label, needed)) = sizes.iter().find(|(_, bytes)| *bytes > limit) else {
        return Ok(sizes.map(|(_, bytes)| bytes));
    };
    // Largest value in 0..high for which `fits` holds (high itself does not fit).
    let largest = |high: u32, fits: &dyn Fn(u32) -> bool| {
        let (mut lo, mut hi) = (0, high.saturating_sub(1));
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fits(mid) { lo = mid } else { hi = mid - 1 }
        }
        lo
    };
    let fewer = largest(samples, &|s| s > 0 && fits([width, height], s));
    let scaled = |w: u32| [w, (u64::from(w) * u64::from(height) / u64::from(width)).max(1) as u32];
    let smaller = largest(width, &|w| w > 0 && fits(scaled(w), samples));
    let mut remedies = Vec::new();
    if fewer >= vork_scene::Settings::for_level(0, Default::default()).samples {
        remedies.push("choose --profile std/draft for a lower camera budget".to_owned());
    }
    if smaller > 0 {
        let [w, h] = scaled(smaller);
        remedies.push(format!("reduce the rendered resolution to {w}x{h} or smaller"));
    }
    let advice = if remedies.is_empty() {
        "use a GPU with larger storage binding and buffer limits; no supported profile budget or resolution fits this allocation".to_owned()
    } else {
        remedies.join(", or ")
    };
    anyhow::bail!(device_limit(label,
        format_args!("{needed} bytes at {width}x{height} with {samples} samples"),
        format_args!("max_storage_buffer_binding_size={} bytes, max_buffer_size={} bytes (camera budget ceiling at this extent={fewer} samples)", limits.max_storage_buffer_binding_size, limits.max_buffer_size),
        &advice))
}

/// How often a wait gives the driver a chance to report a hung GPU.
const LOSS_PROBE: std::time::Duration = std::time::Duration::from_secs(1);

/// Wait for `submission` (all work submitted so far when `None`), however
/// long it takes, then report any device error. NVIDIA reports a timed-out
/// dispatch (Xid 109, CTX SWITCH TIMEOUT) as device loss only to a later
/// submission: an unsliced wait on the hung work never returned (20 minutes on
/// an L4). Each slice therefore submits an empty batch, and the loss arrives as
/// an error within a few slices. The target is fixed before the first slice.
pub(crate) fn wait(device: &wgpu::Device, queue: &wgpu::Queue, errors: &Errors, submission: Option<wgpu::SubmissionIndex>) -> Result<()> {
    let target = submission.unwrap_or_else(|| crate::command_finish::Segments::flush(queue));
    loop {
        match device.poll(wgpu::PollType::Wait { submission_index: Some(target.clone()), timeout: Some(LOSS_PROBE) }) {
            Ok(_) => return errors.check(),
            Err(wgpu::PollError::Timeout) => {
                crate::command_finish::Segments::flush(queue);
                #[cfg(test)]
                WAIT_HEARTBEATS.with(|count| count.set(count.get() + 1));
                errors.check()?;
            }
            Err(error) => {
                errors.check()?;
                return Err(anyhow::Error::from(error).context(crate::failure::FailureKind::Device));
            }
        }
    }
}

// Timeout heartbeats submitted by `wait` on this thread, so submission-count
// tests can tell them from the submissions their subject owes.
#[cfg(test)]
thread_local! {
    static WAIT_HEARTBEATS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn wait_heartbeat_count() -> usize { WAIT_HEARTBEATS.with(|count| count.get()) }

thread_local! {
    static CREATING: RefCell<Option<Request>> = const { RefCell::new(None) };
}

struct Request {
    description: String,
    failed: bool,
}

/// The resource being created on this thread, marked failed by the error.
fn creating_failed() -> Option<String> {
    CREATING.with(|current| {
        current.borrow_mut().as_mut().map(|request| {
            request.failed = true;
            request.description.clone()
        })
    })
}

type WeakErrors = std::sync::Weak<Mutex<Option<Failure>>>;
static DEVICE_ERRORS: Mutex<Vec<(u64, WeakErrors)>> = Mutex::new(Vec::new());
static LEDGERS: Mutex<Vec<(u64, std::sync::Weak<crate::residency::Ledger>)>> = Mutex::new(Vec::new());
pub(crate) fn ledger(device: &wgpu::Device) -> Arc<crate::residency::Ledger> {
    let mut ledgers = LEDGERS.lock().unwrap();
    ledgers.retain(|(_, ledger)|ledger.strong_count()>0);
    let key = device_key(device);
    if let Some(value) = ledgers.iter().find(|(id, _)|*id==key).and_then(|(_, value)|value.upgrade()) {return value;}
    let value = Arc::new(crate::residency::Ledger::default());
    ledgers.push((key, Arc::downgrade(&value)));
    value
}
fn try_allocation<T>(device: &wgpu::Device, _bytes: u64, mut create: impl FnMut() -> T) -> std::result::Result<T, (T, wgpu::Error)> {
    let mut first = true;
    ledger(device).retry(|| {
        if !first { let _ = device.poll(wgpu::PollType::Poll); }
        first = false;
        let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let value = create();
        match pollster::block_on(scope.pop()) {
            None => Ok(value),
            Some(error) => Err((value, error)),
        }
    }, |_| !is_lost(device), || Ok(()))
}

fn allocation<T>(device: &wgpu::Device, bytes: u64, create: impl FnMut() -> T) -> T {
    match try_allocation(device, bytes, create) {
        Ok(value) => value,
        Err((value, error)) => {
            creating_failed();
            let errors = DEVICE_ERRORS.lock().unwrap().iter().find(|(key, _)|*key==device_key(device))
                .and_then(|(_, failure)|failure.upgrade().map(Errors));
            if let Some(errors) = errors {errors.record(error);}
            value
        }
    }
}
/// Optional retention must never turn a successful simulation into an OOM.
pub(crate) fn optional_buffer(device: &wgpu::Device, desc: &wgpu::BufferDescriptor<'_>) -> Option<wgpu::Buffer> {
    if is_lost(device) {return None;}
    match try_allocation(device, desc.size, ||device.create_buffer(desc)) {
        Ok(value) if !is_lost(device) => Some(value),
        _ => None,
    }
}

/// Paged transfers are atomic with respect to admission: failed optional pages
/// release prior allocations without recording copies or mutating working state.
pub(crate) fn optional_pages<T>(count: usize, allocate: impl FnMut(usize) -> Option<T>, mut copy: impl FnMut(usize, &T)) -> Option<Vec<T>> {
    let pages: Option<Vec<_>> = (0..count).map(allocate).collect();
    let pages = pages?;
    for (index, page) in pages.iter().enumerate() { copy(index,page); }
    Some(pages)
}

/// Speculative snapshots do not displace other caches. Unknown telemetry still
/// attempts the native allocation, with OOM contained by the error scope.
pub(crate) fn retention_buffer(device: &wgpu::Device, desc: &wgpu::BufferDescriptor<'_>) -> Option<wgpu::Buffer> {
    if is_lost(device) { return None; }
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let value = device.create_buffer(desc);
    (pollster::block_on(scope.pop()).is_none() && !is_lost(device)).then_some(value)
}

pub(crate) fn optional_buffer_init(device: &wgpu::Device, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> Option<wgpu::Buffer> {
    let size = (contents.len() as u64).div_ceil(wgpu::COPY_BUFFER_ALIGNMENT)*wgpu::COPY_BUFFER_ALIGNMENT;
    let buffer = optional_buffer(device, &wgpu::BufferDescriptor {label:Some(label),size,usage,mapped_at_creation:true})?;
    buffer.get_mapped_range_mut(..).slice(..contents.len()).copy_from_slice(contents);
    buffer.unmap();
    Some(buffer)
}

/// Devices that have been lost, by wgpu identity (its id carries an epoch, so
/// it is never reused). A lost device returns every later resource invalid.
static LOST: Mutex<Vec<u64>> = Mutex::new(Vec::new());
fn device_key(device: &wgpu::Device) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    device.hash(&mut hasher);
    hasher.finish()
}
pub(crate) fn is_lost(device: &wgpu::Device) -> bool {
    LOST.lock().unwrap().contains(&device_key(device))
}

#[derive(Clone, Default)]
pub(crate) struct Errors(Arc<Mutex<Option<Failure>>>);
struct Failure {
    oom: bool,
    message: String,
}
impl Errors {
    pub(crate) fn install(device: &wgpu::Device) -> Self {
        let errors = Self::default();
        {
            let mut registry = DEVICE_ERRORS.lock().unwrap();
            registry.retain(|(_, failure)|failure.strong_count()>0);
            registry.push((device_key(device), Arc::downgrade(&errors.0)));
        }
        let callback = errors.clone();
        device.on_uncaptured_error(Arc::new(move |error| callback.record(error)));
        // wgpu surfaces device loss only here; later calls on the lost device
        // report nothing (wgpu_core.rs handle_error ignores ErrorType::DeviceLost).
        let lost = errors.clone();
        let key = device_key(device);
        device.set_device_lost_callback(move |reason, message| {
            LOST.lock().unwrap().push(key);
            lost.lose(&format!("{reason:?} {message}"));
        });
        errors
    }
    /// The device is gone, so this failure outranks errors it caused; an
    /// out-of-memory failure that preceded it remains the root cause. wgpu
    /// reports a loss synchronously from the call that noticed it, so a
    /// resource being created on this thread is named.
    fn lose(&self, message: &str) {
        let creating = creating_failed();
        let mut first = self.0.lock().unwrap();
        if let Some(failure)=first.as_mut().filter(|failure| failure.oom) {
            failure.message.push_str(&format!("; subsequent device loss: {message}"));
            return;
        }
        let message = match creating {
            Some(resource) => format!("GPU device lost while creating {resource}: {message}"),
            None => format!("GPU device lost: {message}"),
        };
        *first = Some(Failure { oom: false, message });
    }
    fn record(&self, error: wgpu::Error) {
        let kind = match &error {
            wgpu::Error::OutOfMemory { .. } => "out of memory",
            wgpu::Error::Validation { .. } => "validation error",
            wgpu::Error::Internal { .. } => "internal error",
        };
        let context = creating_failed().map(|a| format!(" creating {a}")).unwrap_or_default();
        // Invalid resources can produce secondary validation errors. Retain the
        // creation failure that caused them, including its requested size.
        let mut first = self.0.lock().unwrap();
        let oom = matches!(error, wgpu::Error::OutOfMemory { .. });
        if first.as_ref().is_none_or(|failure| oom && !failure.oom) {
            let mut message=format!("GPU {kind}{context}: {error}");
            if let Some(prior)=first.as_ref() {
                message.push_str(&format!("; earlier GPU failure: {}",prior.message));
            }
            *first = Some(Failure { oom, message });
        }
    }
    /// A completed map alone does not establish that frame execution succeeded.
    pub(crate) fn frame_complete(&self, validity: Result<()>) -> Result<()> {
        if let Err(error) = validity {
            // The callback can lag successful fence/map results after a reset.
            // Preserve an existing native failure (especially OOM), and retain
            // this failure so no successor can publish from the suspect device.
            self.0.lock().unwrap().get_or_insert_with(|| Failure {
                oom: false,
                message: format!("GPU device lost: frame completion could not be verified: {error:#}"),
            });
        }
        self.check().map_err(|error| error.context("frame discarded"))
    }
    pub(crate) fn check(&self) -> Result<()> {
        if let Some(error) = self.0.lock().unwrap().as_ref() {
            let message=error.message.clone();
            return Err(anyhow::anyhow!(message).context(crate::failure::FailureKind::Device));
        }
        Ok(())
    }
}

/// Run `create` with `description` naming the resource in any GPU error it causes.
fn creating<T>(description: String, create: impl FnOnce() -> T) -> T {
    struct Restore(Option<Request>);
    impl Drop for Restore {
        fn drop(&mut self) {
            CREATING.with(|current| current.replace(self.0.take()));
        }
    }
    let _restore = Restore(CREATING.with(|current| {
        current.replace(Some(Request {
            description,
            failed: false,
        }))
    }));
    create()
}

pub(crate) fn render_pipeline(device: &wgpu::Device, desc: &wgpu::RenderPipelineDescriptor<'_>) -> wgpu::RenderPipeline {
    creating(format!("render pipeline {:?}", desc.label.unwrap_or("unnamed")), || allocation(device, 0, ||device.create_render_pipeline(desc)))
}

pub(crate) fn compute_pipeline(device: &wgpu::Device, desc: &wgpu::ComputePipelineDescriptor<'_>) -> wgpu::ComputePipeline {
    creating(format!("compute pipeline {:?}", desc.label.unwrap_or("unnamed")), || allocation(device, 0, ||device.create_compute_pipeline(desc)))
}

#[cfg(test)]
thread_local! {static BUFFER_ALLOCATIONS:RefCell<Option<Vec<String>>>=const {RefCell::new(None)};}
#[cfg(test)]
pub(crate) fn test_buffer_allocations<T>(run:impl FnOnce()->T) -> (T,Vec<String>) {
    BUFFER_ALLOCATIONS.with(|trace| {assert!(trace.borrow().is_none());*trace.borrow_mut()=Some(Vec::new());});
    struct Reset;
    impl Drop for Reset {fn drop(&mut self) {BUFFER_ALLOCATIONS.with(|trace| {trace.borrow_mut().take();});}}
    let _reset=Reset;
    let result=run();
    let allocations=BUFFER_ALLOCATIONS.with(|trace|trace.borrow_mut().take().unwrap());
    (result,allocations)
}
#[cfg(test)]
fn trace_buffer(label:&str) {BUFFER_ALLOCATIONS.with(|trace| {if let Some(labels)=trace.borrow_mut().as_mut() {labels.push(label.to_owned());}});}

pub(crate) fn buffer(device: &wgpu::Device, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::Buffer {
    #[cfg(test)] trace_buffer(desc.label.unwrap_or("unnamed"));
    creating(
        format!(
            "buffer {:?} ({} bytes)",
            desc.label.unwrap_or("unnamed"),
            desc.size
        ),
        || allocation(device, desc.size, || device.create_buffer(desc)),
    )
}

pub(crate) fn buffer_init(
    device: &wgpu::Device,
    label: &str,
    contents: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    #[cfg(test)] trace_buffer(label);
    let bytes =
        (contents.len() as u64).div_ceil(wgpu::COPY_BUFFER_ALIGNMENT) * wgpu::COPY_BUFFER_ALIGNMENT;
    creating(format!("buffer {label:?} ({bytes} bytes)"), || {
        let buffer = allocation(device, bytes, || device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage,
            mapped_at_creation: !contents.is_empty(),
        }));
        // Do not map a failed allocation or a lost device's invalid buffer:
        // mapping would panic before the renderer can return the GPU error.
        let failed = CREATING.with(|current| current.borrow().as_ref().is_some_and(|r| r.failed))
            || is_lost(device);
        if !failed && !contents.is_empty() {
            buffer
                .get_mapped_range_mut(..)
                .slice(..contents.len())
                .copy_from_slice(contents);
            buffer.unmap();
        }
        buffer
    })
}

/// Requested texel bytes, including blocks, mip levels, layers and samples;
/// driver padding and internal allocations are outside the resource descriptor.
pub(crate) fn texture_bytes(desc: &wgpu::TextureDescriptor<'_>) -> u64 {
    let (bw, bh) = desc.format.block_dimensions();
    let block = desc
        .format
        .block_copy_size(Some(wgpu::TextureAspect::DepthOnly))
        .or(desc.format.block_copy_size(None))
        .unwrap_or(std::mem::size_of::<u32>() as u32) as u64;
    (0..desc.mip_level_count)
        .map(|level| {
            let size = desc.mip_level_size(level).unwrap();
            u64::from(size.width.div_ceil(bw))
                * u64::from(size.height.div_ceil(bh))
                * u64::from(size.depth_or_array_layers)
        })
        .sum::<u64>()
        * block
        * u64::from(desc.sample_count)
}

pub(crate) fn texture(device: &wgpu::Device, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::Texture {
    creating(
        format!(
            "texture {:?} ({} bytes; {}x{}x{}, {:?}, {} samples)",
            desc.label.unwrap_or("unnamed"),
            texture_bytes(desc),
            desc.size.width,
            desc.size.height,
            desc.size.depth_or_array_layers,
            desc.format,
            desc.sample_count
        ),
        || allocation(device, texture_bytes(desc), || device.create_texture(desc)),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn limit_messages_include_demand_limit_remedy_and_help() -> Result<()> {
        for (binding, allocation) in [(127, 256), (256, 127)] {
            let limits = wgpu::Limits {
                max_storage_buffer_binding_size: binding,
                max_buffer_size: allocation,
                ..Default::default()
            };
            let error = storage_binding(&limits, "test pool", 128, "reduce pool size").unwrap_err().to_string();
            for expected in ["test pool", "128 bytes", &format!("max_storage_buffer_binding_size={binding} bytes"),
                &format!("max_buffer_size={allocation} bytes"), "reduce pool size", "vork help limits"] {
                assert!(error.contains(expected), "{error}");
            }
            storage_binding(&limits, "test pool", 127, "reduce pool size")?;
        }
        let limits = wgpu::Limits {max_storage_buffer_binding_size: 128, max_buffer_size: 256, ..Default::default()};
        let error = frame_bindings(&limits, [8, 8], 4, |[w,h], samples| {
            [("sample records", u64::from(w)*u64::from(h)*u64::from(samples))]
        }).unwrap_err().to_string();
        for expected in ["256 bytes", "8x8", "4 samples", "128 bytes", "reduce the rendered resolution", "vork help limits"] {
            assert!(error.contains(expected), "{error}");
        }
        Ok(())
    }

    /// Grep every renderer source, including subdirectories, so new literal
    /// refusals cannot bypass the common diagnostic contract.
    #[test]
    fn device_limit_refusals_use_the_message_helper() -> Result<()> {
        let needle = ["exceeds", " device limit"].concat();
        let variants = [needle.clone(), ["exceed", " device limits"].concat(), ["exceeds", " GPU texture side"].concat()];
        let canonical = format!("format!(\"{{label}} {needle}: requested {{requested}}; limit {{limit}}; {{remedy}}; see vork help limits\")");
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut dirs = vec![root.join("src"), root.join("../vork-encode/src")];
        let mut helper_count = 0;
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(dir)? {
                let path = entry?.path();
                if path.is_dir() { dirs.push(path); continue; }
                if path.extension().is_none_or(|ext| ext != "rs") { continue; }
                let source = std::fs::read_to_string(&path)?;
                for (line, text) in source.lines().enumerate() {
                    if !variants.iter().any(|phrase| text.contains(phrase)) { continue; }
                    assert!(path.file_name().is_some_and(|name| name == "gpu_memory.rs") && text.trim() == canonical,
                        "{}:{}: device-limit refusal must use gpu_memory::device_limit", path.display(), line+1);
                    helper_count += 1;
                }
            }
        }
        assert_eq!(helper_count, 1, "the helper must own the refusal text");
        Ok(())
    }

    use super::*;
    #[test]
    fn partial_optional_admission_records_no_transfers() {
        for fail_at in [Some(0),Some(1),Some(2),None] {
            let mut allocated=0; let mut copied=Vec::new();
            let pages=optional_pages(3,|i| {
                allocated+=1;
                (Some(i)!=fail_at).then_some(i)
            },|i,value|copied.push((i,*value)));
            if let Some(failed)=fail_at {
                assert!(pages.is_none()); assert!(copied.is_empty()); assert_eq!(allocated,failed+1);
            } else { assert_eq!(pages,Some(vec![0,1,2])); assert_eq!(copied,vec![(0,0),(1,1),(2,2)]); }
        }
    }
    #[test]
    fn storage_admission_uses_device_limits_without_a_fixed_ceiling() -> Result<()> {
        let mut limits = wgpu::Limits::default();
        // Finer eval fog transport and the three original fur material tables.
        for bytes in [317_030_400, 1_276_706_584, 920_858_124, 557_962_124] {
            limits.max_buffer_size = 2 << 30;
            limits.max_storage_buffer_binding_size = 128 << 20;
            let error = storage_binding(&limits, "authored resource", bytes, "reduce the resource pool size").unwrap_err();
            assert!(error.to_string().contains(&format!("{bytes} bytes")));
            limits.max_storage_buffer_binding_size = bytes;
            storage_binding(&limits, "authored resource", bytes, "reduce the resource pool size")?;
            limits.max_buffer_size = bytes - 1;
            assert!(storage_binding(&limits, "authored resource", bytes, "reduce the resource pool size").is_err());
        }
        Ok(())
    }
    #[test]
    fn allocation_failure_is_an_error_with_resource_size_and_keeps_the_root_cause() {
        let errors = Errors::default();
        creating("buffer \"frame objects\" (1048576 bytes)".into(), || {
            errors.record(wgpu::Error::OutOfMemory {
                source: Box::new(std::io::Error::other("device memory exhausted")),
            })
        });
        errors.record(wgpu::Error::Validation {
            source: Box::new(std::io::Error::other("invalid resource")),
            description: "secondary validation".into(),
        });
        let error = errors.check().unwrap_err();
        assert_eq!(crate::failure::exit_code(&error), 4);
        let message = format!("{error:#}");
        assert!(message.contains("GPU out of memory"), "{message}");
        assert!(
            message.contains("frame objects") && message.contains("1048576 bytes"),
            "{message}"
        );
        assert!(!message.contains("secondary"), "{message}");
        errors.lose("Metal native fault 0x42");
        let combined=format!("{:#}",errors.check().unwrap_err());
        assert!(combined.contains("GPU out of memory") && combined.contains("Metal native fault 0x42"), "{combined}");
    }

    #[test]
    fn invalid_frame_timestamps_discard_frame_without_loss_callback() {
        for (begin, end) in [(0, 0), (0, 20), (20, 0), (20, 19)] {
            let errors = Errors::default();
            let timing = crate::timing::elapsed_ms(begin, end, 1.)
                .map(|_| ())
                .ok_or_else(|| anyhow::anyhow!("invalid GPU timestamp pair for query 24, sample 0: begin={begin}, end={end}"));
            let error = errors.frame_complete(timing).expect_err("invalid frame was published");
            assert_eq!(crate::failure::exit_code(&error), 4);
            let message = format!("{error:#}");
            assert!(message.contains("GPU device lost") && message.contains("frame discarded") && message.contains("query 24"), "{message}");
            assert!(errors.check().is_err(), "a suspect device must not publish successors");
        }
    }

    #[test]
    fn frame_completion_preserves_loss_and_accepts_valid_timestamps() {
        let errors = Errors::default();
        errors.frame_complete(Ok(())).unwrap();
        errors.lose("simulated delayed driver reset");
        let error = errors.frame_complete(Err(anyhow::anyhow!("secondary invalid timestamp"))).unwrap_err();
        assert_eq!(crate::failure::exit_code(&error), 4);
        let message = format!("{error:#}");
        assert!(message.contains("simulated delayed driver reset") && message.contains("frame discarded"), "{message}");
    }

    #[test]
    fn device_loss_names_the_resource_being_created() {
        let errors = Errors::default();
        creating("compute pipeline \"unidirectional path\"".into(), || errors.lose("lost"));
        let error = errors.check().unwrap_err();
        assert_eq!(crate::failure::exit_code(&error), 4);
        let message = format!("{error:#}");
        assert!(message.contains("GPU device lost while creating compute pipeline \"unidirectional path\""), "{message}");
    }

    #[test]
    fn device_loss_retains_reason_after_secondary_errors() {
        let errors = Errors::default();
        errors.lose("Unknown native command fault 0x42");
        errors.record(wgpu::Error::Validation {
            source: Box::new(std::io::Error::other("invalid resource")),
            description: "secondary validation".into(),
        });
        let message = format!("{:#}", errors.check().unwrap_err());
        assert!(message.contains("native command fault 0x42"), "{message}");
        assert!(!message.contains("secondary validation"), "{message}");
        errors.record(wgpu::Error::OutOfMemory {
            source: Box::new(std::io::Error::other("later allocation failure")),
        });
        let combined=format!("{:#}",errors.check().unwrap_err());
        assert!(combined.contains("GPU out of memory") && combined.contains("native command fault 0x42"), "{combined}");
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn lost_device_initialized_buffers_report_instead_of_panicking() -> Result<()> {
        let renderer = crate::test_renderer()?;
        renderer.device.destroy();
        // wgpu reports a destroyed device's loss from the next poll.
        let _ = renderer.device.poll(wgpu::PollType::Poll);
        let _buffer = buffer_init(&renderer.device, "after loss", &[1, 2, 3, 4], wgpu::BufferUsages::VERTEX);
        let message = format!("{:#}", renderer.errors.check().unwrap_err());
        assert!(message.contains("GPU device lost"), "{message}");
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn device_validation_uses_the_normal_error_boundary() -> Result<()> {
        let renderer = crate::test_renderer()?;
        let size = renderer.device.limits().max_buffer_size + 1;
        let _buffer = buffer(
            &renderer.device,
            &wgpu::BufferDescriptor {
                label: Some("oversized test buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        let error = renderer.errors.check().unwrap_err();
        assert_eq!(crate::failure::exit_code(&error), 4);
        let message = format!("{error:#}");
        assert!(
            message.contains("oversized test buffer") && message.contains(&format!("{size} bytes")),
            "{message}"
        );
        Ok(())
    }
}
