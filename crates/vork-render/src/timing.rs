//! Named diagnostic spans plus a separate internal scheduling interval.
use crate::constants;

const _: () = assert!(constants::GPU_SPANS.len() <= u64::BITS as usize);

pub(crate) fn count() -> u32 {
    sample_query() + 2
}

/// Internal counters never contribute to additive diagnostic span totals.
pub(crate) fn sample_query() -> u32 {
    constants::GPU_SPANS.len() as u32 * 2
}

pub(crate) fn boundary(r: &crate::Renderer, encoder: &mut wgpu::CommandEncoder, queries: &wgpu::QuerySet, index: u32) {
    if r.adapter_info.backend == wgpu::Backend::Metal {
        r.timestamp_marker.get_or_init(|| Marker::new(r)).encode(encoder, queries, index);
    } else if r.device.features().contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
        encoder.write_timestamp(queries, index);
    } else {
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sample scheduling boundary"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }),
        }));
    }
}
pub(crate) fn query(name: &str) -> u32 {
    constants::GPU_SPANS
        .iter()
        .position(|&span| span == name)
        .expect("unregistered GPU timing span") as u32
        * 2
}

/// Each written pair has an aligned resolve destination; one exposure copy
/// transfers the retained results without per-span staging copies.
pub(crate) fn resolve_stride() -> u64 {
    u64::from(count() / 2 + 1) * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT
}
pub(crate) fn resolve_offset(sample: usize, query: u32) -> u64 {
    debug_assert_eq!(query % 2, 0);
    sample as u64 * resolve_stride() + u64::from(query / 2) * wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT
}
/// A group can begin and end in different query sets. Each endpoint needs an
/// aligned destination; singleton groups resolve the ordinary adjacent pair.
pub(crate) fn group_end_offset(sample: usize) -> u64 {
    resolve_offset(sample, sample_query()) + wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT
}

pub(crate) fn mask(name: &str) -> u64 {
    1 << (query(name) / 2)
}

pub(crate) fn spans(mask: u64) -> impl Iterator<Item = &'static str> {
    constants::GPU_SPANS
        .iter()
        .copied()
        .enumerate()
        .filter_map(move |(index, name)| (mask & (1 << index) != 0).then_some(name))
}

/// Zero is an unwritten counter, not a zero-cost pass. Equal nonzero samples
/// are valid for work shorter than the timestamp counter's resolution.
pub(crate) fn elapsed_ms(begin: u64, end: u64, period: f32) -> Option<f64> {
    if begin == 0 {
        return None;
    }
    Some(end.checked_sub(begin)? as f64 * period as f64 / 1e6)
}

#[cfg(test)]
mod tests {
    #[test]
    fn missing_or_reversed_timestamps_are_not_free_work() {
        assert_eq!(super::elapsed_ms(0, 0, 1.), None);
        assert_eq!(super::elapsed_ms(0, 20, 1.), None);
        assert_eq!(super::elapsed_ms(20, 0, 1.), None);
        assert_eq!(super::elapsed_ms(20, 19, 1.), None);
        assert_eq!(super::elapsed_ms(20, 20, 1.), Some(0.));
        assert_eq!(super::elapsed_ms(20, 1_000_020, 2.), Some(2.));
    }
}

/// Metal does not reliably sample empty compute passes or standalone blits.
/// A single real store gives a boundary marker for work spanning several views.
/// Lazily shared per renderer; its scratch value has no rendering meaning.
#[derive(Clone)]
pub(crate) struct Marker {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
}
impl Marker {
    pub(crate) fn new(r: &crate::Renderer) -> Self {
        let device = &r.device;
        let shader=shader_module(device,wgpu::ShaderModuleDescriptor {
            label: Some("timestamp boundary marker"),
            source: wgpu::ShaderSource::Wgsl("@group(0) @binding(0) var<storage,read_write> marker:u32; @compute @workgroup_size(1) fn main() {marker=0u;}".into()),
        });
        // Through the shared compiler, like every other pipeline: counted, timed and announced there.
        let pipeline = r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("timestamp boundary marker"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("timestamp marker scratch"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scratch.as_entire_binding(),
            }],
        });
        Self { pipeline, bind }
    }
    pub(crate) fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
        index: u32,
    ) {
        self.encode_timestamps(encoder, wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: None,
        });
    }
    pub(crate) fn encode_timestamps(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: wgpu::ComputePassTimestampWrites<'_>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("timestamp boundary marker"),
            timestamp_writes: Some(timestamps),
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

/// Process-local creation receipts. Concurrent durations overlap; retain every
/// call rather than collapsing repeated labels or mixing Slang with driver JIT.
#[derive(Clone, serde::Serialize)]
pub(crate) struct Creation {
    #[serde(skip)]
    device: u64,
    pub(crate) label: String,
    pub(crate) kind: &'static str,
    slang_ms: f64,
    driver_ms: f64,
    /// Receipts are taken as the work returns; its interval ends there.
    #[serde(skip)]
    end: std::time::Instant,
}
static CREATIONS: std::sync::Mutex<Vec<Creation>> = std::sync::Mutex::new(Vec::new());
// Device ids are unique among live devices only because every Renderer opens
// its device from one process-wide wgpu instance (Renderer::create).
fn device_id(device: &wgpu::Device) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hash=std::collections::hash_map::DefaultHasher::new();
    device.hash(&mut hash);
    hash.finish()
}
pub(crate) fn creation(device: &wgpu::Device, label: &str, kind: &'static str, slang_ms: f64, driver_ms: f64) {
    CREATIONS.lock().unwrap().push(Creation {device:device_id(device),label:label.into(),kind,slang_ms,driver_ms,end:std::time::Instant::now()});
}
pub(crate) fn creations(device: &wgpu::Device) -> Vec<Creation> {
    let id=device_id(device);
    CREATIONS.lock().unwrap().iter().filter(|c|c.device==id).cloned().collect()
}
pub(crate) fn shader_module(device: &wgpu::Device, descriptor: wgpu::ShaderModuleDescriptor<'_>) -> wgpu::ShaderModule {
    let start=std::time::Instant::now();
    let label=descriptor.label.unwrap_or("shader module").to_owned();
    let _event=vork_slang::progress::announce("shader module",&label);
    let module=device.create_shader_module(descriptor);
    creation(device,&label,"shader module",0.,start.elapsed().as_secs_f64()*1000.);
    module
}

pub(crate) fn forget(device: &wgpu::Device) {
    let id=device_id(device);
    CREATIONS.lock().unwrap().retain(|c|c.device!=id);
    cache_saves().retain(|s|s.device!=id);
}

/// One step of saving a device's pipeline cache (`pipelines::Compiler::save`). Saves run on the
/// cache persistence worker, beside or between renders. A step is logged when it begins, so a
/// render's stage breakdown counts the part that overlapped it even while the step is still
/// running; once finished, it is reported once, by the next render report on its device
/// (`pipeline_cache_saves`).
#[derive(Clone, serde::Serialize)]
pub(crate) struct CacheSave {
    #[serde(skip)]
    device: u64,
    #[serde(skip)]
    id: u64,
    /// `serialize`: the driver hands over its cache. `publish`: an entry is written to the disk store.
    step: &'static str,
    /// `pipeline` (the driver's cache), `pipeline-keys` (the known-key ledger) or `shader-warm`.
    entry: &'static str,
    ms: f64,
    bytes: usize,
    /// False when the step produced nothing: the driver gave no data, or the store refused or failed the write.
    done: bool,
    #[serde(skip)]
    start: std::time::Instant,
    /// None while the step runs.
    #[serde(skip)]
    end: Option<std::time::Instant>,
}
static CACHE_SAVES: std::sync::Mutex<Vec<CacheSave>> = std::sync::Mutex::new(Vec::new());
fn cache_saves() -> std::sync::MutexGuard<'static, Vec<CacheSave>> {
    CACHE_SAVES.lock().unwrap_or_else(|poison| poison.into_inner())
}
/// A running save step; `finish` it with what it produced. Dropped unfinished (its work
/// panicked), it ends there having produced nothing.
pub(crate) struct CacheSaveStep(u64);
pub(crate) fn begin_cache_save(device: &wgpu::Device, step: &'static str, entry: &'static str) -> CacheSaveStep {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    cache_saves().push(CacheSave { device: device_id(device), id, step, entry, ms: 0., bytes: 0, done: false, start: std::time::Instant::now(), end: None });
    CacheSaveStep(id)
}
impl CacheSaveStep {
    pub(crate) fn finish(self, bytes: usize, done: bool) {
        self.end(bytes, done);
    }
    fn end(&self, bytes: usize, done: bool) {
        let end = std::time::Instant::now();
        // A step of a forgotten device is no longer logged.
        if let Some(save) = cache_saves().iter_mut().find(|save| save.id == self.0 && save.end.is_none()) {
            (save.ms, save.bytes, save.done, save.end) = ((end - save.start).as_secs_f64() * 1000., bytes, done, Some(end));
        }
    }
}
impl Drop for CacheSaveStep {
    fn drop(&mut self) { self.end(0, false); }
}
/// The device's finished save steps that no report has carried yet.
fn take_finished(log: &mut Vec<CacheSave>, device: u64) -> Vec<CacheSave> {
    let (taken, kept): (Vec<_>, Vec<_>) = std::mem::take(log).into_iter().partition(|save| save.device == device && save.end.is_some());
    *log = kept;
    taken
}

/// Host intervals spent encoding and writing delivered frames: image-file
/// compression and writes, and handing frames to or finishing video encoders.
/// Bounded like the Slang pass timeline; renders read their own wall interval.
struct EncodeLog { intervals: Vec<(std::time::Instant, std::time::Instant)>, dropped_before: Option<std::time::Instant> }
static ENCODE: std::sync::Mutex<EncodeLog> = std::sync::Mutex::new(EncodeLog { intervals: Vec::new(), dropped_before: None });
const ENCODE_CAPACITY: usize = 1 << 16;

pub(crate) fn encoded(start: std::time::Instant) {
    let end = std::time::Instant::now();
    let mut log = ENCODE.lock().unwrap_or_else(|poison| poison.into_inner());
    if log.intervals.len() >= ENCODE_CAPACITY {
        let last = log.intervals.drain(..ENCODE_CAPACITY / 2).map(|(_, end)| end).max();
        log.dropped_before = log.dropped_before.max(last);
    }
    log.intervals.push((start, end));
}

/// Wall time covered by the union of intervals clipped to `[start, end]`, the
/// summed (busy) duration of the clipped intervals, and their count.
fn coverage(mut intervals: Vec<(std::time::Instant, std::time::Instant)>, start: std::time::Instant, end: std::time::Instant) -> serde_json::Value {
    intervals.retain(|&(s, e)| s < end && e > start);
    for interval in &mut intervals { *interval = (interval.0.max(start), interval.1.min(end)); }
    intervals.sort();
    // `+ 0.` turns the empty sum's -0.0 into 0.
    let busy: f64 = intervals.iter().map(|(s, e)| (*e - *s).as_secs_f64()).sum::<f64>() + 0.;
    let (mut wall, mut cursor) = (0., start);
    for &(s, e) in &intervals {
        let s = s.max(cursor);
        if e > s { wall += (e - s).as_secs_f64(); cursor = e; }
    }
    serde_json::json!({"wall_ms": wall * 1000., "busy_ms": busy * 1000., "count": intervals.len()})
}

/// Per-render host and GPU totals the stage breakdown cannot read from the timelines.
pub(crate) struct StageTotals {
    pub(crate) setup_ms: f64,
    pub(crate) first_frame_ms: f64,
    pub(crate) frames_ms: f64,
    pub(crate) gpu_spans_ms: Vec<(&'static str, f64)>,
    pub(crate) denoise_busy_ms: f64,
    pub(crate) denoise_wait_ms: f64,
}

/// Where one render's wall time went (`stages` in every render report). Each
/// compile and encode stage is the union of its intervals inside this render's
/// wall interval (`wall_ms`) beside their summed duration (`busy_ms`): workers
/// overlap one another and the frame loop, so stages do not add up to the wall.
/// Slang and encode intervals are process-wide; another render running
/// concurrently in the same process would be counted in both. The two
/// pipeline-cache stages are the background save's steps where they overlapped
/// this render, a step still running counted up to `end`. Also returns the
/// device's finished save steps for the report (`pipeline_cache_saves`), taken
/// from the log under the same lock as the stages read it: a step is in exactly
/// one report's list, and each render's stages hold the part inside its interval.
pub(crate) fn stages(device: &wgpu::Device, start: std::time::Instant, end: std::time::Instant, totals: StageTotals) -> (serde_json::Value, Vec<CacheSave>) {
    let (passes, slang_truncated) = vork_slang::progress::passes(start, end);
    let pass = |kind| passes.iter().filter(|p| p.0 == kind).map(|&(_, s, e)| (s, e)).collect();
    let id = device_id(device);
    let driver = CREATIONS.lock().unwrap().iter()
        .filter(|c| c.device == id && c.driver_ms > 0. && matches!(c.kind, "pipeline" | "shader module"))
        .map(|c| (c.end - std::time::Duration::from_secs_f64(c.driver_ms / 1000.), c.end)).collect();
    let (serialize, publish, saves) = {
        let mut log = cache_saves();
        let step = |step: &str| log.iter().filter(|s| s.device == id && s.step == step)
            .map(|s| (s.start, s.end.unwrap_or(end))).collect::<Vec<_>>();
        let (serialize, publish) = (step("serialize"), step("publish"));
        (serialize, publish, take_finished(&mut log, id))
    };
    let (encode, encode_truncated) = {
        let log = ENCODE.lock().unwrap_or_else(|poison| poison.into_inner());
        (log.intervals.clone(), log.dropped_before.is_some_and(|dropped| dropped > start))
    };
    let gpu_pass_ms: serde_json::Map<_, _> = totals.gpu_spans_ms.iter()
        .filter(|(_, ms)| *ms > 0.).map(|(name, ms)| ((*name).to_owned(), (*ms).into())).collect();
    (serde_json::json!({
        "version": 1,
        "wall_ms": (end - start).as_secs_f64() * 1000.,
        "setup_ms": totals.setup_ms,
        "first_frame_ms": totals.first_frame_ms,
        "frames_ms": totals.frames_ms,
        "slang_front_end": coverage(pass(vork_slang::progress::Pass::FrontEnd), start, end),
        "slang_code_generation": coverage(pass(vork_slang::progress::Pass::CodeGeneration), start, end),
        "driver_pipeline": coverage(driver, start, end),
        "pipeline_cache_serialize": coverage(serialize, start, end),
        "pipeline_cache_publish": coverage(publish, start, end),
        "gpu_ms": totals.gpu_spans_ms.iter().map(|(_, ms)| ms).sum::<f64>() + 0.,
        "gpu_pass_ms": gpu_pass_ms,
        "denoise": {"busy_ms": totals.denoise_busy_ms, "wait_ms": totals.denoise_wait_ms},
        "encode": coverage(encode, start, end),
        "truncated": slang_truncated || encode_truncated,
    }), saves)
}

#[cfg(test)]
mod stage_tests {
    use std::time::{Duration, Instant};
    #[test]
    fn coverage_unions_overlaps_clips_to_the_window_and_keeps_busy_time() {
        let t = Instant::now();
        let ms = |n: u64| t + Duration::from_millis(n);
        // [0,10] and [5,15] overlap; [20,30] is disjoint; [40,60] is clipped at 50; [70,80] is outside.
        let value = super::coverage(vec![(ms(20), ms(30)), (ms(0), ms(10)), (ms(5), ms(15)), (ms(40), ms(60)), (ms(70), ms(80))], ms(0), ms(50));
        let close = |key: &str, expected: f64| assert!((value[key].as_f64().unwrap() - expected).abs() < 1e-6, "{key}: {value}");
        close("wall_ms", 35.);
        close("busy_ms", 40.);
        assert_eq!(value["count"], 4);
        let empty = super::coverage(Vec::new(), ms(0), ms(1));
        assert_eq!(empty["wall_ms"], 0.);
        assert!(empty["busy_ms"].as_f64().unwrap().is_sign_positive(), "empty busy time is 0, not -0: {empty}");
    }
}
