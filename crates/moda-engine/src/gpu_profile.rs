//! Opt-in per-frame GPU profiling for benches (Moda's `examples/camera_bench.rs`; export-example-absent: Moda's bench, not ported): draw calls, pipeline and bind-group
//! switches, vertex-buffer binds, passes, buffer / texture uploads, and GPU time from timestamp queries written at
//! the start and end of every render / compute pass the engine records.
//!
//! Off unless a bench calls [`enable`]; every hook is then a single relaxed atomic load. Process-global (the bench
//! drives one device), so no engine API grows a parameter for it.
//!
//! Timestamps need `wgpu::Features::TIMESTAMP_QUERY`, which [`crate::device::GpuContext::new`] requests only when
//! `MODA_GPU_TIMESTAMPS=1` (or `MODA_RENDER_PROFILE=gpu`) and the adapter has it (Metal on Apple silicon:
//! pass-boundary sampling).
//!
//! An export profiled with `MODA_RENDER_PROFILE=gpu` enables it too, and each pair carries the [`crate::prof`] span
//! open when its pass was recorded, so [`resolve_into_prof`] adds each pass's GPU time to that stage.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $(static $name: AtomicU64 = AtomicU64::new(0);)*
    };
}
counters!(
    PASSES,
    DRAWS,
    INSTANCES,
    PIPELINES,
    BIND_GROUPS,
    VERTEX_BUFFERS,
    BUFFER_WRITES,
    BUFFER_BYTES,
    TEXTURE_WRITES,
    TEXTURE_BYTES
);

/// What one frame (between two [`take_counters`] calls) recorded.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct FrameCounters {
    pub passes: u64,
    pub draws: u64,
    pub instances: u64,
    pub pipelines: u64,
    pub bind_groups: u64,
    pub vertex_buffers: u64,
    pub buffer_writes: u64,
    pub buffer_bytes: u64,
    pub texture_writes: u64,
    pub texture_bytes: u64,
}

#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

#[inline]
fn bump(c: &AtomicU64, n: u64) {
    if enabled() {
        c.fetch_add(n, Ordering::Relaxed);
    }
}

pub fn count_pass() {
    bump(&PASSES, 1);
}
pub fn count_draw(instances: u32) {
    if enabled() {
        DRAWS.fetch_add(1, Ordering::Relaxed);
        INSTANCES.fetch_add(u64::from(instances), Ordering::Relaxed);
    }
}
pub fn count_pipeline() {
    bump(&PIPELINES, 1);
}
pub fn count_bind_group() {
    bump(&BIND_GROUPS, 1);
}
pub fn count_vertex_buffer() {
    bump(&VERTEX_BUFFERS, 1);
}
pub fn count_buffer_write(bytes: usize) {
    if enabled() {
        BUFFER_WRITES.fetch_add(1, Ordering::Relaxed);
        BUFFER_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}
pub fn count_texture_write(bytes: usize) {
    if enabled() {
        TEXTURE_WRITES.fetch_add(1, Ordering::Relaxed);
        TEXTURE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    }
}

/// The counters since the last call, reset to zero.
pub fn take_counters() -> FrameCounters {
    let t = |c: &AtomicU64| c.swap(0, Ordering::Relaxed);
    FrameCounters {
        passes: t(&PASSES),
        draws: t(&DRAWS),
        instances: t(&INSTANCES),
        pipelines: t(&PIPELINES),
        bind_groups: t(&BIND_GROUPS),
        vertex_buffers: t(&VERTEX_BUFFERS),
        buffer_writes: t(&BUFFER_WRITES),
        buffer_bytes: t(&BUFFER_BYTES),
        texture_writes: t(&TEXTURE_WRITES),
        texture_bytes: t(&TEXTURE_BYTES),
    }
}

// --- timestamps ------------------------------------------------------------------------------------------

/// Query pairs per frame; passes past this are simply not timed (reported as `untimedPasses`).
const MAX_PAIRS: u32 = 1024;

struct Timer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    next_pair: u32,
    untimed: u32,
    // The stage ([`crate::prof::current`]) that recorded each pair's pass.
    labels: Vec<&'static str>,
}

thread_local! {
    // Per thread: wgpu handles are not `Send` on wasm (WebGPU), and the render loop that enabled profiling is the
    // one that times its passes. Passes begun on any other thread go untimed (counters still count them).
    static TIMER: RefCell<Option<Timer>> = const { RefCell::new(None) };
}

/// Turn the counters on, and GPU timestamps too when `device` has `TIMESTAMP_QUERY`. Returns whether
/// timestamps are available.
pub fn enable(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    ENABLED.store(true, Ordering::Relaxed);
    if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return false;
    }
    let size = u64::from(MAX_PAIRS) * 2 * 8;
    let timer = Timer {
        queries: device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu_profile timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_PAIRS * 2,
        }),
        resolve: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profile resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        readback: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_profile readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        period_ns: queue.get_timestamp_period(),
        next_pair: 0,
        untimed: 0,
        labels: Vec::new(),
    };
    TIMER.with(|t| *t.borrow_mut() = Some(timer));
    true
}

/// A begin/end timestamp pair for the next pass, or `None` when timing is off (or this frame ran out of pairs).
/// The `QuerySet` is a cheap handle clone the caller borrows for the pass descriptor.
/// Also counts the pass, so every call site calls this once per pass it begins.
pub fn pass_timestamps() -> Option<(wgpu::QuerySet, u32)> {
    if !enabled() {
        return None;
    }
    count_pass();
    TIMER.with(|cell| {
        let mut guard = cell.borrow_mut();
        let t = guard.as_mut()?;
        if t.next_pair >= MAX_PAIRS {
            t.untimed += 1;
            return None;
        }
        let pair = t.next_pair;
        t.next_pair += 1;
        t.labels.push(crate::prof::current());
        Some((t.queries.clone(), pair * 2))
    })
}

/// Render-pass timestamp writes for a pair from [`pass_timestamps`].
pub fn render_writes(
    ts: &Option<(wgpu::QuerySet, u32)>,
) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
    ts.as_ref().map(|(q, i)| wgpu::RenderPassTimestampWrites {
        query_set: q,
        beginning_of_pass_write_index: Some(*i),
        end_of_pass_write_index: Some(*i + 1),
    })
}

/// Compute-pass timestamp writes for a pair from [`pass_timestamps`].
pub fn compute_writes(
    ts: &Option<(wgpu::QuerySet, u32)>,
) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
    ts.as_ref().map(|(q, i)| wgpu::ComputePassTimestampWrites {
        query_set: q,
        beginning_of_pass_write_index: Some(*i),
        end_of_pass_write_index: Some(*i + 1),
    })
}

/// GPU time of one frame's passes.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct GpuTime {
    /// Sum of every timed pass's (end − begin), in ms: the GPU's busy time for the frame.
    pub busy_ms: f64,
    /// First begin to last end, in ms (includes gaps between submits).
    pub span_ms: f64,
    pub timed_passes: u32,
    pub untimed_passes: u32,
}

/// Resolve the timestamps written since the last call (the caller has submitted the frame), wait for them, and
/// reset for the next frame. `None` when timestamps are off.
pub fn resolve_frame(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<GpuTime> {
    TIMER.with(|cell| resolve_timer(cell.borrow_mut().as_mut()?, device, queue, |_, _| {}))
}

/// [`resolve_frame`] for a profiled export (`MODA_RENDER_PROFILE=gpu`): each timed pass's GPU time is added to
/// `gpu:<stage>` in [`crate::prof`]'s totals (one call per pass), all of them to `gpu.busy`, and the passes past a
/// frame's pairs are counted in `gpu.untimed` (calls only). Call it once the frame's passes are submitted. A no-op
/// when timestamps are off.
pub fn resolve_into_prof(device: &wgpu::Device, queue: &wgpu::Queue) {
    let Some(time) = TIMER.with(|cell| {
        resolve_timer(cell.borrow_mut().as_mut()?, device, queue, |stage, ns| {
            crate::prof::add(gpu_stage_name(stage), ns);
        })
    }) else {
        return;
    };
    if time.timed_passes > 0 {
        crate::prof::add_work("gpu.busy", (time.busy_ms * 1e6) as u128, u64::from(time.timed_passes));
    }
    if time.untimed_passes > 0 {
        crate::prof::add_work("gpu.untimed", 0, u64::from(time.untimed_passes));
    }
}

/// `gpu:<stage>`, made once per stage name (a profile has a few dozen).
fn gpu_stage_name(stage: &'static str) -> &'static str {
    static NAMES: std::sync::Mutex<Vec<(&'static str, &'static str)>> = std::sync::Mutex::new(Vec::new());
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&(_, name)) = names.iter().find(|(s, _)| *s == stage) {
        return name;
    }
    let name: &'static str = Box::leak(format!("gpu:{stage}").into_boxed_str());
    names.push((stage, name));
    name
}

fn resolve_timer(t: &mut Timer, device: &wgpu::Device, queue: &wgpu::Queue, mut per_pass: impl FnMut(&'static str, u128)) -> Option<GpuTime> {
    let pairs = t.next_pair;
    let untimed = t.untimed;
    let labels = std::mem::take(&mut t.labels);
    t.next_pair = 0;
    t.untimed = 0;
    if pairs == 0 {
        return Some(GpuTime {
            untimed_passes: untimed,
            ..Default::default()
        });
    }
    let bytes = u64::from(pairs) * 2 * 8;
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu_profile resolve"),
    });
    enc.resolve_query_set(&t.queries, 0..pairs * 2, &t.resolve, 0);
    enc.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, bytes);
    crate::scratch::flush();
    queue.submit([enc.finish()]);
    let slice = t.readback.slice(0..bytes);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let stamps: Vec<u64> = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, u64>(&data).to_vec()
    };
    t.readback.unmap();
    let to_ms = f64::from(t.period_ns) / 1e6;
    let mut busy = 0u64;
    let mut first = u64::MAX;
    let mut last = 0u64;
    for (k, p) in stamps.chunks_exact(2).enumerate() {
        let (b, e) = (p[0], p[1]);
        if b == 0 || e < b {
            continue;
        }
        busy += e - b;
        per_pass(labels.get(k).copied().unwrap_or("(none)"), ((e - b) as f64 * to_ms * 1e6) as u128);
        first = first.min(b);
        last = last.max(e);
    }
    Some(GpuTime {
        busy_ms: busy as f64 * to_ms,
        span_ms: if last > first {
            (last - first) as f64 * to_ms
        } else {
            0.0
        },
        timed_passes: pairs,
        untimed_passes: untimed,
    })
}

/// A render pass, or a render bundle being recorded for one, that counts what it records (when profiling is on).
/// The frame records its main pass through this, so the same code records a bundle a later frame replays
/// (`Engine::render_plan_to_texture_impl`, the prepared frame's bundle).
pub enum CountedPass<'p> {
    Pass(wgpu::RenderPass<'p>),
    Bundle(wgpu::RenderBundleEncoder<'p>),
}

macro_rules! each {
    ($self:ident, $e:ident => $body:expr) => {
        match $self {
            CountedPass::Pass($e) => $body,
            CountedPass::Bundle($e) => $body,
        }
    };
}

impl<'p> CountedPass<'p> {
    pub fn new(pass: wgpu::RenderPass<'p>) -> Self {
        CountedPass::Pass(pass)
    }
    pub fn bundle(encoder: wgpu::RenderBundleEncoder<'p>) -> Self {
        CountedPass::Bundle(encoder)
    }
    /// The recorded bundle (`None` for a pass).
    pub fn finish_bundle(self, label: &str) -> Option<wgpu::RenderBundle> {
        match self {
            CountedPass::Pass(_) => None,
            CountedPass::Bundle(e) => Some(e.finish(&wgpu::RenderBundleDescriptor { label: Some(label) })),
        }
    }
    #[inline]
    pub fn set_pipeline(&mut self, p: &'p wgpu::RenderPipeline) {
        count_pipeline();
        each!(self, e => e.set_pipeline(p));
    }
    #[inline]
    pub fn set_bind_group<'a, B: Into<Option<&'a wgpu::BindGroup>>>(
        &mut self,
        index: u32,
        bg: B,
        offsets: &[wgpu::DynamicOffset],
    ) {
        count_bind_group();
        let bg: Option<&wgpu::BindGroup> = bg.into();
        each!(self, e => e.set_bind_group(index, bg, offsets));
    }
    #[inline]
    pub fn set_vertex_buffer(&mut self, slot: u32, slice: wgpu::BufferSlice<'p>) {
        count_vertex_buffer();
        each!(self, e => e.set_vertex_buffer(slot, slice));
    }
    #[inline]
    pub fn set_index_buffer(&mut self, slice: wgpu::BufferSlice<'p>, format: wgpu::IndexFormat) {
        each!(self, e => e.set_index_buffer(slice, format));
    }
    #[inline]
    pub fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        count_draw(instances.end.saturating_sub(instances.start));
        each!(self, e => e.draw(vertices, instances));
    }
    #[inline]
    pub fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        count_draw(instances.end.saturating_sub(instances.start));
        each!(self, e => e.draw_indexed(indices, base_vertex, instances));
    }
    /// A pass's viewport (a bundle can't set one: the frame never records a bundle with a region viewport).
    pub fn set_viewport(&mut self, x: f32, y: f32, w: f32, h: f32, min_depth: f32, max_depth: f32) {
        match self {
            CountedPass::Pass(p) => p.set_viewport(x, y, w, h, min_depth, max_depth),
            CountedPass::Bundle(_) => debug_assert!(false, "no viewport in a render bundle"),
        }
    }
    /// A pass's scissor rectangle (not recorded into bundles: the frame never bundles a pass that scissors).
    pub fn set_scissor_rect(&mut self, x: u32, y: u32, w: u32, h: u32) {
        match self {
            CountedPass::Pass(p) => p.set_scissor_rect(x, y, w, h),
            CountedPass::Bundle(_) => debug_assert!(false, "no scissor in a render bundle"),
        }
    }
    /// Replay bundles into a pass.
    pub fn execute_bundles<'a>(&mut self, bundles: impl IntoIterator<Item = &'a wgpu::RenderBundle>) {
        if let CountedPass::Pass(p) = self {
            p.execute_bundles(bundles);
        }
    }
}
