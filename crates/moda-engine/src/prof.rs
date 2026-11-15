//! Opt-in wall-clock stage totals for the export path (`MODA_RENDER_PROFILE=1`): the perf-export bench
//! (scripts/bench/perf-export) reads them to split an export into evaluation, scene build, planning, GPU work,
//! readback, pixel-graph work and the encoder's back-pressure.
//!
//! Off by default: every [`span`] is then one relaxed atomic load and no clock read. Totals are process-global and
//! inclusive (a span inside another counts in both); names say where they nest (`frame.draw`, `frame.draw.plan`).
//!
//! `MODA_RENDER_PROFILE=gpu` also times every GPU pass the export thread records (`gpu_profile`'s timestamps) and adds
//! each pass's GPU time to `gpu:<stage>`, the innermost span open when the pass was recorded ([`current`]), plus
//! `gpu.busy` for all of them. A frame's timestamps are resolved at its end ([`crate::gpu_profile::resolve_into_prof`]),
//! which waits for the GPU: wall times in this mode lose the CPU/GPU overlap, the `gpu:` totals don't.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static STATE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 off, 2 on
static TOTALS: Mutex<Vec<(&'static str, u128, u64)>> = Mutex::new(Vec::new());

/// Whether profiling is on (`MODA_RENDER_PROFILE` set to anything but `0` or empty).
#[inline]
pub fn on() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var("MODA_RENDER_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Whether the profile is `MODA_RENDER_PROFILE=gpu` (the module docs).
pub fn gpu() -> bool {
    on() && std::env::var("MODA_RENDER_PROFILE").is_ok_and(|v| v == "gpu")
}

thread_local! {
    // The spans open on this thread, outermost first: what [`current`] names a GPU pass after.
    static OPEN: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// The innermost span open on this thread (`"(none)"` outside every span, or with profiling off).
pub fn current() -> &'static str {
    OPEN.with(|open| open.borrow().last().copied().unwrap_or("(none)"))
}

/// A timed stage: its wall time is added to `name`'s total when it drops.
pub struct Span {
    name: &'static str,
    start: Option<Instant>,
    // Its place in this thread's open spans; dropping it closes it and any span opened after it.
    depth: usize,
}

/// Time a stage until the returned guard drops.
#[inline]
pub fn span(name: &'static str) -> Span {
    if !on() {
        return Span { name, start: None, depth: 0 };
    }
    let depth = OPEN.with(|open| {
        let mut open = open.borrow_mut();
        open.push(name);
        open.len() - 1
    });
    Span { name, start: Some(Instant::now()), depth }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(s) = self.start {
            add(self.name, s.elapsed().as_nanos());
            OPEN.with(|open| open.borrow_mut().truncate(self.depth));
        }
    }
}

/// Add `ns` to `name`'s total (one more call).
pub fn add(name: &'static str, ns: u128) {
    add_work(name, ns, 1);
}

pub fn add_work(name: &'static str, ns: u128, calls: u64) {
    let mut t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    match t.iter_mut().find(|(n, _, _)| *n == name) {
        Some(e) => {
            e.1 += ns;
            e.2 += calls;
        }
        None => t.push((name, ns, calls)),
    }
}

/// The totals so far as one JSON object: `{ "<stage>": { "ms": …, "calls": … }, … }`, in first-seen order.
pub fn report_json() -> String {
    let t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let parts: Vec<String> = t
        .iter()
        .map(|(n, ns, c)| format!("\"{n}\":{{\"ms\":{:.1},\"calls\":{c}}}", *ns as f64 / 1e6))
        .collect();
    format!("{{{}}}", parts.join(","))
}

/// With `MODA_RENDER_PROFILE=sync`, wait for the GPU to finish what was submitted, so the enclosing span holds its
/// GPU time rather than leaving it to the next readback's wait. A no-op otherwise.
pub fn sync(device: &wgpu::Device) {
    if on() && std::env::var("MODA_RENDER_PROFILE").is_ok_and(|v| v == "sync") {
        crate::scratch::flush();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// [`sync`] when dropped: declared after a [`span`], it drops first, so the span holds the GPU time of what it
/// submitted (with `MODA_RENDER_PROFILE=sync`; a no-op otherwise).
pub struct SyncOnDrop<'a>(pub &'a wgpu::Device);

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        sync(self.0);
    }
}
