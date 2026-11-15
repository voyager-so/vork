//! Optional process progress, enabled by the CLI. No output or observer thread in library users.
use std::sync::{Arc, Mutex, OnceLock, atomic::{AtomicUsize, Ordering}};
use std::time::Instant;

#[derive(Clone, Default)]
pub struct State {
    pub phase: &'static str,
    pub total: usize,
    pub samples_done: Arc<AtomicUsize>,
    pub samples_recorded: usize,
    pub samples_total: usize,
    pub shaders_started: usize,
    pub shaders_done: usize,
    pub pipelines_started: usize,
    pub pipelines_done: usize,
}
static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static COMPLETED: AtomicUsize = AtomicUsize::new(0);

pub fn enable(phase: &'static str) {
    *STATE.get_or_init(Default::default).lock().unwrap() = State { phase, ..Default::default() };
    COMPLETED.store(0, Ordering::Relaxed);
}
pub fn phase(phase: &'static str, total: usize) {
    if let Some(state) = STATE.get() {
        let mut state = state.lock().unwrap();
        state.phase = phase;
        state.total = total;
        state.samples_recorded = 0;
        state.samples_total = 0;
        COMPLETED.store(0, Ordering::Relaxed);
    }
}
/// Recording progress is distinct from completed GPU work; never call it samples rendered.
pub fn sample_pass(recorded: usize, total: usize) {
    if let Some(state) = STATE.get() {
        let mut state = state.lock().unwrap();
        if recorded == 0 { state.samples_done = Arc::default(); }
        state.samples_recorded = recorded;
        state.samples_total = total;
    }
}
/// A frame-local counter: callbacks from older frames cannot update a newer frame.
pub fn sample_counter() -> Option<Arc<AtomicUsize>> {
    STATE.get().map(|state| state.lock().unwrap().samples_done.clone())
}
/// One atomic update per delivered frame/setup configuration; never called in a sample or GPU loop.
pub fn completed(count: usize) { COMPLETED.fetch_max(count, Ordering::Relaxed); }
pub fn snapshot() -> (State, usize) {
    (STATE.get().expect("progress enabled").lock().unwrap().clone(), COMPLETED.load(Ordering::Relaxed))
}
/// Creation events, for the CLI that owns this process's stderr. Like the
/// progress above they are process-wide and off unless the CLI turns them on,
/// once, for its own command: a library user or a retained session writes
/// nothing. While on, every announced creation writes one `creation: {json}`
/// line when it begins and one when it ends, so a process that is stopped
/// mid-compile has still said what it finished, how long each took, and what
/// was in flight.
static EVENTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
pub fn events(on: bool) { EVENTS.store(on, Ordering::Relaxed); }

/// One announced creation: its `begin` line is written by `announce`, its
/// `end` line, with the milliseconds between, when this is dropped. Every
/// `end` therefore has a `begin`.
pub struct Announced { kind: &'static str, label: String, start: Instant }
/// Announce a creation of KIND named LABEL. None, and no output, unless the CLI enabled events.
pub fn announce(kind: &'static str, label: &str) -> Option<Announced> {
    if !EVENTS.load(Ordering::Relaxed) { return None; }
    eprintln!("creation: {{\"event\":\"begin\",\"kind\":{},\"label\":{}}}", json_string(kind), json_string(label));
    Some(Announced { kind, label: label.to_owned(), start: Instant::now() })
}
impl Drop for Announced {
    fn drop(&mut self) {
        eprintln!("creation: {{\"event\":\"end\",\"kind\":{},\"label\":{},\"ms\":{:.3}}}",
            json_string(self.kind), json_string(&self.label), self.start.elapsed().as_secs_f64() * 1000.);
    }
}
fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Count Slang invocations and driver pipeline creations, including driver-only work.
/// Slang jobs also enter the pass timeline, whether or not progress is enabled.
pub struct Job { driver: bool, enabled: bool, pass: Option<(Pass, Instant)>, announced: Option<Announced> }
impl Job {
    /// A code-generating Slang invocation (`driver == false`) or a driver pipeline creation.
    pub fn start(driver: bool) -> Self {
        Self::begin(driver, (!driver).then_some(Pass::CodeGeneration))
    }
    /// A Slang invocation that resolves, preprocesses and checks without emitting code.
    pub fn front_end() -> Self {
        Self::begin(false, Some(Pass::FrontEnd))
    }
    fn begin(driver: bool, pass: Option<Pass>) -> Self {
        let enabled = STATE.get().is_some();
        if let Some(state) = STATE.get() {
            let mut state = state.lock().unwrap();
            if driver { state.pipelines_started += 1; } else { state.shaders_started += 1; }
        }
        Self { driver, enabled, pass: pass.map(|pass| (pass, Instant::now())), announced: None }
    }
    /// Name the job, which announces it (see `events`): a Slang front end, a
    /// Slang code generation or a driver pipeline, by what it is.
    pub fn named(mut self, label: &str) -> Self {
        let kind = match (self.driver, self.pass) {
            (true, _) => "pipeline",
            (false, Some((Pass::FrontEnd, _))) => "Slang front end",
            (false, _) => "Slang",
        };
        self.announced = announce(kind, label);
        self
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        if let Some((pass, start)) = self.pass {
            record(pass, start, Instant::now());
        }
        if self.enabled {
            let mut state = STATE.get().unwrap().lock().unwrap();
            if self.driver { state.pipelines_done += 1; } else { state.shaders_done += 1; }
        }
    }
}

/// Slang compiler work by kind. A code-generation invocation repeats the
/// front end in its own process; the two kinds are separate invocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass { FrontEnd, CodeGeneration }

/// Bounded process timeline of completed Slang invocations. Renders read the
/// passes overlapping their own wall interval; older entries are dropped first.
struct Timeline { passes: Vec<(Pass, Instant, Instant)>, dropped_before: Option<Instant> }
static TIMELINE: Mutex<Timeline> = Mutex::new(Timeline { passes: Vec::new(), dropped_before: None });
const TIMELINE_CAPACITY: usize = 1 << 16;

fn record(pass: Pass, start: Instant, end: Instant) {
    let mut timeline = TIMELINE.lock().unwrap_or_else(|poison| poison.into_inner());
    if timeline.passes.len() >= TIMELINE_CAPACITY {
        let drop = TIMELINE_CAPACITY / 2;
        let last = timeline.passes.drain(..drop).map(|(_, _, end)| end).max();
        timeline.dropped_before = timeline.dropped_before.max(last);
    }
    timeline.passes.push((pass, start, end));
}

/// Passes that overlap `[start, end]`, and whether entries that could overlap
/// it were already dropped from the bounded timeline.
pub fn passes(start: Instant, end: Instant) -> (Vec<(Pass, Instant, Instant)>, bool) {
    let timeline = TIMELINE.lock().unwrap_or_else(|poison| poison.into_inner());
    let overlapping = timeline.passes.iter().copied()
        .filter(|&(_, s, e)| s < end && e > start).collect();
    (overlapping, timeline.dropped_before.is_some_and(|dropped| dropped > start))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slang_jobs_enter_the_pass_timeline_by_kind_and_driver_jobs_do_not() {
        let start = Instant::now();
        drop(Job::front_end());
        drop(Job::start(false));
        drop(Job::start(true));
        let (passes, truncated) = passes(start, Instant::now());
        assert!(!truncated);
        let kinds: Vec<_> = passes.iter().map(|&(kind, _, _)| kind).collect();
        // Other tests in this process can add passes; ours are present and ordered.
        let front = kinds.iter().position(|&k| k == Pass::FrontEnd).expect("front end recorded");
        assert!(kinds[front..].contains(&Pass::CodeGeneration));
        assert!(passes.iter().all(|&(_, s, e)| s <= e));
        assert!(super::passes(Instant::now(), Instant::now()).0.is_empty());
    }

    #[test]
    fn recording_does_not_count_as_a_delivered_frame_and_resets_at_phase_changes() {
        enable("test");
        phase("Rendering frames", 2);
        sample_pass(32, 64);
        let (state, done) = snapshot();
        assert_eq!(done, 0);
        assert_eq!((state.samples_recorded, state.samples_total), (32, 64));
        let old_frame = sample_counter().unwrap();
        sample_pass(0, 32);
        old_frame.store(64, Ordering::Relaxed);
        assert_eq!(sample_counter().unwrap().load(Ordering::Relaxed), 0);
        completed(1);
        phase("Finishing output", 0);
        let (state, done) = snapshot();
        assert_eq!(done, 0);
        assert_eq!((state.samples_recorded, state.samples_total), (0, 0));
    }
}
