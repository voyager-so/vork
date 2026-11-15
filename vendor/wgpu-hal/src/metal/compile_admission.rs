//! Admission of Metal compiler work.
//!
//! Metal compiles libraries and pipeline states out of process, in one
//! `MTLCompilerService` per client process. Every concurrent compilation of
//! this process therefore shares that service's memory. When the service's
//! allocator runs out (`malloc: Failed to allocate segment from range group -
//! out of space`) the service dies, and every compilation in flight fails
//! with `XPC_ERROR_CONNECTION_INTERRUPTED` once Metal's own retries, which
//! restart the same concurrent load, have failed too.
//!
//! This module bounds the estimated service memory of the compilations in
//! flight. It is process-wide because the service is: every device, every
//! library compilation and every pipeline-state compilation (render and
//! compute, with or without a binary archive) passes through it. A binary
//! archive hit does not compile and is not admitted.
//!
//! Order: a compilation starts as soon as its estimate fits beside the ones in
//! flight, so a small one passes a large one that is itself waiting. A
//! compilation whose estimate alone exceeds the limit starts when nothing else
//! is in flight. Passing is bounded so that every compilation starts: later
//! arrivals may pass a waiting compilation until their estimates add up to the
//! limit, one full budget of other work; after that, later arrivals wait
//! behind it, what is in flight drains, and it starts.

use alloc::{
    collections::BTreeMap,
    string::{String, ToString as _},
};
use std::sync::{Condvar, Mutex, OnceLock};

use objc2_foundation::{NSError, NSProcessInfo};

/// Service memory per byte of submitted source.
///
/// Measured on an M2 Ultra, macOS 27.2: compiling one pipeline state from
/// 2,220,880 bytes of MSL alone raised the service's resident size to
/// 5,308,592 KiB, which is 2,448 bytes per source byte. Source size is the only
/// cost signal available before compiling, and this is the largest ratio
/// measured, so other sources are charged as if they were as expensive.
const SERVICE_BYTES_PER_SOURCE_BYTE: u64 = 2_448;

/// The largest resident size at which a compiler service was seen alive.
///
/// Same machine: with 5 to 24 such compilations in flight, twelve services in
/// a row died of allocator exhaustion; the largest size sampled before a death
/// was 14,563,888 KiB. The machine has 128 GiB, so this is a ceiling of the
/// service process, not of physical memory.
const SERVICE_CEILING_BYTES: u64 = 14_563_888 * 1024;

/// How Metal's driver describes a compilation lost with its compiler service.
/// The description of the failure this module exists for, in full: "Compilation
/// failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED.
/// This error occurred after multiple retries."
const INTERRUPTED: &str =
    "Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED";

/// One failed attempt, classified from the native error before it became text.
pub(super) struct Failure {
    message: String,
    interrupted: bool,
}

impl Failure {
    /// `message` is what the caller reports for `error`.
    pub(super) fn new(error: &NSError, message: String) -> Self {
        let interrupted = is_interruption(&error.localizedDescription().to_string());
        Self { message, interrupted }
    }
}

/// Whether an error's own description is the driver's report of a lost
/// service. Metal reports it under more than one domain and code (the driver's
/// domain for a pipeline state, the library domain's compile failure for a
/// library), so those do not identify it; the description does. It must be the
/// driver's sentence from its first byte: a compiler diagnostic begins with its
/// source location and only quotes the source after it, so a source that does
/// not compile is never an interruption, whatever it contains.
fn is_interruption(description: &str) -> bool {
    description.starts_with(INTERRUPTED)
}

/// Estimated service memory for compiling `source_bytes` of source.
fn estimate(source_bytes: u64) -> u64 {
    source_bytes.saturating_mul(SERVICE_BYTES_PER_SOURCE_BYTE)
}

/// The estimate allowed in flight on this machine: the service ceiling, or
/// half of physical memory when that is smaller. The service's memory is the
/// machine's unified memory, which the renderer and the GPU also allocate
/// from, so compilers are held to half of it.
fn machine_limit() -> u64 {
    SERVICE_CEILING_BYTES.min(NSProcessInfo::processInfo().physicalMemory() / 2)
}

#[derive(Default)]
struct State {
    /// Sum of the estimates in flight.
    estimated: u64,
    /// Compilations in flight.
    running: usize,
    /// Retries waiting for, or holding, the compiler alone.
    recovering: usize,
    /// Arrival number of the next compilation that has to wait.
    arrivals: u64,
    /// Waiting compilations by arrival, each with the estimates admitted past it.
    waiting: BTreeMap<u64, u64>,
}

pub(super) struct Admission {
    limit: u64,
    state: Mutex<State>,
    changed: Condvar,
}

/// One admitted compilation; frees its place when dropped.
struct Permit<'a> {
    admission: &'a Admission,
    estimate: u64,
    recovery: bool,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut state = self.admission.state.lock().unwrap();
        state.estimated -= self.estimate;
        state.running -= 1;
        if self.recovery {
            state.recovering -= 1;
        }
        drop(state);
        self.admission.changed.notify_all();
    }
}

impl Admission {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            state: Mutex::default(),
            changed: Condvar::new(),
        }
    }

    /// Admit beside the compilations in flight, once no retry is pending.
    ///
    /// An arrival that fits may pass the compilations already waiting, and is
    /// charged to each one it passes. Once a waiting compilation has been
    /// passed by a full limit of estimates, nothing that arrived after it
    /// starts before it does.
    fn admit(&self, estimate: u64) -> Permit<'_> {
        // Work with no estimate still occupies the compiler, so it still counts as passing.
        let passing = estimate.max(1);
        let mut state = self.state.lock().unwrap();
        let mut arrival = None;
        loop {
            let fits = state.recovering == 0
                && (state.running == 0 || state.estimated.saturating_add(estimate) <= self.limit);
            // Everything waiting arrived before a compilation that has not had to wait.
            let may_pass = state
                .waiting
                .range(..arrival.unwrap_or(u64::MAX))
                .all(|(_, passed)| passed.saturating_add(passing) <= self.limit);
            if fits && may_pass {
                break;
            }
            if arrival.is_none() {
                let number = state.arrivals;
                state.arrivals += 1;
                state.waiting.insert(number, 0);
                arrival = Some(number);
            }
            state = self.changed.wait(state).unwrap();
        }
        if let Some(arrival) = arrival {
            state.waiting.remove(&arrival);
        }
        for (_, passed) in state.waiting.range_mut(..arrival.unwrap_or(u64::MAX)) {
            *passed = passed.saturating_add(passing);
        }
        state.estimated += estimate;
        state.running += 1;
        drop(state);
        // A waiter leaving the line can unblock the ones that arrived after it.
        if arrival.is_some() {
            self.changed.notify_all();
        }
        Permit {
            admission: self,
            estimate,
            recovery: false,
        }
    }

    /// Admit alone: stop new admissions, then wait for everything in flight
    /// (other retries included) to finish.
    fn admit_alone(&self, estimate: u64) -> Permit<'_> {
        let mut state = self.state.lock().unwrap();
        state.recovering += 1;
        let mut state = self
            .changed
            .wait_while(state, |state| state.running != 0)
            .unwrap();
        state.estimated += estimate;
        state.running += 1;
        Permit {
            admission: self,
            estimate,
            recovery: true,
        }
    }

    /// Run one compilation of `source_bytes` of source under admission.
    ///
    /// `attempt` is everything that can reach the compiler for this work: for
    /// a pipeline with a binary archive, the compilation and its recording in
    /// the archive. When it fails because the compiler service was interrupted,
    /// the service died under the compilations sharing it, not because of this
    /// source. It is then attempted once more, alone. Any other failure,
    /// including a source that does not compile, is returned as is.
    pub(super) fn compile<T>(
        &self,
        label: Option<&str>,
        source_bytes: u64,
        attempt: impl Fn() -> Result<T, Failure>,
    ) -> Result<T, String> {
        let estimate = estimate(source_bytes);
        let first = {
            let _permit = self.admit(estimate);
            attempt()
        };
        let interruption = match first {
            Err(failure) if failure.interrupted => failure.message,
            other => return other.map_err(|failure| failure.message),
        };
        let label = label.unwrap_or("unnamed");
        log::warn!(
            "Metal compiler service interrupted while compiling {label:?} ({source_bytes} source bytes, \
             {estimate} bytes estimated of {} allowed); retrying once with nothing else compiling",
            self.limit,
        );
        let _permit = self.admit_alone(estimate);
        match attempt() {
            Ok(compiled) => {
                log::warn!("Metal compiled {label:?} on the retry");
                Ok(compiled)
            }
            Err(Failure { message: error, .. }) => Err(format!(
                "{error}; this was the one retry with nothing else compiling ({source_bytes} source bytes), \
                 after the compiler service was interrupted: {interruption}"
            )),
        }
    }
}

/// Compiler work postponed past the permit of the compilation it belongs to:
/// recording a compiled pipeline in a binary archive whose lock was taken. It
/// keeps what that compilation was admitted with, and can be repeated for the
/// retry after a service interruption.
pub(super) struct Deferred {
    label: Option<String>,
    source_bytes: u64,
    work: alloc::boxed::Box<dyn Fn() -> Result<(), Failure>>,
}

impl Deferred {
    pub(super) fn new(label: Option<&str>, source_bytes: u64, work: impl Fn() -> Result<(), Failure> + 'static) -> Self {
        Self { label: label.map(String::from), source_bytes, work: alloc::boxed::Box::new(work) }
    }
    pub(super) fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
    pub(super) fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
}

impl Admission {
    /// Run postponed work, each piece as its own admitted request with its own
    /// estimate and the same retry as a compilation.
    ///
    /// The caller must not hold a permit of this admission: admission waits
    /// for the work in flight to finish, which would include the caller's own.
    /// Work that fails is only missing from its archive; its pipeline exists.
    pub(super) fn record(&self, deferred: impl IntoIterator<Item = Deferred>) {
        for deferred in deferred {
            if let Err(error) = self.compile(deferred.label(), deferred.source_bytes, || (deferred.work)()) {
                log::warn!("the Metal binary archive refused a compiled pipeline: {error}");
            }
        }
    }
}

/// The admission shared by every Metal device of this process.
pub(super) fn service() -> &'static Admission {
    static SERVICE: OnceLock<Admission> = OnceLock::new();
    SERVICE.get_or_init(|| Admission::new(machine_limit()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::{sync::mpsc, thread, time::Duration};

    const HEAVY_SOURCE: u64 = 2_220_880;

    fn admission(heavy_compilations: u64) -> Arc<Admission> {
        Arc::new(Admission::new(estimate(HEAVY_SOURCE) * heavy_compilations))
    }

    /// Start a compilation that reports when it is admitted and holds its
    /// place until released.
    fn hold(
        admission: &Arc<Admission>,
        source_bytes: u64,
    ) -> (mpsc::Receiver<()>, mpsc::Sender<()>, thread::JoinHandle<Result<(), String>>) {

        let (admitted, is_admitted) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let admission = admission.clone();
        let released = Mutex::new(released);
        let worker = thread::spawn(move || {
            admission.compile(None, source_bytes, || {
                admitted.send(()).unwrap();
                released.lock().unwrap().recv().unwrap();
                Ok(())
            })
        });
        (is_admitted, release, worker)
    }

    /// A source diagnostic that quotes the driver's whole report.
    const DIAGNOSTIC: &str = "program_source:1:2: error: Compilation failed due to an interrupted connection: \
                              XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries.";

    fn interruption() -> Failure {
        let description = format!("{INTERRUPTED}. This error occurred after multiple retries.");
        Failure { interrupted: is_interruption(&description), message: description }
    }

    #[test]
    fn only_the_drivers_own_report_is_an_interruption() {
        assert!(interruption().interrupted);
        assert!(!is_interruption(DIAGNOSTIC));
        assert!(!is_interruption("Compiler encountered an internal error"));
    }

    #[test]
    fn a_waiting_compilation_is_passed_by_at_most_one_limit_of_estimates() {
        let admission = admission(1);
        // Each small compilation is two fifths of the limit: two fit together, and two may pass.
        let small = HEAVY_SOURCE * 2 / 5;
        let (a_in, a_release, a) = hold(&admission, small);
        a_in.recv().unwrap();
        let (large_in, large_release, large) = hold(&admission, HEAVY_SOURCE * 3);
        assert!(large_in.recv_timeout(SETTLE).is_err());

        // Overlapping small compilations keep one in flight, which the large one cannot join.
        let (b_in, b_release, b) = hold(&admission, small);
        b_in.recv().unwrap();
        a_release.send(()).unwrap();
        a.join().unwrap().unwrap();
        let (c_in, c_release, c) = hold(&admission, small);
        c_in.recv().unwrap();
        b_release.send(()).unwrap();
        b.join().unwrap().unwrap();

        // A third would fit beside the one in flight, but the large one has been passed enough.
        let (d_in, d_release, d) = hold(&admission, small);
        assert!(d_in.recv_timeout(SETTLE).is_err());
        assert!(large_in.recv_timeout(SETTLE).is_err());
        c_release.send(()).unwrap();
        c.join().unwrap().unwrap();
        large_in.recv().unwrap();
        assert!(d_in.recv_timeout(SETTLE).is_err());
        large_release.send(()).unwrap();
        large.join().unwrap().unwrap();
        d_in.recv().unwrap();
        d_release.send(()).unwrap();
        d.join().unwrap().unwrap();
        let state = admission.state.lock().unwrap();
        assert!(state.waiting.is_empty());
    }

    const SETTLE: Duration = Duration::from_millis(200);

    #[test]
    fn measured_ceiling_admits_two_heavy_compilations_and_not_three() {
        assert!(estimate(HEAVY_SOURCE) * 2 <= SERVICE_CEILING_BYTES);
        assert!(estimate(HEAVY_SOURCE) * 3 > SERVICE_CEILING_BYTES);
    }

    #[test]
    fn heavy_compilations_queue_and_a_small_one_passes_them() {
        let admission = admission(2);
        let (first_in, first_release, first) = hold(&admission, HEAVY_SOURCE);
        let (second_in, second_release, second) = hold(&admission, HEAVY_SOURCE - 100_000);
        first_in.recv().unwrap();
        second_in.recv().unwrap();

        let (third_in, third_release, third) = hold(&admission, HEAVY_SOURCE);
        assert!(third_in.recv_timeout(SETTLE).is_err(), "a third heavy compilation must wait");

        // The small one fits in what the two in flight leave, although a heavy one is waiting.
        let (small_in, small_release, small) = hold(&admission, 50_000);
        small_in.recv().unwrap();
        small_release.send(()).unwrap();
        small.join().unwrap().unwrap();

        first_release.send(()).unwrap();
        first.join().unwrap().unwrap();
        third_in.recv().unwrap();
        for release in [second_release, third_release] {
            release.send(()).unwrap();
        }
        for worker in [second, third] {
            worker.join().unwrap().unwrap();
        }
    }

    #[test]
    fn a_compilation_over_the_limit_runs_alone() {
        let admission = admission(1);
        let (small_in, small_release, small) = hold(&admission, 50_000);
        small_in.recv().unwrap();
        let (large_in, large_release, large) = hold(&admission, HEAVY_SOURCE * 3);
        assert!(large_in.recv_timeout(SETTLE).is_err());
        small_release.send(()).unwrap();
        small.join().unwrap().unwrap();
        large_in.recv().unwrap();
        large_release.send(()).unwrap();
        large.join().unwrap().unwrap();
    }

    #[test]
    fn an_interrupted_compilation_retries_once_alone() {
        let admission = admission(2);
        let attempts = AtomicUsize::new(0);
        let (other_in, other_release, other) = hold(&admission, 50_000);
        other_in.recv().unwrap();

        thread::scope(|scope| {
            let retried = scope.spawn(|| {
                admission.compile(Some("heavy"), HEAVY_SOURCE, || {
                    match attempts.fetch_add(1, Ordering::SeqCst) {
                        0 => Err(interruption()),
                        _ => {
                            // The retry holds the compiler alone.
                            let state = admission.state.lock().unwrap();
                            assert_eq!((state.running, state.recovering), (1, 1));
                            Ok(())
                        }
                    }
                })
            });
            // The retry waits for the compilation in flight, and nothing new starts ahead of it.
            thread::sleep(SETTLE);
            assert_eq!(attempts.load(Ordering::SeqCst), 1);
            let (late_in, late_release, late) = hold(&admission, 50_000);
            assert!(late_in.recv_timeout(SETTLE).is_err());
            other_release.send(()).unwrap();
            other.join().unwrap().unwrap();
            retried.join().unwrap().unwrap();
            late_in.recv().unwrap();
            late_release.send(()).unwrap();
            late.join().unwrap().unwrap();
        });
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    /// A recording deferred by a heavy pipeline is drained later, by the next
    /// pipeline or a serialization. It is admitted with the heavy pipeline's
    /// estimate, not run under whatever permit the drainer's own work has.
    #[test]
    fn a_deferred_recording_is_admitted_with_its_own_estimate() {
        let admission = admission(1);
        let (small_in, small_release, small) = hold(&admission, 50_000);
        small_in.recv().unwrap();

        let (started, has_started) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let drain = {
            let admission = admission.clone();
            thread::spawn(move || {
                let released = Mutex::new(released);
                admission.record([Deferred::new(Some("heavy"), HEAVY_SOURCE, move || {
                    started.send(()).unwrap();
                    released.lock().unwrap().recv().unwrap();
                    Ok(())
                })]);
            })
        };
        // The heavy estimate does not fit beside the small compilation, so it waits for it.
        assert!(has_started.recv_timeout(SETTLE).is_err(), "a deferred recording ran outside admission");
        small_release.send(()).unwrap();
        small.join().unwrap().unwrap();
        has_started.recv().unwrap();
        {
            let state = admission.state.lock().unwrap();
            assert_eq!((state.estimated, state.running), (estimate(HEAVY_SOURCE), 1));
        }
        release.send(()).unwrap();
        drain.join().unwrap();
        let state = admission.state.lock().unwrap();
        assert_eq!((state.estimated, state.running, state.recovering), (0, 0, 0));
    }

    /// A drain that meets a recovery, as a serialization can, does not reach
    /// the compiler until the retry that holds it alone has finished.
    #[test]
    fn a_deferred_recording_waits_for_a_recovery() {
        let admission = admission(2);
        let attempts = AtomicUsize::new(0);
        let recorded = Arc::new(AtomicUsize::new(0));
        let (other_in, other_release, other) = hold(&admission, 50_000);
        other_in.recv().unwrap();
        let (recovering, is_recovering) = mpsc::channel();
        let (resume, resumed) = mpsc::channel::<()>();
        let resumed = Mutex::new(resumed);

        thread::scope(|scope| {
            let retried = scope.spawn(|| {
                admission.compile(Some("heavy"), HEAVY_SOURCE, || match attempts.fetch_add(1, Ordering::SeqCst) {
                    0 => Err(interruption()),
                    _ => {
                        recovering.send(()).unwrap();
                        resumed.lock().unwrap().recv().unwrap();
                        Ok(())
                    }
                })
            });
            // The retry is pending behind the compilation in flight.
            thread::sleep(SETTLE);
            assert_eq!(attempts.load(Ordering::SeqCst), 1);
            let drain = scope.spawn(|| {
                let recorded = recorded.clone();
                admission.record([Deferred::new(Some("deferred"), 50_000, move || {
                    recorded.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })]);
            });
            // Neither a pending retry nor one holding the compiler lets the recording start.
            thread::sleep(SETTLE);
            assert_eq!(recorded.load(Ordering::SeqCst), 0);
            other_release.send(()).unwrap();
            other.join().unwrap().unwrap();
            is_recovering.recv().unwrap();
            thread::sleep(SETTLE);
            assert_eq!(recorded.load(Ordering::SeqCst), 0, "a deferred recording overlapped the exclusive retry");
            resume.send(()).unwrap();
            retried.join().unwrap().unwrap();
            drain.join().unwrap();
        });
        assert_eq!(recorded.load(Ordering::SeqCst), 1);
        let state = admission.state.lock().unwrap();
        assert_eq!((state.estimated, state.running, state.recovering), (0, 0, 0));
    }

    /// An interrupted deferred recording gets the same single retry alone. One
    /// the archive refuses is dropped, and the recordings after it still run.
    #[test]
    fn an_interrupted_deferred_recording_retries_alone_and_a_refused_one_is_dropped() {
        let admission = admission(2);
        let attempts = Arc::new(AtomicUsize::new(0));
        admission.record([Deferred::new(Some("deferred"), HEAVY_SOURCE, {
            let (attempts, admission) = (attempts.clone(), admission.clone());
            move || match attempts.fetch_add(1, Ordering::SeqCst) {
                0 => Err(interruption()),
                _ => {
                    let state = admission.state.lock().unwrap();
                    assert_eq!((state.running, state.recovering), (1, 1));
                    Ok(())
                }
            }
        })]);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);

        let (refused, next) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        admission.record([
            Deferred::new(None, 1_000, {
                let refused = refused.clone();
                move || {
                    refused.fetch_add(1, Ordering::SeqCst);
                    Err(Failure { message: DIAGNOSTIC.into(), interrupted: is_interruption(DIAGNOSTIC) })
                }
            }),
            Deferred::new(None, 1_000, {
                let next = next.clone();
                move || {
                    next.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
            }),
        ]);
        assert_eq!((refused.load(Ordering::SeqCst), next.load(Ordering::SeqCst)), (1, 1));
        let state = admission.state.lock().unwrap();
        assert_eq!((state.estimated, state.running, state.recovering), (0, 0, 0));
        assert!(state.waiting.is_empty());
    }

    #[test]
    fn a_second_interruption_and_an_invalid_source_are_not_retried() {
        let admission = admission(2);
        let attempts = AtomicUsize::new(0);
        let error = admission
            .compile(Some("heavy"), HEAVY_SOURCE, || -> Result<(), Failure> {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(interruption())
            })
            .unwrap_err();
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(error.contains("the one retry"), "{error}");

        let attempts = AtomicUsize::new(0);
        let error = admission
            .compile(None, 1_000, || -> Result<(), Failure> {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(Failure { message: DIAGNOSTIC.into(), interrupted: is_interruption(DIAGNOSTIC) })
            })
            .unwrap_err();
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(error, DIAGNOSTIC);
        let state = admission.state.lock().unwrap();
        assert_eq!((state.estimated, state.running, state.recovering), (0, 0, 0));
        assert!(state.waiting.is_empty());
    }
}
