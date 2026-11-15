//! Reuse the bounded command-validation workers across exposures.
use std::sync::{Arc, Mutex, MutexGuard, mpsc};

/// Ordered recording segments. Streaming plans publish each completed segment
/// immediately; deferred plans retain it, unfinished, until their packed upload
/// is published. Encoders made here bound their passes per segment where the
/// backend admits a bounded number of unsubmitted native command buffers.
/// Recording hands segments over automatically at that bound, so no recording's
/// native buffer count grows with resolution, bands, samples or work blocks.
#[derive(Default)]
pub(crate) struct Segments {
    state: Arc<Mutex<State>>,
}
#[derive(Default)]
struct State {
    encoders: Vec<wgpu::CommandEncoder>,
    stream: bool,
    segmented: bool,
    finish_ms: f64,
}
impl State {
    fn complete(&mut self, queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) {
        self.segmented = true;
        if self.stream {
            let (_, finish_ms) = Segments::finish_encoder(queue, encoder);
            self.finish_ms += finish_ms;
        } else { self.encoders.push(encoder); }
    }
}
impl Segments {
    fn state(&self) -> MutexGuard<'_, State> { self.state.lock().expect("recording segments poisoned") }
    pub(crate) fn begin(&mut self, stream: bool) {
        let mut state = self.state();
        state.encoders.clear();
        state.stream = stream;
        state.segmented = false;
        state.finish_ms = 0.;
    }
    /// An empty recording owned by these segments.
    pub(crate) fn encoder(&self, r: &crate::Renderer, label: Option<&str>) -> wgpu::CommandEncoder {
        let mut encoder = r.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label });
        if let Some(passes) = segment_passes(r) {
            let (state, device, queue) = (Arc::clone(&self.state), r.device.clone(), r.queue.clone());
            encoder.bound_passes(passes, Arc::new(move |encoder: &mut wgpu::CommandEncoder| {
                // The replacement inherits the bound from the recording encoder.
                let finished = std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
                state.lock().expect("recording segments poisoned").complete(&queue, finished);
            }));
        }
        encoder
    }
    fn complete(&mut self, r: &crate::Renderer, encoder: wgpu::CommandEncoder) {
        self.state().complete(&r.queue, encoder);
    }
    /// Retire a complete recording chunk through the existing submission owner.
    /// Unlike a Vulkan native boundary, this also releases encoder host state.
    pub(crate) fn cut(&mut self, r: &crate::Renderer, encoder: &mut wgpu::CommandEncoder) {
        let finished = std::mem::replace(encoder, self.encoder(r, None));
        self.complete(r, finished);
    }
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool { self.state().encoders.is_empty() }
    #[cfg(test)]
    pub(crate) fn segmented(&self) -> bool { self.state().segmented }
    pub(crate) fn push(&mut self, encoder: wgpu::CommandEncoder) { self.state().encoders.push(encoder); }
    /// Close a sample with the caller's existing measured/validation boundaries.
    pub(crate) fn finish_sample(
        &mut self, r: &crate::Renderer, mut encoder: wgpu::CommandEncoder,
        batch: Option<&mut Batch<'_>>, submission_end: bool, retain: bool,
    ) -> (Option<wgpu::CommandEncoder>, Option<wgpu::SubmissionIndex>, f64) {
        if batch.is_some() && submission_end { encoder.submission_boundary(); }
        let mut state = self.state();
        state.encoders.push(encoder);
        if let Some(batch) = batch {
            if !state.segmented && retain { return (state.encoders.pop(), None, 0.); }
            for encoder in state.encoders.drain(..) { batch.push(encoder); }
            (None, None, 0.)
        } else {
            let (submission, finish_ms) = Self::submit_encoders(r, state.encoders.drain(..));
            (None, submission, finish_ms + state.finish_ms)
        }
    }

    /// Synchronous interop consumes every preceding cut and the current
    /// recording. The replacement encoder establishes the frame's final fence.
    pub(crate) fn interop(&mut self, r: &crate::Renderer, encoder: &mut wgpu::CommandEncoder, label: &str)
        -> (Option<wgpu::SubmissionIndex>, f64)
    {
        let pending = std::mem::replace(encoder, self.encoder(r, Some(label)));
        let mut state = self.state();
        Self::submit_encoders(r, state.encoders.drain(..).chain(std::iter::once(pending)))
    }

    pub(crate) fn publish(
        r: &crate::Renderer, mut batch: Batch<'_>, upload: &wgpu::Buffer,
        bytes: &[u8], stream_ready: bool,
    ) -> (wgpu::SubmissionIndex, f64) {
        if !bytes.is_empty() { r.queue.write_buffer(upload, 0, bytes); }
        let start = std::time::Instant::now();
        batch.uploads_published();
        let submission = batch.submit(&r.queue, stream_ready);
        (submission, start.elapsed().as_secs_f64() * 1000.)
    }

    /// Finish and submit synchronous cuts in recording order.
    pub(crate) fn submit_encoders(r: &crate::Renderer, encoders: impl IntoIterator<Item = wgpu::CommandEncoder>) -> (Option<wgpu::SubmissionIndex>, f64) {
        let mut last = None;
        let mut finish_ms = 0.;
        for encoder in encoders {
            let (submission, elapsed) = Self::finish_encoder(&r.queue, encoder);
            last = Some(submission);
            finish_ms += elapsed;
        }
        (last, finish_ms)
    }

    /// Standalone preparation/install work keeps its existing single submission.
    pub(crate) fn submit(queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        Self::finish_encoder(queue, encoder).0
    }

    /// Publish queue writes or surface loss while polling, without recording work.
    pub(crate) fn flush(queue: &wgpu::Queue) -> wgpu::SubmissionIndex {
        let mut ownership = Ownership::default();
        ownership.publish();
        Self::submit_commands(queue, &mut ownership, std::iter::empty())
    }

    fn finish_encoder(queue: &wgpu::Queue, encoder: wgpu::CommandEncoder) -> (wgpu::SubmissionIndex, f64) {
        let mut ownership = Ownership::default();
        ownership.record();
        // Streaming/interop uploads are queue writes made during preparation.
        ownership.publish();
        let start = std::time::Instant::now();
        let command = encoder.finish();
        let finish_ms = start.elapsed().as_secs_f64() * 1000.;
        ownership.finish();
        (Self::submit_commands(queue, &mut ownership, std::iter::once(command)), finish_ms)
    }

    fn submit_commands(
        queue: &wgpu::Queue, ownership: &mut Ownership,
        commands: impl ExactSizeIterator<Item = wgpu::CommandBuffer>,
    ) -> wgpu::SubmissionIndex {
        ownership.submit(commands.len());
        #[cfg(test)]
        SUBMISSIONS.with(|count| count.set(count.get() + 1));
        queue.submit(commands)
    }
}

/// End an ordered GPU group without waiting for GPU completion. Metal owns
/// native pass/barrier buffers already during recording, so streaming groups
/// must be submitted here, rather than retained until the entire sample ends.
pub(crate) fn submission_boundary(r: &crate::Renderer, encoder: &mut wgpu::CommandEncoder, segments: &mut Segments) {
    segments.state().segmented = true;
    if r.adapter_info.backend == wgpu::Backend::Vulkan { encoder.submission_boundary(); }
    else {
        segments.cut(r, encoder);
    }
}

/// Passes per recording segment, where the backend bounds unsubmitted native
/// command storage. Vulkan also bounds recordings before native encoding: a
/// single media recording can contain millions of indirect work blocks and fill
/// driver memory before anything is submitted. The existing ordered owner and
/// deferred validation window publish and retire those portions.
/// wgpu-hal Metal admits `METAL_UNSUBMITTED_BUFFERS`; wgpu-core
/// makes at most `NATIVE_BUFFERS_PER_PASS` per pass (barriers, the pass, then
/// following transfers). A deferred batch may hold `BATCH_WINDOW` finished
/// segments, and the recording itself one more.
fn segment_passes(r: &crate::Renderer) -> Option<u32> {
    match r.adapter_info.backend {
        wgpu::Backend::Metal => Some(METAL_SEGMENT_PASSES),
        wgpu::Backend::Vulkan => Some(crate::constants::VULKAN_RECORDING_PASSES),
        _ => None,
    }
}
const METAL_UNSUBMITTED_BUFFERS: u32 = 4096;
const NATIVE_BUFFERS_PER_PASS: u32 = 3;
const METAL_SEGMENT_PASSES: u32 = METAL_UNSUBMITTED_BUFFERS / NATIVE_BUFFERS_PER_PASS / (METAL_BATCH_WINDOW as u32 + 1) - 1;
/// Finished, unsubmitted segments a Metal batch holds while validating.
const METAL_BATCH_WINDOW: usize = crate::constants::COMMAND_FINISH_THREADS;
const _: () = assert!(METAL_SEGMENT_PASSES >= 64);

/// Portable validation ownership ledger. Finished commands remain owned by
/// wgpu after submission until the frame slot establishes its final fence.
#[derive(Default, Debug)]
struct Ownership {
    recorded: usize,
    finished: usize,
    submitted: usize,
    uploads: bool,
}
impl Ownership {
    fn record(&mut self) {
        assert!(!self.uploads, "recording after packed upload publication");
        self.recorded += 1;
    }
    fn publish(&mut self) {
        assert!(!self.uploads, "uploads published twice");
        self.uploads = true;
    }
    fn finish(&mut self) {
        assert!(self.finished < self.recorded, "finishing an unowned command");
        self.finished += 1;
    }
    fn submit(&mut self, count: usize) {
        assert!(self.uploads, "submission before packed uploads");
        assert!(self.submitted + count <= self.finished, "submission before finish");
        self.submitted += count;
    }
}

/// Final-fence ownership of a frame slot. Failed waits leave it pending;
/// rejected output may retire only after successful completion and mapping.
#[derive(Default, Debug)]
pub(crate) struct FlightOwnership { pending: bool, complete: bool }
impl FlightOwnership {
    pub(crate) fn submit(&mut self) {
        assert!(!self.pending, "reusing a slot before retirement");
        self.pending = true;
        self.complete = false;
    }
    pub(crate) fn complete(&mut self) {
        assert!(self.pending && !self.complete, "completion without pending flight");
        self.complete = true;
    }
    pub(crate) fn retire(&mut self) {
        assert!(self.pending && self.complete, "retirement before final fence");
        self.pending = false;
    }
}

struct Workers {
    senders: Vec<mpsc::Sender<wgpu::CommandEncoder>>,
    completed: Vec<mpsc::Receiver<wgpu::CommandBuffer>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.senders.clear();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

pub(crate) struct CommandFinish {
    workers: Mutex<Workers>,
}

impl CommandFinish {
    pub(crate) fn new() -> Self {
        let count = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(crate::constants::COMMAND_FINISH_THREADS);
        let mut completed = Vec::with_capacity(count);
        let mut senders = Vec::with_capacity(count);
        let mut threads = Vec::with_capacity(count);
        for index in 0..count {
            // Recording cannot block on completion: all packed uploads publish
            // after recording, before these exposure-sized queues are drained.
            let (tx, rx) = mpsc::channel::<wgpu::CommandEncoder>();
            let (completed_tx, completed_rx) = mpsc::channel();
            completed.push(completed_rx);
            threads.push(
                std::thread::Builder::new()
                    .name(format!("vork-command-{index}"))
                    .spawn(move || {
                        while let Ok(encoder) = rx.recv() {
                            if completed_tx.send(encoder.finish()).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn command validation worker"),
            );
            senders.push(tx);
        }
        Self {
            workers: Mutex::new(Workers {
                senders,
                completed,
                threads,
            }),
        }
    }

    /// `window` bounds finished, unsubmitted segments ([`batch_window`]).
    pub(crate) fn batch(&self, window: Option<usize>) -> Batch<'_> {
        Batch {
            workers: self.workers.lock().expect("command validation poisoned"),
            sent: 0,
            received: 0,
            submitted: 0,
            window,
            queued: std::collections::VecDeque::new(),
            ownership: Ownership::default(),
        }
    }
}

/// Finished segments a deferred batch may hold before submission, where the
/// backend bounds unsubmitted native command buffers.
pub(crate) fn batch_window(r: &crate::Renderer) -> Option<usize> {
    segment_passes(r).map(|_| crate::constants::COMMAND_FINISH_THREADS)
}

/// Own the worker lease through recording and drain submitted work on every exit.
pub(crate) struct Batch<'a> {
    workers: std::sync::MutexGuard<'a, Workers>,
    sent: usize,
    received: usize,
    submitted: usize,
    /// Segments beyond the window wait, unfinished, until earlier ones submit.
    window: Option<usize>,
    queued: std::collections::VecDeque<wgpu::CommandEncoder>,
    ownership: Ownership,
}

impl Batch<'_> {
    /// Called only after the exposure's packed queue write (including an empty upload).
    fn uploads_published(&mut self) { self.ownership.publish(); }

    /// Submit ready validation prefixes in order.
    /// Validation may overlap later recording; submission never waits for the GPU.
    fn submit(mut self, queue: &wgpu::Queue, stream_ready: bool) -> wgpu::SubmissionIndex {
        let mut last = None;
        let mut commands = Vec::new();
        loop {
            while let Some(command) = self.next() {
                commands.push(command.expect("command validation worker stopped"));
                // Cold/ GPU-bound flights validate before submission. Interleaving
                // small queue submissions with driver replay can stall both paths.
                if !stream_ready { continue; }
                // Native recording markers stay inside the submitted batch.
                // Submit ready CPU prefixes while later validation finishes.
                while self.received < self.sent {
                    let index = self.received % self.workers.completed.len();
                    match self.workers.completed[index].try_recv() {
                        Ok(command) => {
                            self.received += 1;
                            self.ownership.finish();
                            commands.push(command);
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => panic!("command validation worker stopped"),
                    }
                }
                last = Some(self.submit_commands(queue, &mut commands));
            }
            // A bounded window submits what it holds before validating more.
            if !commands.is_empty() { last = Some(self.submit_commands(queue, &mut commands)); }
            // Submission refilled the window; finish once nothing remains.
            if self.received == self.sent { break; }
        }
        last.expect("nonempty sample batch")
    }
    fn submit_commands(&mut self, queue: &wgpu::Queue, commands: &mut Vec<wgpu::CommandBuffer>) -> wgpu::SubmissionIndex {
        self.submitted += commands.len();
        let submission = Segments::submit_commands(queue, &mut self.ownership, commands.drain(..));
        self.refill();
        submission
    }
    fn push(&mut self, encoder: wgpu::CommandEncoder) {
        self.queued.push_back(encoder);
        self.ownership.record();
        self.refill();
    }
    /// Validate queued segments while the window has room.
    fn refill(&mut self) {
        while self.window.is_none_or(|window| self.sent - self.submitted < window) {
            let Some(encoder) = self.queued.pop_front() else { break };
            self.workers.senders[self.sent % self.workers.senders.len()]
                .send(encoder)
                .expect("command validation worker stopped");
            self.sent += 1;
        }
    }
    fn next(&mut self) -> Option<Result<wgpu::CommandBuffer, mpsc::RecvError>> {
        if self.received == self.sent { return None; }
        let index = self.received % self.workers.completed.len();
        self.received += 1;
        let result = self.workers.completed[index].recv();
        self.ownership.finish();
        Some(result)
    }
}

impl Drop for Batch<'_> {
    fn drop(&mut self) {
        // A preparation error can leave earlier groups validating. Collect their
        // results before another exposure leases the same completion channels.
        while self.next().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    #[test]
    fn standalone_consumers_use_submission_owner() {
        for source in [include_str!("rt.rs"), include_str!("materials.rs"), include_str!("vdb.rs"), include_str!("gpu_bc7.rs"), include_str!("display_stats.rs")] {
            assert!(!source.contains("queue.submit(") && !source.contains(".device.poll("), "use Segments and Renderer::wait");
        }
    }

    #[test]
    #[ignore = "requires real Metal GPU; scripts/check-metal-contracts"]
    fn metal_contract_completion_and_error() -> anyhow::Result<()> {
        let r = crate::metal_contract_renderer()?;
        let output = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("completion contract"), size: 4, mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        r.queue.write_buffer(&output, 0, &37u32.to_le_bytes());
        let target = r.queue.submit([]);
        let (tx, rx) = mpsc::channel();
        output.map_async(wgpu::MapMode::Read, .., move |result| { tx.send(result).unwrap(); });
        r.wait(Some(target))?;
        // Mapping completion may require polling after the fixed submission fence.
        r.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        assert_eq!(&*output.get_mapped_range(..), &37u32.to_le_bytes());
        output.unmap();
        r.wait(Some(r.queue.submit([])))?;
        // Deliberate device destruction is isolated to this test's device. No
        // watchdog hang or shared-desktop GPU reset is needed to test delivery.
        r.device.destroy();
        let _ = r.device.poll(wgpu::PollType::Poll);
        let error = r.wait(None).expect_err("lost device must reject completion");
        assert_eq!(crate::failure::exit_code(&error), 4);
        assert!(format!("{error:#}").contains("GPU device lost"));
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn sample_owner_preserves_packed_uploads_interop_and_final_fence() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let finish = CommandFinish::new();
        let expected: Vec<u32> = (0..32).map(|sample| sample * 1664525 + 1013904223).collect();
        let source = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 128, mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let output = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 128, mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        // The same raw sample identities cross streaming, deferred validation,
        // and an interop drain. Packed bytes publish only after recording.
        for deferred in [false, true] {
            let mut segments = Segments::default();
            segments.begin(!deferred);
            let mut batch = deferred.then(|| finish.batch(None));
            if deferred { r.queue.write_buffer(&source, 0, &[0; 128]); }
            else { r.queue.write_buffer(&source, 0, bytemuck::cast_slice(&expected)); }
            let mut recording = None;
            let mut final_submission = None;
            for sample in 0..32 {
                let mut encoder = recording.take().unwrap_or_else(|| r.device.create_command_encoder(&Default::default()));
                encoder.copy_buffer_to_buffer(&source, sample * 4, &output, sample * 4, 4);
                let (next, submission, _) = segments.finish_sample(&r, encoder, batch.as_mut(), sample % 8 == 7, sample % 4 != 3);
                recording = next;
                if submission.is_some() { final_submission = submission; }
            }
            assert!(recording.is_none());
            if let Some(batch) = batch {
                final_submission = Some(Segments::publish(&r, batch, &source, bytemuck::cast_slice(&expected), true).0);
            }
            r.wait(final_submission)?;
            let (tx, rx) = mpsc::channel();
            output.map_async(wgpu::MapMode::Read, .., move |result| { tx.send(result).unwrap(); });
            r.wait(None)?;
            rx.recv()??;
            assert_eq!(&*output.get_mapped_range(..), bytemuck::cast_slice::<u32, u8>(&expected));
            output.unmap();
        }
        // Interop drains retained work, then the post-interop encoder owns the
        // final fence even when it contains no additional colour work.
        let mut segments = Segments::default();
        segments.begin(false);
        let mut retained = r.device.create_command_encoder(&Default::default());
        retained.clear_buffer(&output, 0, None);
        retained.copy_buffer_to_buffer(&source, 0, &output, 0, 128);
        segments.push(retained);
        let (interop, _) = segments.interop(&r, &mut r.device.create_command_encoder(&Default::default()), "after interop");
        r.wait(interop)?;
        segments.begin(true);
        let (_, final_fence, _) = segments.finish_sample(&r, r.device.create_command_encoder(&Default::default()), None, true, false);
        assert!(final_fence.is_some());
        r.wait(final_fence)?;
        let (tx, rx) = mpsc::channel();
        output.map_async(wgpu::MapMode::Read, .., move |result| { tx.send(result).unwrap(); });
        r.wait(None)?;
        rx.recv()??;
        assert_eq!(&*output.get_mapped_range(..), bytemuck::cast_slice::<u32, u8>(&expected));
        output.unmap();
        let standalone = Segments::submit(&r.queue, r.device.create_command_encoder(&Default::default()));
        r.wait(Some(standalone))?;
        r.wait(Some(Segments::flush(&r.queue)))?;
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn recordings_with_more_passes_than_native_admission_complete_in_order() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let finish = CommandFinish::new();
        let source = r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&(0..64u32).collect::<Vec<_>>()),
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let output = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 4, mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        // Every pass costs native command buffers on Metal; this recording
        // needs several times the queue's unsubmitted admission.
        let passes = METAL_UNSUBMITTED_BUFFERS as usize * 2;
        for (stream, batched, stream_ready) in [(true, false, false), (false, false, false), (false, true, false), (false, true, true)] {
            let mut segments = Segments::default();
            segments.begin(stream);
            let mut batch = batched.then(|| finish.batch(batch_window(&r)));
            let mut encoder = segments.encoder(&r, None);
            for pass in 0..passes {
                drop(encoder.begin_compute_pass(&Default::default()));
                // Later copies overwrite earlier ones only in recording order.
                if pass % 61 == 0 { encoder.copy_buffer_to_buffer(&source, (pass / 61 % 64) as u64 * 4, &output, 0, 4); }
                if let Some(passes) = segment_passes(&r) { assert!(encoder.segment_passes().unwrap() <= passes); }
            }
            let expected = ((passes - 1) / 61 % 64) as u32;
            let (recording, submission, _) = segments.finish_sample(&r, encoder, batch.as_mut(), true, false);
            assert!(recording.is_none());
            let submission = match batch {
                Some(batch) => Some(Segments::publish(&r, batch, &source, &[], stream_ready).0),
                None => submission,
            };
            r.wait(submission)?;
            let (tx, rx) = mpsc::channel();
            output.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
            r.wait(None)?;
            rx.recv()??;
            assert_eq!(&*output.get_mapped_range(..), &expected.to_le_bytes(), "stream {stream} batch {batched}");
            output.unmap();
        }
        r.errors.check()?;
        Ok(())
    }

    #[test]
    fn final_fence_owns_resources_even_after_error() {
        let mut flight = FlightOwnership::default();
        for _ in 0..3 {
            flight.submit();
            // A failed poll/partial submission cannot establish completion.
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| flight.retire())).is_err());
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| flight.submit())).is_err());
            flight.complete();
            flight.retire();
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Event { Record, Upload, Finish, Submit(usize), Fence, Retire }

    fn replay(events: &[Event]) -> Ownership {
        let mut owner = Ownership::default();
        let mut flight = FlightOwnership::default();
        for event in events {
            match *event {
                Event::Record => owner.record(),
                Event::Upload => owner.publish(),
                Event::Finish => owner.finish(),
                Event::Submit(n) => {
                    owner.submit(n);
                    if !flight.pending { flight.submit(); }
                }
                Event::Fence => flight.complete(),
                Event::Retire => flight.retire(),
            }
        }
        owner
    }

    #[test]
    fn ownership_event_log() {
        use Event::*;
        // Validation may finish before publication. Ready prefixes submit in
        // order; resources retire only after the final completion fence.
        let owner = replay(&[Record, Record, Finish, Upload, Submit(1), Finish, Submit(1), Fence, Retire]);
        assert_eq!((owner.recorded, owner.finished, owner.submitted), (2, 2, 2));
        // Preparation/finish failure drains results without publishing commands.
        let abandoned = replay(&[Record, Record, Finish, Finish]);
        assert_eq!((abandoned.finished, abandoned.submitted), (2, 0));
        for invalid in [vec![Record, Finish, Submit(1)],
            vec![Record, Upload, Submit(1)], vec![Finish],
            vec![Upload, Upload], vec![Upload, Record],
            vec![Record, Finish, Upload, Submit(1), Submit(1)],
            vec![Record, Finish, Upload, Submit(1), Retire],
            vec![Fence], vec![Record, Finish, Upload, Submit(1), Fence, Retire, Retire]] {
            assert!(std::panic::catch_unwind(|| replay(&invalid)).is_err(), "accepted {invalid:?}");
        }
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn streaming_groups_publish_before_the_sample_finishes() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let source = r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::cast_slice(&[11u32, 29]),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let output = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 8, mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        let mut segments = Segments::default();
        segments.begin(true);
        // The second group has not even been recorded when the first is
        // published. No GPU wait, poll, or collection occurs between groups.
        for index in 0..2 {
            let mut encoder = r.device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&source, index * 4, &output, index * 4, 4);
            segments.complete(&r, encoder);
            assert!(segments.is_empty(), "completed streaming groups must not accumulate native buffers");
        }
        assert!(segments.segmented());
        let (tx, rx) = mpsc::channel();
        output.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
        r.wait(Some(r.queue.submit([])))?;
        rx.recv()??;
        assert_eq!(&*output.get_mapped_range(..), bytemuck::cast_slice::<u32, u8>(&[11u32, 29]));
        output.unmap();
        // Deferred uploads must retain the opposite policy until publication.
        segments.begin(false);
        segments.complete(&r, r.device.create_command_encoder(&Default::default()));
        assert_eq!(segments.state().encoders.len(), 1);
        r.wait(Segments::submit_encoders(&r, segments.state().encoders.drain(..)).0)?;
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn read_only_depth_across_native_cuts_preserves_contents() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let depth = r.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("read-only depth across native cuts"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = depth.create_view(&Default::default());
        let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = r.device.create_command_encoder(&Default::default());
        for ops in [Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(0.375), store: wgpu::StoreOp::Store,
        }), None, None] {
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clear then read-only depth"),
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &view, depth_ops: ops, stencil_ops: None,
                    }),
                    ..Default::default()
                });
            }
            encoder.submission_boundary();
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &depth, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::DepthOnly },
            wgpu::TexelCopyBufferInfo { buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let submission = r.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
        r.wait(Some(submission))?;
        rx.recv()??;
        assert_eq!(f32::from_le_bytes(readback.get_mapped_range(..)[..4].try_into()?), 0.375);
        readback.unmap();
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn many_native_recording_cuts_preserve_barriers_and_fence() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let values: Vec<u32> = (0..1024).collect();
        let source = r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("native cut values"),
            contents: bytemuck::cast_slice(&values),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let scratch = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reused across native cuts"), size: 4,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("all native cut results"), size: values.len() as u64 * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // One core encoder, thousands of native cuts, and RAW/WAR dependencies
        // through one scratch buffer. Check every result, not just the last copy.
        for small_submissions in [false,true] {for repetition in 0..3 {
            let started=std::time::Instant::now();
            let mut encoder = r.device.create_command_encoder(&Default::default());
            for index in 0..values.len() as u64 {
                encoder.copy_buffer_to_buffer(&source, index * 4, &scratch, 0, 4);
                encoder.submission_boundary();
                encoder.copy_buffer_to_buffer(&scratch, 0, &readback, index * 4, 4);
                encoder.submission_boundary();
                // Every call is below the native-buffer budget; collectively
                // these must obey the same pending bound as one large call.
                if small_submissions {
                    r.queue.submit([encoder.finish()]);
                    encoder=r.device.create_command_encoder(&Default::default());
                }
            }
            let submission = r.queue.submit([encoder.finish()]);
            let (tx, rx) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
            r.wait(Some(submission))?;
            rx.recv()??;
            assert_eq!(&*readback.get_mapped_range(..), bytemuck::cast_slice::<u32, u8>(&values));
            readback.unmap();
            eprintln!("native backlog small_submissions={small_submissions} repetition={repetition} completion_ms={:.3}",started.elapsed().as_secs_f64()*1000.);
        }}
        r.wait(Some(r.queue.submit([])))?;
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn abandoned_batches_and_native_boundaries_preserve_order() -> anyhow::Result<()> {
        let renderer = crate::test_renderer()?;
        let finish = CommandFinish::new();
        let source = renderer
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ordered worker values"),
                contents: bytemuck::cast_slice(&(0..(crate::constants::COMMAND_FINISH_THREADS * 3) as u32).collect::<Vec<_>>()),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        let output = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let stage = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ordered native submission stage"), size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (count, stream) in [(crate::constants::COMMAND_FINISH_THREADS * 3, false),
            (crate::constants::COMMAND_FINISH_THREADS * 3, true), (1, false), (1, true)] {
            // Model a preparation error after some groups have been handed off.
            let mut abandoned = finish.batch(None);
            abandoned.push(renderer.device.create_command_encoder(&Default::default()));
            drop(abandoned);
            let mut batch = finish.batch(None);
            for index in 0..count {
                let mut encoder = renderer.device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(&source, index as u64 * 4, &stage, 0, 4);
                encoder.submission_boundary();
                encoder.copy_buffer_to_buffer(&stage, 0, &output, 0, 4);
                encoder.submission_boundary();
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, 4);
                batch.push(encoder);
            }
            batch.uploads_published();
            let submission = batch.submit(&renderer.queue, stream);
            let (tx, rx) = mpsc::channel();
            readback.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
            renderer.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
            rx.recv()??;
            let mapped = readback.get_mapped_range(..);
            assert_eq!(
                u32::from_le_bytes(mapped[..4].try_into().unwrap()),
                count as u32 - 1
            );
            drop(mapped);
            readback.unmap();
        }
        // Even an empty native submission must signal its lifetime fence.
        renderer.wait(Some(renderer.queue.submit([])))?;
        Ok(())
    }
}

// Per-test-thread counts exclude unrelated renderers running in parallel.
#[cfg(test)]
thread_local! {
    static SUBMISSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn submission_count() -> usize { SUBMISSIONS.with(|count| count.get()) }
