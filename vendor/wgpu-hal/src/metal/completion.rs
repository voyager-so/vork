//! Queue completion is a prefix of *all* native buffers, not just fence signals.
//!
//! Every tracked buffer's completed handler reports its serial ([`Finished`]): the queue knows the serial up to
//! which every buffer has finished without asking any buffer for its status. A wait for a fence value, and a
//! submission that finds the queue over its budget of pending buffers ([`crate::MAX_PENDING_NATIVE_BUFFERS`]),
//! sleep until that serial reaches the one they need, and are woken when it reaches the lowest serial any waiter
//! needs. Each commit lets go of the buffers reported finished. Neither sleeps on the clock, and neither walks the
//! pending list: with a status poll of every pending buffer on each commit and a wake-up on
//! each completion, a clip of 46 command buffers a frame spent 0.12 ms a frame more recording than without them.
use alloc::{
    collections::{BTreeSet, VecDeque},
    string::String,
    sync::Arc,
};
use core::time::Duration;
use parking_lot::Condvar;

use super::*;
use objc2::Message;

#[derive(Debug)]
struct Pending<T> {
    issued: u64,
    completed: u64,
    buffers: VecDeque<(u64, T)>,
}

impl<T> Default for Pending<T> {
    fn default() -> Self {
        Self {
            issued: 0,
            completed: 0,
            buffers: VecDeque::new(),
        }
    }
}

impl<T> Pending<T> {
    fn push(&mut self, buffer: T) -> u64 {
        self.issued += 1;
        self.buffers.push_back((self.issued, buffer));
        self.issued
    }

    /// Let go of every buffer up to `serial`, which their completed handlers have reported finished, and count
    /// them completed. One of them may have failed: its failure was latched before it was reported, and
    /// `QueueCompletion::poll` reads the latch after the completed serial, so it is never published as a success. A commit does this, so the buffers this queue retains are the unfinished
    /// ones and those that finished since the last commit, whatever the caller's cadence of submissions and waits.
    fn retire_through(&mut self, serial: u64) {
        while self.buffers.front().is_some_and(|(issued, _)| *issued <= serial) {
            self.buffers.pop_front();
        }
        self.completed = self.completed.max(serial.min(self.issued));
    }

    fn poll(
        &mut self,
        mut status: impl FnMut(&T) -> Result<bool, crate::DeviceError>,
    ) -> Result<u64, crate::DeviceError> {
        let mut prefix = true;
        let mut completed = self.completed;
        // Inspect even buffers beyond an unfinished prefix: a later abort must
        // be reported promptly. Retire nothing until every status was checked.
        for (serial, buffer) in &self.buffers {
            prefix &= status(buffer)?;
            if prefix {
                completed = *serial;
            }
        }
        self.completed = completed;
        while self
            .buffers
            .front()
            .is_some_and(|(serial, _)| *serial <= completed)
        {
            self.buffers.pop_front();
        }
        Ok(completed)
    }
}

/// Which tracked buffers have finished, as their completed handlers report it. Serials are the queue's, issued
/// from 1 in commit order; buffers usually finish in that order.
#[derive(Debug, Default)]
struct Finished {
    /// Every buffer up to this serial has finished.
    prefix: u64,
    /// Serials past the prefix that finished before an earlier buffer did.
    ahead: BTreeSet<u64>,
    /// The lowest serial a waiter needs the prefix to reach. Handlers wake the waiters when it does, and no sooner.
    wanted: Option<u64>,
    /// A buffer failed: every waiter is woken, and the queue's poll reports why.
    failed: bool,
}

impl Finished {
    /// The buffer `serial` finished (`failed`: in error). Returns whether to wake the waiters.
    fn finish(&mut self, serial: u64, failed: bool) -> bool {
        if serial == self.prefix + 1 {
            self.prefix = serial;
            while self.ahead.remove(&(self.prefix + 1)) {
                self.prefix += 1;
            }
        } else if serial > self.prefix {
            self.ahead.insert(serial);
        }
        self.failed |= failed;
        let wake = self.failed || self.wanted.is_some_and(|wanted| self.prefix >= wanted);
        if wake {
            // Woken waiters that need more ask again.
            self.wanted = None;
        }
        wake
    }

    /// Whether a waiter for `serial` is done waiting; if not, it is recorded as wanting it.
    fn reached(&mut self, serial: u64) -> bool {
        if self.failed || self.prefix >= serial {
            return true;
        }
        self.wanted = Some(self.wanted.map_or(serial, |wanted| wanted.min(serial)));
        false
    }
}

#[derive(Debug, Default)]
pub(super) struct QueueCompletion {
    pending: Mutex<Pending<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    // Callbacks own only this latch and the finished state below, never the pending buffers (no retain cycle).
    failure: Arc<Mutex<Option<String>>>,
    finished: Arc<(Mutex<Finished>, Condvar)>,
}

// Metal command buffers permit concurrent status queries; access to the retained
// list is serialized by the mutex, as it was for HAL fences.
unsafe impl Send for QueueCompletion {}
unsafe impl Sync for QueueCompletion {}

fn failure(buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> Option<String> {
    (buffer.status() == MTLCommandBufferStatus::Error).then(|| {
        format!(
            "Metal command buffer {:?} failed: {:?}",
            buffer.label(),
            buffer.error()
        )
    })
}

impl QueueCompletion {
    /// Commit `buffer` on the queue, the one way a native command buffer reaches it: tracked until it finishes,
    /// committed, and the queue brought back within its budget of pending buffers
    /// ([`crate::MAX_PENDING_NATIVE_BUFFERS`]) before the caller commits another. Returns the buffer's serial and
    /// what keeping the budget found (a failed earlier buffer).
    pub(super) fn commit(
        &self,
        buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> (u64, Result<(), crate::DeviceError>) {
        let serial = self.track(buffer);
        buffer.commit();
        (serial, self.make_room(serial, crate::MAX_PENDING_NATIVE_BUFFERS as u64))
    }

    fn track(&self, buffer: &ProtocolObject<dyn MTLCommandBuffer>) -> u64 {
        // What has finished leaves the retained list as the next buffer joins it. A failed buffer is latched before
        // it is reported finished, and every poll reports the latch before it publishes anything.
        let finished_through = self.finished.0.lock().prefix;
        // The serial first: the handler reports it.
        let serial = {
            let mut pending = self.pending.lock();
            pending.retire_through(finished_through);
            pending.push(buffer.retain())
        };
        let latch = Arc::clone(&self.failure);
        let finished = Arc::clone(&self.finished);
        let block = block2::RcBlock::new(
            move |buffer: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                // Latch a failure before waiters can be woken for it: the poll they make reports the latch.
                let failed = failure(unsafe { buffer.as_ref() }).map(|reason| {
                    latch.lock().get_or_insert(reason);
                });
                let (state, wake) = &*finished;
                let wake_waiters = state.lock().finish(serial, failed.is_some());
                if wake_waiters {
                    wake.notify_all();
                }
            },
        );
        unsafe {
            buffer.addCompletedHandler(block2::RcBlock::as_ptr(&block));
        }
        serial
    }

    /// Block until every tracked buffer up to `serial` has finished, a buffer has failed, or `timeout` has passed.
    /// A buffer that has not finished always runs its handler later, so there is always a wake-up to wait for.
    pub(super) fn wait_for(&self, serial: u64, timeout: Option<Duration>) {
        let deadline = timeout.map(|timeout| std::time::Instant::now() + timeout);
        let (state, wake) = &*self.finished;
        let mut state = state.lock();
        while !state.reached(serial) {
            match deadline {
                Some(deadline) => {
                    if wake.wait_until(&mut state, deadline).timed_out() {
                        return;
                    }
                }
                None => wake.wait(&mut state),
            }
        }
    }

    /// Wait until at most `limit` buffers up to `serial`, the newest, are unfinished. They are already committed,
    /// so they finish without anything further from this thread. No status is polled: the handlers' reports say
    /// what has finished, and the next commit lets it go (`Pending::retire_through`).
    fn make_room(&self, serial: u64, limit: u64) -> Result<(), crate::DeviceError> {
        if serial > limit {
            self.wait_for(serial - limit, None);
        }
        match self.failure.lock().as_ref() {
            Some(reason) => Err(crate::DeviceError::LostWithReason(reason.clone())),
            None => Ok(()),
        }
    }

    pub(super) fn poll(&self) -> Result<u64, crate::DeviceError> {
        if let Some(reason) = self.failure.lock().as_ref() {
            return Err(crate::DeviceError::LostWithReason(reason.clone()));
        }
        let result = self.pending.lock().poll(|buffer| {
            // Metal can expose terminal status before invoking the handler.
            // Check it here too, before retiring buffers or publishing a fence.
            if let Some(reason) = failure(buffer) {
                return Err(crate::DeviceError::LostWithReason(reason));
            }
            Ok(buffer.status() == MTLCommandBufferStatus::Completed)
        });
        if let Err(crate::DeviceError::LostWithReason(reason)) = &result {
            self.failure.lock().get_or_insert_with(|| reason.clone());
        }
        // A commit on another thread may have retired a failed buffer, and counted it completed, between the
        // check above and the scan (`Pending::retire_through`: a failure is latched before its buffer is reported
        // finished, so the latch is set before any such retirement). The latch is read again after the completed
        // serial: no value at or past a failed buffer is returned as a success.
        if let Some(reason) = self.failure.lock().as_ref() {
            return Err(crate::DeviceError::LostWithReason(reason.clone()));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_success_cannot_hide_earlier_error() {
        let mut pending = Pending::default();
        pending.push(0);
        pending.push(1);
        pending.push(2);
        let result = pending.poll(|&id| {
            if id == 1 {
                Err(crate::DeviceError::LostWithReason("native abort".into()))
            } else {
                Ok(true)
            }
        });
        assert!(
            matches!(result, Err(crate::DeviceError::LostWithReason(reason)) if reason == "native abort")
        );
        assert_eq!(pending.completed, 0);
        assert_eq!(pending.buffers.len(), 3);
    }

    #[test]
    fn completion_requires_the_entire_prefix() {
        let mut pending = Pending::default();
        pending.push(0);
        pending.push(1);
        pending.push(2);
        assert_eq!(pending.poll(|&id| Ok(id != 1)).unwrap(), 1);
        assert_eq!(pending.buffers.len(), 2);
        assert_eq!(pending.poll(|_| Ok(true)).unwrap(), 3);
        assert!(pending.buffers.is_empty());
        assert_eq!(pending.poll(|_| unreachable!()).unwrap(), 3);
    }

    #[test]
    fn commits_retire_what_finished_without_a_poll() {
        // Presents with no submission or wait between them: each commit retires what the handlers reported, so
        // the retained list stays at the unfinished buffers and never grows with the number of commits.
        let mut pending = Pending::default();
        let mut finished = Finished::default();
        for commit in 1..=1000u64 {
            pending.retire_through(finished.prefix);
            assert_eq!(pending.push(commit), commit);
            assert!(pending.buffers.len() <= 2, "retained {} buffers", pending.buffers.len());
            // Each buffer finishes one commit late.
            if commit > 1 {
                finished.finish(commit - 1, false);
            }
        }
        assert_eq!(pending.completed, 998);
        // A poll after that agrees with it and takes the rest.
        assert_eq!(pending.poll(|_| Ok(true)).unwrap(), 1000);
        assert!(pending.buffers.is_empty());
        // Retiring past what was issued, or behind what is completed, changes nothing.
        pending.retire_through(5000);
        pending.retire_through(3);
        assert_eq!(pending.completed, 1000);
    }

    #[test]
    fn the_finished_prefix_waits_for_every_earlier_buffer() {
        // Buffers behind an unfinished one (empty submissions' signals, say) do not advance the prefix, so they
        // count against the budget until it finishes; then the prefix takes them all at once.
        let mut finished = Finished::default();
        assert!(!finished.finish(2, false));
        assert!(!finished.finish(4, false));
        assert!(!finished.finish(3, false));
        assert_eq!(finished.prefix, 0);
        assert!(!finished.finish(1, false));
        assert_eq!(finished.prefix, 4);
        assert!(finished.ahead.is_empty());
    }

    #[test]
    fn waiters_are_woken_when_the_prefix_reaches_the_lowest_serial_wanted() {
        let mut finished = Finished::default();
        assert!(!finished.reached(3));
        assert!(!finished.reached(2));
        assert!(!finished.finish(1, false), "woke a waiter before its serial");
        assert!(finished.finish(2, false), "did not wake the waiter for serial 2");
        // The waiter for 3 asks again once woken; one that is already satisfied does not wait.
        assert!(finished.reached(2));
        assert!(!finished.reached(3));
        assert!(finished.finish(3, false));
        // A failure wakes everyone, whatever they wait for.
        assert!(!finished.reached(9));
        assert!(finished.finish(5, true));
        assert!(finished.reached(9));
    }

    #[test]
    fn retired_buffers_leave_the_pending_count() {
        // The budget counts what `poll` has not retired: the unfinished buffer and everything behind it.
        let mut pending = Pending::default();
        for id in 0..5 {
            pending.push(id);
        }
        assert_eq!(pending.poll(|&id| Ok(id < 2)).unwrap(), 2);
        assert_eq!(pending.buffers.len(), 3);
        assert_eq!(pending.poll(|&id| Ok(id != 2)).unwrap(), 2);
        assert_eq!(pending.buffers.len(), 3);
        assert_eq!(pending.poll(|_| Ok(true)).unwrap(), 5);
        assert!(pending.buffers.is_empty());
    }

    #[test]
    fn error_beyond_pending_prefix_is_reported() {
        let mut pending = Pending::default();
        pending.push(false);
        pending.push(true);
        assert!(pending
            .poll(|&failed| if failed {
                Err(crate::DeviceError::LostWithReason("abort".into()))
            } else {
                Ok(false)
            })
            .is_err());
        assert_eq!(pending.completed, 0);
    }
}
