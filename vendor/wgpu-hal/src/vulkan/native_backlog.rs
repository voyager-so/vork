//! Native command storage remains pending until its queue prefix completes.
use super::map_host_device_oom_and_lost_err;
use alloc::{collections::VecDeque, vec::Vec};
use ash::{vk, Device};

// Scheduling budget established by the L4/gVisor deep-backlog reproducer; Metal keeps the same one.
pub(super) const LIMIT: usize = crate::MAX_PENDING_NATIVE_BUFFERS;

#[derive(Default)]
pub(super) struct NativeBacklog {
    pending: VecDeque<(vk::Fence, usize)>,
    free: Vec<vk::Fence>,
    buffers: usize,
}
impl NativeBacklog {
    fn retire(&mut self) {
        let (fence, count) = self.pending.pop_front().unwrap();
        self.buffers -= count;
        self.free.push(fence);
    }
    // The callbacks make the cross-call accounting independently testable.
    fn make_room<E>(
        &mut self,
        incoming: usize,
        mut completed: impl FnMut(vk::Fence) -> Result<bool, E>,
        mut wait: impl FnMut(vk::Fence) -> Result<(), E>,
    ) -> Result<(), E> {
        assert!((1..=LIMIT).contains(&incoming));
        while let Some(&(fence, _)) = self.pending.front() {
            if completed(fence)? {
                self.retire();
            } else if self.buffers + incoming > LIMIT {
                wait(fence)?;
                self.retire();
            } else {
                break;
            }
        }
        Ok(())
    }
    pub(super) unsafe fn prepare(
        &mut self,
        device: &Device,
        incoming: usize,
    ) -> Result<vk::Fence, crate::DeviceError> {
        self.make_room(
            incoming,
            |fence| {
                unsafe { device.get_fence_status(fence) }.map_err(map_host_device_oom_and_lost_err)
            },
            |fence| {
                unsafe { device.wait_for_fences(&[fence], true, u64::MAX) }
                    .map_err(map_host_device_oom_and_lost_err)
            },
        )?;
        if let Some(fence) = self.free.pop() {
            unsafe { device.reset_fences(&[fence]) }.map_err(map_host_device_oom_and_lost_err)?;
            Ok(fence)
        } else {
            unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .map_err(map_host_device_oom_and_lost_err)
        }
    }
    pub(super) fn cancelled(&mut self, fence: vk::Fence) {
        self.free.push(fence);
    }
    pub(super) fn submitted(&mut self, fence: vk::Fence, count: usize) {
        self.pending.push_back((fence, count));
        self.buffers += count;
        debug_assert!(self.buffers <= LIMIT);
    }
    pub(super) unsafe fn destroy(&mut self, device: &Device) {
        for (fence, _) in self.pending.drain(..) {
            unsafe { device.destroy_fence(fence, None) };
        }
        for fence in self.free.drain(..) {
            unsafe { device.destroy_fence(fence, None) };
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use ash::vk::Handle;
    #[test]
    fn small_submissions_share_the_pending_budget() {
        let mut backlog = NativeBacklog::default();
        let mut waited = Vec::new();
        // Individually far below the limit, collectively 32 times the budget.
        for i in 1..=512 {
            backlog
                .make_room(
                    8,
                    |_| Ok::<_, ()>(false),
                    |f| {
                        waited.push(f.as_raw());
                        Ok(())
                    },
                )
                .unwrap();
            backlog.submitted(vk::Fence::from_raw(i), 8);
            assert!(backlog.buffers <= LIMIT);
        }
        assert_eq!(waited, (1..=496).collect::<Vec<_>>());
        assert_eq!(backlog.buffers, 128);
        // Already-completed work is reclaimed without a blocking wait.
        backlog
            .make_room(
                128,
                |_| Ok::<_, ()>(true),
                |_| panic!("completed prefix must not wait"),
            )
            .unwrap();
        assert_eq!(backlog.buffers, 0);
    }
    #[test]
    fn waits_only_for_the_oldest_necessary_prefix() {
        let mut backlog = NativeBacklog::default();
        for (id, n) in [(1, 48), (2, 32), (3, 48)] {
            backlog.submitted(vk::Fence::from_raw(id), n);
        }
        let mut waited = Vec::new();
        backlog
            .make_room(
                64,
                |_| Ok::<_, ()>(false),
                |f| {
                    waited.push(f.as_raw());
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(waited, vec![1, 2]);
        assert_eq!(backlog.buffers, 48);
    }
}
