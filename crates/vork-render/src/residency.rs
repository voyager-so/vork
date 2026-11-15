//! Device-wide retention. A lease pins a cache through its last GPU consumer;
//! pressure may only drop entries with no leases. No renderer byte allowance.
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, Ordering},
};

trait Entry: Send + Sync {
    fn age(&self) -> u64;
    fn evict(&self) -> u64;
    fn bytes(&self) -> u64;
}
#[derive(Default)]
pub(crate) struct Ledger {
    clock: AtomicU64,
    entries: Mutex<Vec<Weak<dyn Entry>>>,
}
struct State<T> {
    value: Option<Arc<T>>,
    bytes: u64,
}
struct Inner<T> {
    state: Mutex<State<T>>,
    touched: AtomicU64,
}
impl<T: Send + Sync> Entry for Inner<T> {
    fn age(&self) -> u64 {
        self.touched.load(Ordering::Relaxed)
    }
    fn bytes(&self) -> u64 {
        self.state.lock().unwrap().bytes
    }
    fn evict(&self) -> u64 {
        let Ok(mut state) = self.state.try_lock() else {
            return 0;
        };
        if state
            .value
            .as_ref()
            .is_some_and(|v| Arc::strong_count(v) == 1)
        {
            state.value.take();
            std::mem::take(&mut state.bytes)
        } else {
            0
        }
    }
}
pub(crate) struct Cache<T> {
    inner: Arc<Inner<T>>,
    ledger: Arc<Ledger>,
}
impl<T: Send + Sync + 'static> Cache<T> {
    pub(crate) fn new(ledger: &Arc<Ledger>) -> Self {
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                value: None,
                bytes: 0,
            }),
            touched: AtomicU64::new(0),
        });
        let entry: Arc<dyn Entry> = inner.clone();
        ledger.entries.lock().unwrap().push(Arc::downgrade(&entry));
        Self {
            inner,
            ledger: ledger.clone(),
        }
    }
    pub(crate) fn set_bytes(&self, bytes: u64) {
        self.inner.state.lock().unwrap().bytes = bytes;
    }
    pub(crate) fn is_resident(&self) -> bool {
        self.inner.state.lock().unwrap().value.is_some()
    }
    pub(crate) fn get(&self) -> Option<Arc<T>> {
        let value = self.inner.state.lock().unwrap().value.clone();
        if value.is_some() {
            self.inner.touched.store(
                self.ledger.clock.fetch_add(1, Ordering::Relaxed),
                Ordering::Relaxed,
            );
        }
        value
    }
    pub(crate) fn insert(&self, value: T, bytes: u64) -> Arc<T> {
        let value = Arc::new(value);
        *self.inner.state.lock().unwrap() = State {
            value: Some(value.clone()),
            bytes,
        };
        self.inner.touched.store(
            self.ledger.clock.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        value
    }
    pub(crate) fn get_or_try_init(
        &self,
        build: impl FnOnce() -> anyhow::Result<(T, u64)>,
    ) -> anyhow::Result<Arc<T>> {
        let mut state = self.inner.state.lock().unwrap();
        if state.value.is_none() {
            let (value, bytes) = build()?;
            *state = State {
                value: Some(Arc::new(value)),
                bytes,
            };
        }
        self.inner.touched.store(
            self.ledger.clock.fetch_add(1, Ordering::Relaxed),
            Ordering::Relaxed,
        );
        Ok(state.value.as_ref().unwrap().clone())
    }
    pub(crate) fn take(&self) -> Option<Arc<T>> {
        let mut state = self.inner.state.lock().unwrap();
        state.bytes = 0;
        state.value.take()
    }
}
impl Ledger {
    /// Native allocation is authoritative. Retire consumers only after pressure,
    /// then release one LRU entry per retry; never flush a roomy device's caches.
    pub(crate) fn retry<T, E>(
        &self,
        mut allocate: impl FnMut() -> Result<T, E>,
        pressure: impl Fn(&E) -> bool,
        mut retire: impl FnMut() -> Result<(), E>,
    ) -> Result<T, E> {
        loop {
            match allocate() {
                Ok(value) => return Ok(value),
                Err(error) => {
                    if !pressure(&error) { return Err(error); }
                    retire()?;
                    if self.evict(1) == 0 { return Err(error); }
                }
            }
        }
    }

    pub(crate) fn retained_bytes(&self) -> u64 {
        let entries: Vec<_> = self
            .entries
            .lock()
            .unwrap()
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        entries.iter().map(|e| e.bytes()).sum()
    }
    /// Release least recently useful entries until demand is covered. Pinned
    /// entries are skipped, and revisited after dependent cache entries drop.
    pub(crate) fn evict(&self, needed: u64) -> u64 {
        let mut registry = self.entries.lock().unwrap();
        registry.retain(|e| e.strong_count() > 0);
        let mut entries: Vec<_> = registry.iter().filter_map(Weak::upgrade).collect();
        drop(registry);
        entries.sort_by_key(|e| e.age());
        let mut freed: u64 = 0;
        loop {
            let before = freed;
            for entry in &entries {
                if freed >= needed {
                    return freed;
                }
                freed = freed.saturating_add(entry.evict());
            }
            if before == freed {
                return freed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_headroom_ignores_staging_but_not_local_pressure() {
        assert_eq!(local_headroom::<u64>([(true, 4<<30), (false, 32<<30)]), Some(4<<30));
        assert_eq!(local_headroom::<u64>([(true, 0), (false, 32<<30)]), Some(0));
        assert_eq!(local_headroom::<u64>([(true, 4<<30), (true, 2<<30)]), None);
        assert_eq!(local_headroom::<u64>([(false, 32<<30)]), None);
    }

    #[test]
    fn camera_usage_and_headroom_come_from_the_same_local_heap() {
        let heaps = [(false, (1_u64 << 30, 31_u64 << 30)), (true, (2 << 30, 6 << 30))];
        assert_eq!(local_headroom(heaps), Some((2 << 30, 6 << 30)));
        assert_eq!(local_headroom([(true, (2_u64 << 30, 6_u64 << 30)),
            (true, (4 << 30, 1 << 30))]), None);
    }

    #[test]
    fn unrelated_heaps_never_supply_an_admission_budget() {
        assert_eq!(common_headroom::<u64>([]), None);
        assert_eq!(common_headroom::<u64>([4096]), Some(4096));
        // Device-local small/large heaps, and host-visible staging respectively.
        assert_eq!(common_headroom::<u64>([0, 4096]), None);
        assert_eq!(common_headroom::<u64>([4096, 0]), None);
        assert_eq!(common_headroom::<u64>([0, 4096, 8192]), None);
    }
    #[test]
    fn native_retry_evicts_only_after_pressure_and_stops_when_pinned() {
        let ledger=Arc::new(Ledger::default());
        let a=Cache::new(&ledger); let b=Cache::new(&ledger);
        drop(a.insert(1,10)); let lease=b.insert(2,20);
        let mut retired=0;
        assert_eq!(ledger.retry(||Ok::<_,bool>(7),|e|*e,||{retired+=1;Ok(())}),Ok(7));
        assert_eq!(retired,0); assert_eq!(ledger.retained_bytes(),30);
        assert_eq!(ledger.retry(||Err::<(),_>(false),|e|*e,||{retired+=1;Ok(())}),Err(false));
        assert_eq!(retired,0); assert!(a.is_resident());
        let mut attempts=0;
        assert_eq!(ledger.retry(||{attempts+=1;if attempts==1 {Err(true)} else {Ok(9)}},
            |e|*e,||{retired+=1;Ok(())}),Ok(9));
        assert_eq!(attempts,2); assert_eq!(retired,1);
        assert!(!a.is_resident()); assert!(b.is_resident());
        assert_eq!(ledger.retry(||Err::<(),_>(true),|e|*e,||Ok(())),Err(true));
        assert!(b.is_resident()); drop(lease);
    }
    #[test]
    fn retry_exhausts_unpinned_storage_without_looping_forever() {
        let ledger=Arc::new(Ledger::default());
        let a=Cache::new(&ledger); let b=Cache::new(&ledger);
        drop(a.insert(1,10)); drop(b.insert(2,20));
        let mut attempts=0;
        assert_eq!(ledger.retry(||{attempts+=1;Err::<(),_>(true)},|e|*e,||Ok(())),Err(true));
        assert_eq!(attempts,3); assert_eq!(ledger.retained_bytes(),0);
    }
    #[test]
    fn pressure_respects_leases_and_use_order() {
        let ledger = Arc::new(Ledger::default());
        let a = Cache::new(&ledger);
        let b = Cache::new(&ledger);
        let pinned = a.insert(1, 30);
        drop(b.insert(2, 20));
        assert_eq!(ledger.retained_bytes(), 50);
        assert_eq!(ledger.evict(1), 20);
        assert!(b.get().is_none());
        assert_eq!(ledger.evict(1), 0);
        drop(pinned);
        drop(b.insert(2, 20));
        drop(a.get());
        assert_eq!(ledger.evict(1), 20);
        assert!(a.get().is_some());
        assert_eq!(ledger.evict(u64::MAX), 30);
        assert_eq!(ledger.retained_bytes(), 0);
    }
    #[test]
    fn registry_does_not_retain_abandoned_caches() {
        let ledger = Arc::new(Ledger::default());
        {
            let cache = Cache::new(&ledger);
            drop(cache.insert(1, 100));
        }
        assert_eq!(ledger.retained_bytes(), 0);
        assert_eq!(ledger.evict(1), 0);
    }
}

/// Only a single shared Vulkan heap has resource-independent headroom. wgpu's
/// selected memory type is not exposed here: multiple heaps (including staging
/// heaps) are unknown, never a minimum/maximum admission budget.
fn common_headroom<T>(heaps: impl IntoIterator<Item = T>) -> Option<T> {
    let mut heaps = heaps.into_iter();
    let first = heaps.next()?;
    heaps.next().is_none().then_some(first)
}

/// Snapshot bytes of startup headroom and its matching usage baseline together.
/// The qualification override lowers headroom, so it has the same relative basis.
pub(crate) fn camera_budget(device: &wgpu::Device, device_memory: u64, override_bytes: Option<u64>) -> (u64, u64, bool) {
    let (usage, headroom) = camera_usage_headroom(device).unwrap_or_else(|| {
        let usage = device.generate_allocator_report().map_or(0, |m| m.total_reserved_bytes);
        (usage, device_memory.saturating_sub(usage))
    });
    let native_budget = headroom.min(device_memory);
    let budget = override_bytes.map_or(native_budget, |bytes| native_budget.min(bytes));
    (budget, usage, budget < native_budget)
}

/// Process usage and currently usable headroom in bytes from the same native
/// heap. Driver reservations and other processes reduce headroom, not our usage.
/// Camera storage selects the single local heap independently of host staging;
/// multiple local heaps remain unknown because their allocation type is opaque.
pub(crate) fn camera_usage_headroom(device: &wgpu::Device) -> Option<(u64, u64)> {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if let Some(hal) = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() } {
        use ash::{ext, vk};
        if !hal.enabled_device_extensions().contains(&ext::memory_budget::NAME) {return None;}
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {hal.shared_instance().raw_instance().get_physical_device_memory_properties2(hal.raw_physical_device(), &mut properties);}
        return local_headroom(properties.memory_properties.memory_heaps_as_slice().iter().enumerate()
            .map(|(i, heap)| (heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL), (budget.heap_usage[i], budget.heap_budget[i].saturating_sub(budget.heap_usage[i])))));
    }
    #[cfg(target_os = "macos")]
    if let Some(hal) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
        use objc2_metal::MTLDevice;
        let native = hal.raw_device();
        let capacity = native.recommendedMaxWorkingSetSize();
        let usage = native.currentAllocatedSize() as u64;
        return Some((usage, capacity.saturating_sub(usage)));
    }
    None
}
fn local_headroom<T>(heaps: impl IntoIterator<Item = (bool, T)>) -> Option<T> {
    common_headroom(heaps.into_iter().filter_map(|(local, free)| local.then_some(free)))
}

pub(crate) fn available(device: &wgpu::Device) -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if let Some(hal) = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() } {
        use ash::{ext, vk};
        if !hal.enabled_device_extensions().contains(&ext::memory_budget::NAME) {
            return None;
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {
            hal.shared_instance().raw_instance().get_physical_device_memory_properties2(
                hal.raw_physical_device(), &mut properties);
        }
        return common_headroom((0..properties.memory_properties.memory_heap_count as usize)
            .map(|i| budget.heap_budget[i].saturating_sub(budget.heap_usage[i])));
    }
    #[cfg(target_os = "macos")]
    if let Some(hal) = unsafe { device.as_hal::<wgpu::hal::api::Metal>() } {
        use objc2_metal::MTLDevice;
        let native = hal.raw_device();
        return Some(native.recommendedMaxWorkingSetSize()
            .saturating_sub(native.currentAllocatedSize() as u64));
    }
    None
}

/// Headroom for the heap of an existing buffer, only when its actual allocator
/// properties identify one heap. Never combine device and staging heap budgets.
pub(crate) fn buffer_available(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Option<u64> {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if let (Some(hal),Some(native)) = (unsafe {device.as_hal::<wgpu::hal::api::Vulkan>()}, unsafe {buffer.as_hal::<wgpu::hal::api::Vulkan>()}) {
        use ash::{ext,vk};
        if !hal.enabled_device_extensions().contains(&ext::memory_budget::NAME) {return None;}
        let flags=native.memory_properties()?;
        let mut budget=vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties=vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        unsafe {hal.shared_instance().raw_instance().get_physical_device_memory_properties2(hal.raw_physical_device(),&mut properties);}
        let heap=unique_heap(properties.memory_properties.memory_types_as_slice().iter()
            .filter(|m|m.property_flags==flags).map(|m|m.heap_index))? as usize;
        return Some(budget.heap_budget[heap].saturating_sub(budget.heap_usage[heap]));
    }
    available(device)
}
#[cfg(any(target_os="linux",target_os="windows",test))]
fn unique_heap(heaps: impl IntoIterator<Item=u32>) -> Option<u32> {
    let mut heaps=heaps.into_iter();let first=heaps.next()?;
    heaps.all(|heap|heap==first).then_some(first)
}
#[cfg(test)]
mod buffer_heap_tests {
    #[test]
    fn actual_properties_must_identify_one_heap() {
        assert_eq!(super::unique_heap([0,0,0]),Some(0));
        assert_eq!(super::unique_heap([1]),Some(1));
        assert_eq!(super::unique_heap([0,1]),None);
        assert_eq!(super::unique_heap([]),None);
    }
}
