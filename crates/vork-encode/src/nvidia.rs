//! Vulkan/CUDA boundary. wgpu owns the exported buffer; CUDA borrows its allocation.
//! A completed wgpu copy precedes NVENC submission. Graph ring entries overlap encode
//! and rendering; a slot is drained before its pixels are reused. Queue-family
//! release/acquire barriers make ownership explicit across the two APIs.
use super::*;
use ash::{
    khr::{external_memory_fd, external_semaphore_fd},
    vk,
};
use std::{
    ffi::c_void,
    io::Write,
    os::fd::{FromRawFd, IntoRawFd, OwnedFd},
    process::ChildStdin,
};
use wgpu::hal::{api::Vulkan, vulkan};

const VENDOR_ID: u32 = 0x10de;

/// Whether the rendering device is NVIDIA's: NVENC, native or through FFmpeg, follows that adapter.
pub(super) fn is_adapter(device: &wgpu::Device) -> bool {
    // SAFETY: the HAL guard retains the instance and physical device.
    unsafe { device.as_hal::<Vulkan>() }.is_some_and(|hal| {
        unsafe {
            hal.shared_instance()
                .raw_instance()
                .get_physical_device_properties(hal.raw_physical_device())
        }
        .vendor_id
            == VENDOR_ID
    })
}

unsafe extern "C" {
    fn vork_nv_prepare(error: *mut i32) -> *mut c_void;
    fn vork_nv_open(
        prepared: *mut c_void,
        fd: i32,
        size: u64,
        uuid: *const u8,
        w: u32,
        h: u32,
        pitch: u32,
        fps_num: u32,
        fps_den: u32,
        bitrate: u32,
        gop_frames: u32,
        primaries: u32,
        capacity: u32,
        error: *mut i32,
    ) -> *mut c_void;
    fn vork_nv_sync(session: *mut c_void, fd: i32) -> i32;
    fn vork_nv_push(session: *mut c_void, index: u64, bgra: i32, value: u64) -> i32;
    fn vork_nv_read(session: *mut c_void, slot: u32, data: *mut *const u8, size: *mut u32)
    -> i32;
    fn vork_nv_unlock(session: *mut c_void, slot: u32) -> i32;
    fn vork_nv_finish(session: *mut c_void, diagnostics: i32) -> i32;
    fn vork_nv_drop(session: *mut c_void);
}
struct Native(*mut c_void);
// SAFETY: preparation has no context; attached entry points bind/unbind it on
// the calling thread. Exclusive Session access prevents moving during a call.
unsafe impl Send for Native {}
// SAFETY: render and drain threads operate on distinct ring slots. Slot credits
// are returned only after unlock/unmap, and teardown joins the drain thread.
unsafe impl Sync for Native {}
impl Drop for Native {
    fn drop(&mut self) {
        unsafe { vork_nv_drop(self.0) }
    }
}

/// Loaded drivers and initialized CUDA, without a context or output file.
/// Attachment still selects the device by the renderer's Vulkan UUID.
pub(super) struct Prepared(Native);
impl Prepared {
    pub(super) fn new() -> Result<Self> {
        let mut error = 0;
        let native = unsafe { vork_nv_prepare(&mut error) };
        ensure!(!native.is_null(), "CUDA driver initialization status {error}");
        Ok(Self(Native(native)))
    }
    fn into_raw(mut self) -> *mut c_void {
        std::mem::replace(&mut self.0.0, std::ptr::null_mut())
    }
}

pub(super) struct Session {
    // Drop CUDA import before Vulkan memory. Neither can outlive the wgpu device.
    native: std::sync::Arc<Native>,
    sync: Option<Timeline>,
    buffer: wgpu::Buffer,
    raw_buffer: vk::Buffer,
    raw_device: ash::Device,
    family: u32,
    pitch: u32,
    capacity: usize,
    installed: usize,
    next: u64,
    child: Child,
    drain_jobs: Option<std::sync::mpsc::SyncSender<usize>>,
    drained: std::sync::mpsc::Receiver<usize>,
    drain_thread: Option<std::thread::JoinHandle<Result<f64>>>,
    copy_wait_ms: f64,
    mux_write_ms: f64,
}
/// Imported allocation range and completion state belonging to one graph entry.
pub(super) struct Surface {
    buffer: wgpu::Buffer,
    slot: usize,
    external: bool,
    busy: bool,
}
// CUDA import is dropped before this exported Vulkan payload. At teardown,
// wait for any signals queued before an error interrupted native submission.
struct Timeline {
    semaphore: vk::Semaphore,
    raw_device: ash::Device,
    device: wgpu::Device,
}
impl Drop for Timeline {
    fn drop(&mut self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        unsafe {
            self.raw_device.destroy_semaphore(self.semaphore, None);
        }
    }
}
// Keep raw Vulkan objects exception-safe until ownership transfers into wgpu.
struct Allocation {
    device: ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
impl Session {
    pub(super) fn open(
        gpu: &DeviceInfo,
        prepared: Prepared,
        width: u32,
        height: u32,
        fps: Time,
        bitrate: u32,
        gop_frames: u32,
        // The H.273 code of the frames' primaries, written to the stream's VUI.
        primaries: u32,
        path: &Path,
        ring_depth: usize,
    ) -> Result<Self> {
        // SAFETY: guards retain wgpu's device throughout native allocation/import.
        let hal = unsafe { gpu.device.as_hal::<Vulkan>() }.context("NVENC requires Vulkan")?;
        ensure!(
            hal.enabled_device_extensions()
                .contains(&external_memory_fd::NAME),
            "VK_KHR_external_memory_fd unavailable"
        );
        let device = hal.raw_device();
        let instance = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        let properties = unsafe { instance.get_physical_device_properties(physical) };
        ensure!(
            properties.vendor_id == VENDOR_ID,
            "NVENC requires an NVIDIA Vulkan device"
        );
        let mut ids = vk::PhysicalDeviceIDProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
        unsafe { instance.get_physical_device_properties2(physical, &mut properties2) };
        let pitch = (width * 4).div_ceil(COPY_ROW_ALIGNMENT as u32) * COPY_ROW_ALIGNMENT as u32;
        // The graph supplies the one ring depth to both allocation and native import.
        let capacity = u32::try_from(ring_depth)?;
        ensure!(capacity > 0, "NVENC ring capacity must be positive");
        let bytes = u64::from(pitch) * u64::from(height) * u64::from(capacity);
        validate_ring_limit(
            bytes, gpu.device.limits().max_buffer_size, width, height, capacity,
        )?;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let info = vk::BufferCreateInfo::default()
            .size(bytes)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .push_next(&mut external);
        let raw_buffer = unsafe { device.create_buffer(&info, None) }?;
        let mut allocation = Allocation {
            device: device.clone(),
            buffer: raw_buffer,
            memory: vk::DeviceMemory::null(),
        };
        let requirements = unsafe { device.get_buffer_memory_requirements(raw_buffer) };
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let memory_type = memory_properties
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .find(|(i, m)| {
                requirements.memory_type_bits & (1 << i) != 0
                    && m.property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .context("no device-local external memory type")?
            .0 as u32;
        let mut export = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(raw_buffer);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type)
            .push_next(&mut export)
            .push_next(&mut dedicated);
        allocation.memory = unsafe { device.allocate_memory(&allocate, None) }?;
        unsafe { device.bind_buffer_memory(raw_buffer, allocation.memory, 0) }?;
        // Fully initialize before importing into wgpu (including row padding). Native
        // recording uses wgpu's queue submission, preserving its queue synchronization.
        let mut initialize = gpu.device.create_command_encoder(&Default::default());
        unsafe {
            initialize.as_hal_mut::<Vulkan, _, _>(|encoder| {
                device.cmd_fill_buffer(encoder.unwrap().raw_handle(), raw_buffer, 0, bytes, 0);
            })
        };
        gpu.queue.submit([initialize.finish()]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        let export_api = external_memory_fd::Device::new(instance, device);
        let fd = unsafe {
            export_api.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(allocation.memory)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD),
            )
        }?;
        let fd = unsafe { OwnedFd::from_raw_fd(fd) };
        let mut error = 0;
        // CUDA consumes fd, including on failure. Native initialization owns its cleanup.
        let native = unsafe {
            vork_nv_open(
                prepared.into_raw(),
                fd.into_raw_fd(),
                requirements.size,
                ids.device_uuid.as_ptr(),
                width,
                height,
                pitch,
                fps.numerator() as u32,
                fps.denominator() as u32,
                bitrate,
                gop_frames,
                primaries,
                capacity,
                &mut error,
            )
        };
        ensure!(
            !native.is_null(),
            "CUDA/NVENC initialization status {error}"
        );
        let native = std::sync::Arc::new(Native(native));
        let sync = if hal
            .enabled_device_extensions()
            .contains(&external_semaphore_fd::NAME)
        {
            let mut export = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
            let mut timeline =
                vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
            let semaphore = unsafe {
                device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default()
                        .push_next(&mut export)
                        .push_next(&mut timeline),
                    None,
                )
            }?;
            let signal = Timeline {
                semaphore,
                raw_device: device.clone(),
                device: gpu.device.clone(),
            };
            let api = external_semaphore_fd::Device::new(instance, device);
            let fd = unsafe {
                api.get_semaphore_fd(
                    &vk::SemaphoreGetFdInfoKHR::default()
                        .semaphore(semaphore)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD),
                )
            }?;
            let status = unsafe { vork_nv_sync(native.0, fd) };
            ensure!(
                status == 0,
                "CUDA timeline semaphore import status {status}"
            );
            eprintln!("NVENC copy synchronization: Vulkan timeline → CUDA stream");
            Some(signal)
        } else {
            None
        };
        let mut child = crate::ffmpeg::mux_command(fps, path)?
            .stdin(Stdio::piped())
            .spawn()
            .context("start ffmpeg H.264 MP4 muxer: Linux MP4 requires ffmpeg on PATH even with NVENC; install with sudo apt install ffmpeg")?;
        let input = child.stdin.take().context("missing muxer input")?;
        let (drain_jobs, jobs) = std::sync::mpsc::sync_channel(capacity as usize);
        let (credits, drained) = std::sync::mpsc::channel();
        let worker_native = native.clone();
        let drain_thread = std::thread::Builder::new()
            .name("vork-nvenc-drain".into())
            .spawn(move || drain_bitstreams(worker_native, input, jobs, credits))?;
        let buffer = unsafe {
            gpu.device.create_buffer_from_hal::<Vulkan>(
                vulkan::Buffer::from_raw_managed(
                    raw_buffer,
                    allocation.memory,
                    0,
                    requirements.size,
                ),
                &wgpu::BufferDescriptor {
                    label: Some("NVENC external input"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )
        };
        // wgpu now owns both handles.
        allocation.buffer = vk::Buffer::null();
        allocation.memory = vk::DeviceMemory::null();
        Ok(Self {
            native,
            sync,
            buffer,
            raw_buffer,
            raw_device: device.clone(),
            family: hal.queue_family_index(),
            pitch,
            capacity: capacity as usize,
            installed: 0,
            next: 0,
            child,
            drain_jobs: Some(drain_jobs),
            drained,
            drain_thread: Some(drain_thread),
            copy_wait_ms: 0.,
            mux_write_ms: 0.,
        })
    }
    pub(super) fn create_surface(&mut self) -> Result<Surface> {
        ensure!(
            self.installed < self.capacity,
            "all NVENC ring entries are installed"
        );
        let surface = Surface {
            buffer: self.buffer.clone(),
            slot: self.installed,
            external: false,
            busy: false,
        };
        self.installed += 1;
        Ok(surface)
    }
    fn ownership(&self, encoder: &mut wgpu::CommandEncoder, release: bool, slot: usize) {
        let (src, dst, src_access, dst_access, src_stage, dst_stage) = if release {
            (
                self.family,
                vk::QUEUE_FAMILY_EXTERNAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            )
        } else {
            (
                vk::QUEUE_FAMILY_EXTERNAL,
                self.family,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            )
        };
        let barrier = vk::BufferMemoryBarrier::default()
            .buffer(self.raw_buffer)
            .offset(self.buffer.size() / self.capacity as u64 * slot as u64)
            .size(self.buffer.size() / self.capacity as u64)
            .src_queue_family_index(src)
            .dst_queue_family_index(dst)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access);
        // SAFETY: wgpu serializes native recording. The buffer remains alive; this
        // only changes queue ownership, while wgpu tracks its COPY_DST use.
        unsafe {
            encoder.as_hal_mut::<Vulkan, _, _>(|encoder| {
                self.raw_device.cmd_pipeline_barrier(
                    encoder.unwrap().raw_handle(),
                    src_stage,
                    dst_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[barrier],
                    &[],
                );
            })
        }
    }
    pub(super) fn push(
        &mut self,
        gpu: &DeviceInfo,
        texture: &wgpu::Texture,
        index: u64,
        surface: &mut Surface,
    ) -> Result<()> {
        let slot = surface.slot;
        ensure!(
            slot == (index % self.capacity as u64) as usize,
            "NVENC entry submitted out of ring order"
        );
        ensure!(
            !surface.busy,
            "release the ring entry before reusing its NVENC surface"
        );
        let copy_start = std::time::Instant::now();
        // wgpu 29 forbids mixing raw and wgpu recording within one encoder.
        // Submit these command buffers together, in ownership/copy/release order.
        let mut commands = Vec::with_capacity(3);
        if surface.external {
            let mut acquire = gpu.device.create_command_encoder(&Default::default());
            self.ownership(&mut acquire, false, slot);
            commands.push(acquire.finish());
        }
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &surface.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: self.buffer.size() / self.capacity as u64 * slot as u64,
                    bytes_per_row: Some(self.pitch),
                    rows_per_image: None,
                },
            },
            texture.size(),
        );
        commands.push(encoder.finish());
        let mut release = gpu.device.create_command_encoder(&Default::default());
        self.ownership(&mut release, true, slot);
        commands.push(release.finish());
        // The renderer is the ordered producer on this queue: attach the signal
        // immediately before submitting the copy that transfers this slot.
        if let Some(sync) = &self.sync {
            let queue =
                unsafe { gpu.queue.as_hal::<Vulkan>() }.context("Vulkan queue unavailable")?;
            queue.add_signal_semaphore(sync.semaphore, Some(index + 1));
        }
        let submission = gpu.queue.submit(commands);
        if self.sync.is_none() {
            gpu.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })?;
        }
        self.copy_wait_ms += copy_start.elapsed().as_secs_f64() * 1000.;
        surface.external = true;
        let status = unsafe {
            vork_nv_push(
                self.native.0,
                index,
                i32::from(texture.format() == wgpu::TextureFormat::Bgra8Unorm),
                index + 1,
            )
        };
        ensure!(status == 0, "NVENC frame {index} status {status}");
        self.drain_jobs
            .as_ref()
            .context("closed drain worker")?
            .send(slot)?;
        surface.busy = true;
        self.next = index + 1;
        Ok(())
    }
    pub(super) fn release(&mut self, index: u64, surface: &mut Surface) -> Result<()> {
        let slot = surface.slot;
        ensure!(
            slot == (index % self.capacity as u64) as usize,
            "NVENC entry released out of ring order"
        );
        if surface.busy {
            let completed = self.drained.recv().context("NVENC drain worker stopped")?;
            ensure!(completed == slot, "NVENC slot completion out of order");
            surface.busy = false;
        }
        Ok(())
    }
    pub(super) fn finish(&mut self, diagnostics: bool) -> Result<()> {
        drop(self.drain_jobs.take());
        if let Some(worker) = self.drain_thread.take() {
            self.mux_write_ms = worker
                .join()
                .map_err(|_| anyhow::anyhow!("NVENC drain worker panicked"))??;
        }
        let status = unsafe { vork_nv_finish(self.native.0, i32::from(diagnostics)) };
        ensure!(status == 0, "NVENC EOS status {status}");
        let mux_start = std::time::Instant::now();
        ensure!(self.child.wait()?.success(), "NVENC MP4 muxer failed");
        if diagnostics {
        eprintln!(
            "nvenc stages: copy_wait_ms={:.3} mux_write_ms={:.3} mux_finish_ms={:.3}",
            self.copy_wait_ms,
            self.mux_write_ms,
            mux_start.elapsed().as_secs_f64() * 1000.
        );
        }
        Ok(())
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.child.kill();
        drop(self.drain_jobs.take());
        if let Some(worker) = self.drain_thread.take() {
            let _ = worker.join();
        }
        let _ = self.child.wait();
    }
}

// One ordered consumer writes compressed bytes while the render thread feeds
// other ring slots. A returned credit means both bitstream and input are free.
fn drain_bitstreams(
    native: std::sync::Arc<Native>,
    mut input: ChildStdin,
    jobs: std::sync::mpsc::Receiver<usize>,
    credits: std::sync::mpsc::Sender<usize>,
) -> Result<f64> {
    let mut mux_write_ms = 0.;
    for slot in jobs {
        let mut data = std::ptr::null();
        let mut size = 0;
        let status = unsafe { vork_nv_read(native.0, slot as u32, &mut data, &mut size) };
        ensure!(status == 0, "NVENC bitstream read status {status}");
        let start = std::time::Instant::now();
        let result = input.write_all(unsafe { std::slice::from_raw_parts(data, size as usize) });
        mux_write_ms += start.elapsed().as_secs_f64() * 1000.;
        let status = unsafe { vork_nv_unlock(native.0, slot as u32) };
        result?;
        ensure!(status == 0, "NVENC bitstream unlock status {status}");
        if credits.send(slot).is_err() {
            break;
        }
    }
    Ok(mux_write_ms)
}

// Keep the allocation predicate in one CPU-testable check; limits come from the device.
fn validate_ring_limit(
    bytes: u64,
    max_buffer_size: u64,
    width: u32,
    height: u32,
    capacity: u32,
) -> Result<()> {
    ensure!(
        bytes <= max_buffer_size,
        gpu_memory::device_limit(
            "NVENC ring",
            format_args!("{bytes} bytes at {width}x{height} across {capacity} frames"),
            format_args!("max_buffer_size={max_buffer_size} bytes"),
            "reduce the output resolution or use a GPU with a larger buffer limit",
        )
    );
    Ok(())
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    #[test]
    fn nvenc_ring_refusal_reports_demand_limit_and_recovery() {
        for (bytes, limit, width, height, capacity) in [(1024, 1023, 8, 2, 2), (4096, 2048, 16, 4, 4)] {
            let error = validate_ring_limit(bytes, limit, width, height, capacity).unwrap_err();
            assert!(error.is::<DeviceLimit>());
            let message = error.to_string();
            for expected in [
                format!("{bytes} bytes"),
                format!("{width}x{height}"),
                format!("{capacity} frames"),
                format!("max_buffer_size={limit} bytes"),
                "reduce the output resolution".into(),
                "vork help limits".into(),
            ] {
                assert!(message.contains(&expected), "{message}");
            }
            validate_ring_limit(bytes, bytes, width, height, capacity).unwrap();
            validate_ring_limit(bytes, bytes + 1, width, height, capacity).unwrap();
        }
    }
}
