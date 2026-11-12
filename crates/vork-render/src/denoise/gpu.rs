//! Same-device OIDN memory sharing. wgpu owns native allocations; OIDN imports
//! remain alive until filters retire. Only GPU buffers cross this boundary.
use super::*;

#[cfg(any(target_os = "linux", target_os = "windows"))]
const STORAGE_USAGE: ash::vk::BufferUsageFlags = ash::vk::BufferUsageFlags::from_raw(ash::vk::BufferUsageFlags::STORAGE_BUFFER.as_raw() | ash::vk::BufferUsageFlags::TRANSFER_DST.as_raw());
#[cfg(target_os = "linux")]
const EXTERNAL_HANDLE: ash::vk::ExternalMemoryHandleTypeFlags = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
#[cfg(target_os = "windows")]
const EXTERNAL_HANDLE: ash::vk::ExternalMemoryHandleTypeFlags = ash::vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32;
// OIDNExternalMemoryTypeFlag for EXTERNAL_HANDLE.
#[cfg(target_os = "linux")]
const OIDN_EXTERNAL_TYPE: u32 = 1;
#[cfg(target_os = "windows")]
const OIDN_EXTERNAL_TYPE: u32 = 4;
// OIDN device types that can share Vulkan memory, most preferred first. CUDA
// stays first so NVIDIA hosts keep selecting the backend they used before;
// HIP follows on Linux so AMD hosts can use the same zero-copy path.
#[cfg(target_os = "linux")]
const GPU_BACKENDS: &[(i32, &str, Identity)] = &[
    (3, "CUDA", Identity::VulkanUuid),
    (4, "HIP", Identity::PciAddress),
];
#[cfg(target_os = "windows")]
const GPU_BACKENDS: &[(i32, &str, Identity)] = &[(3, "CUDA", Identity::VulkanUuid)];
#[cfg(target_os = "linux")]
const GPU_BACKENDS_LABEL: &str = "CUDA or HIP";
#[cfg(target_os = "windows")]
const GPU_BACKENDS_LABEL: &str = "CUDA";

/// How an OIDN backend names the renderer's adapter.
#[cfg(any(target_os = "linux", target_os = "windows"))]
#[derive(Clone, Copy, PartialEq, Eq)]
enum Identity {
    /// The backend reports the 16-byte UUID Vulkan exposes (CUDA).
    VulkanUuid,
    /// OIDN's HIP backend names an AMD device by its hardware UUID, which RADV
    /// does not expose (RADV derives its own Vulkan UUID), so HIP devices are
    /// matched by PCI address, which both OIDN and VK_EXT_pci_bus_info report.
    #[cfg(target_os = "linux")]
    PciAddress,
}

/// `domain:bus:device.function`, the PCI address spelling used in diagnostics.
#[cfg(target_os = "linux")]
fn pci_bdf(address: [u32; 4]) -> String {
    format!("{:04x}:{:02x}:{:02x}.{:x}", address[0], address[1], address[2], address[3])
}

/// A zero PCI address means the driver left the properties unset; no real
/// device lives at 0000:00:00.0.
#[cfg(target_os = "linux")]
fn set_pci(address: [u32; 4]) -> Option<[u32; 4]> {
    (address != [0; 4]).then_some(address)
}

/// The adapter's `domain:bus:device.function`, if its Vulkan driver reports it
/// through VK_EXT_pci_bus_info. An all-zero address means the driver left the
/// struct unset, so it yields `None`.
#[cfg(target_os = "linux")]
fn adapter_pci(instance: &ash::Instance, physical: ash::vk::PhysicalDevice) -> Option<[u32; 4]> {
    let mut pci = ash::vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
    let mut properties = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut pci);
    unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
    set_pci([pci.pci_domain, pci.pci_bus, pci.pci_device, pci.pci_function])
}

/// The physical device's `domain:bus:device.function`, if OIDN reports it.
/// Only the HIP backend populates this; CUDA devices may leave it unset.
#[cfg(target_os = "linux")]
fn oidn_pci(api: &Api, index: i32) -> Option<[u32; 4]> {
    if unsafe { (api.physical_int)(index, c"pciAddressSupported".as_ptr()) } == 0 { return None; }
    Some([
        unsafe { (api.physical_int)(index, c"pciDomain".as_ptr()) } as u32,
        unsafe { (api.physical_int)(index, c"pciBus".as_ptr()) } as u32,
        unsafe { (api.physical_int)(index, c"pciDevice".as_ptr()) } as u32,
        unsafe { (api.physical_int)(index, c"pciFunction".as_ptr()) } as u32,
    ])
}
#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" { fn CloseHandle(handle: Handle) -> i32; }

/// Return an already committed GPU device that can share this renderer's memory,
/// or why there is none. Metal must retain its queue at commit.
pub(super) fn select(api: &Api, r: &Renderer) -> std::result::Result<Handle, String> {
    let committed = |handle: Handle, name: &str| -> std::result::Result<Handle, String> {
        if handle.is_null() { return Err(format!("OIDN could not create its {name} device")); }
        unsafe { (api.commit_device)(handle); }
        let mut message = std::ptr::null();
        if unsafe { (api.error)(handle, &mut message) } != 0 {
            let message = if message.is_null() { "unknown error".into() } else { unsafe { CStr::from_ptr(message) }.to_string_lossy() };
            let error = format!("OIDN could not start its {name} device: {message}");
            unsafe { (api.release_device)(handle); }
            return Err(error);
        }
        Ok(handle)
    };
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    if let Some(hal) = unsafe { r.device.as_hal::<wgpu::hal::api::Vulkan>() } {
        use ash::vk;
        #[cfg(target_os = "linux")]
        let extension = ash::khr::external_memory_fd::NAME;
        #[cfg(target_os = "windows")]
        let extension = ash::khr::external_memory_win32::NAME;
        if !hal.enabled_device_extensions().contains(&extension) {
            return Err(format!("its Vulkan driver lacks {}", extension.to_string_lossy()));
        }
        let instance = hal.shared_instance().raw_instance();
        let query = vk::PhysicalDeviceExternalBufferInfo::default().usage(STORAGE_USAGE).handle_type(EXTERNAL_HANDLE);
        let mut external = vk::ExternalBufferProperties::default();
        unsafe { instance.get_physical_device_external_buffer_properties(hal.raw_physical_device(), &query, &mut external); }
        if !external.external_memory_properties.external_memory_features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE) {
            return Err("its Vulkan driver cannot export storage memory".into());
        }
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        unsafe { instance.get_physical_device_properties2(hal.raw_physical_device(), &mut properties); }
        #[cfg(target_os = "linux")]
        let adapter = adapter_pci(instance, hal.raw_physical_device());
        let count = unsafe { (api.physical_count)() };
        #[cfg(target_os = "linux")]
        let mut hip_seen = false;
        for &(device_type, label, identity) in GPU_BACKENDS {
            for index in 0..count {
                if unsafe { (api.physical_int)(index, c"type".as_ptr()) } != device_type { continue; }
                #[cfg(target_os = "linux")]
                if device_type == 4 { hip_seen = true; }
                let matching = match identity {
                    Identity::VulkanUuid => {
                        let mut bytes = 0;
                        let uuid = unsafe { (api.physical_data)(index, c"uuid".as_ptr(), &mut bytes) };
                        if bytes != 16 || uuid.is_null() { continue; }
                        let uuid = unsafe { std::slice::from_raw_parts(uuid.cast::<u8>(), 16) };
                        uuid == id.device_uuid
                    }
                    #[cfg(target_os = "linux")]
                    Identity::PciAddress => adapter.is_some() && oidn_pci(api, index) == adapter,
                };
                if !matching { continue; }
                let handle = committed(unsafe { (api.new_by_id)(index) }, label)?;
                let supported = unsafe { (api.device_int)(handle, c"externalMemoryTypes".as_ptr()) };
                if supported as u32 & OIDN_EXTERNAL_TYPE == 0 {
                    unsafe { (api.release_device)(handle); }
                    return Err(format!("OIDN's {label} device cannot import Vulkan memory"));
                }
                return Ok(handle);
            }
        }
        #[cfg(target_os = "linux")]
        return Err(match adapter {
            Some(address) if hip_seen =>
                format!("OIDN found no {GPU_BACKENDS_LABEL} device matching it at PCI {}", pci_bdf(address)),
            Some(address) =>
                format!("OIDN found no {GPU_BACKENDS_LABEL} device matching it at PCI {}; for AMD, OIDN's HIP backend needs the ROCm runtime", pci_bdf(address)),
            None => format!("OIDN found no {GPU_BACKENDS_LABEL} device matching it; its Vulkan driver reports no PCI address (VK_EXT_pci_bus_info)"),
        });
        #[cfg(target_os = "windows")]
        return Err(format!("OIDN found no {GPU_BACKENDS_LABEL} device matching it"));
    }
    #[cfg(target_os = "macos")]
    if let Some(hal) = unsafe { r.device.as_hal::<wgpu::hal::api::Metal>() } {
        use objc2_metal::MTLDevice;
        let native = hal.raw_device();
        let ptr = &**native as *const _ as *mut c_void;
        if unsafe { (api.metal_supported)(ptr) } {
            let queue = native.newCommandQueue().ok_or("OIDN could not get a Metal command queue")?;
            let ptr = &*queue as *const _ as *mut c_void;
            // Commit while the caller's queue is alive: MetalEngine retains it
            // during initialization (OIDN 2.5.1 metal_engine.mm).
            return committed(unsafe { (api.new_metal)(&ptr, 1) }, "Metal");
        }
        return Err("OIDN does not support this Metal device".into());
    }
    let _ = (api, r, &committed);
    Err("OIDN has no GPU device for this backend".into())
}

struct Shared {
    // Release OIDN's imported memory before wgpu's native allocation.
    oidn: Buffer,
    buffer: wgpu::Buffer,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    raw: ash::vk::Buffer,
}
impl Shared {
    fn new(r: &Renderer, device: &Arc<Device>, bytes: u64) -> Result<Self> {
        ensure!(bytes <= u64::from(r.device.limits().max_storage_buffer_binding_size)
            && bytes <= r.device.limits().max_buffer_size, "{}", crate::gpu_memory::limit_message("OIDN workspace", format_args!("{bytes} bytes"), format_args!("storage binding={} bytes, buffer={} bytes",r.device.limits().max_storage_buffer_binding_size,r.device.limits().max_buffer_size), "reduce resolution or disable denoising with --set denoise=false"));
        let desc = wgpu::BufferDescriptor {label: Some("OIDN shared images"), size: bytes,
            usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false};
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            use ash::vk;
            let hal = unsafe { r.device.as_hal::<wgpu::hal::api::Vulkan>() }.context("OIDN needs Vulkan")?;
            let raw = hal.raw_device();
            let instance = hal.shared_instance().raw_instance();
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(EXTERNAL_HANDLE);
            let info = vk::BufferCreateInfo::default().size(bytes).usage(STORAGE_USAGE)
                .sharing_mode(vk::SharingMode::EXCLUSIVE).push_next(&mut external);
            let buffer = unsafe { raw.create_buffer(&info, None) }?;
            let requirements = unsafe { raw.get_buffer_memory_requirements(buffer) };
            let memory_properties = unsafe { instance.get_physical_device_memory_properties(hal.raw_physical_device()) };
            let memory_type = (0..memory_properties.memory_type_count).find(|&i|
                requirements.memory_type_bits & (1 << i) != 0 && memory_properties.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL));
            let Some(memory_type) = memory_type else {
                unsafe { raw.destroy_buffer(buffer, None); }
                anyhow::bail!("No device-local memory for OIDN");
            };
            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(EXTERNAL_HANDLE);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
            let allocation = vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(memory_type)
                .push_next(&mut export).push_next(&mut dedicated);
            let memory = match unsafe { raw.allocate_memory(&allocation, None) } {
                Ok(memory) => memory,
                Err(error) => { unsafe { raw.destroy_buffer(buffer, None); } return Err(error.into()); }
            };
            let imported = (|| -> Result<Buffer> {
                unsafe { raw.bind_buffer_memory(buffer, memory, 0) }?;
                #[cfg(target_os = "linux")]
                let handle = {
                    use std::os::fd::{FromRawFd, IntoRawFd, AsRawFd, OwnedFd};
                    let extension = ash::khr::external_memory_fd::Device::new(instance, raw);
                    let info = vk::MemoryGetFdInfoKHR::default().memory(memory).handle_type(EXTERNAL_HANDLE);
                    let fd = unsafe { OwnedFd::from_raw_fd(extension.get_memory_fd(&info)?) };
                    let handle = unsafe { (device.api.shared_fd)(device.handle as Handle, OIDN_EXTERNAL_TYPE | (1 << 30), fd.as_raw_fd(), requirements.size as usize) };
                    // OIDN consumes the fd only on success, including when a later
                    // error query fails. OwnedFd handles every failed import.
                    if !handle.is_null() { let _ = fd.into_raw_fd(); }
                    handle
                };
                #[cfg(target_os = "windows")]
                let handle = {
                    let extension = ash::khr::external_memory_win32::Device::new(instance, raw);
                    let info = vk::MemoryGetWin32HandleInfoKHR::default().memory(memory).handle_type(EXTERNAL_HANDLE);
                    let native = unsafe { extension.get_memory_win32_handle(&info)? } as Handle;
                    let handle = unsafe { (device.api.shared_win32)(device.handle as Handle, OIDN_EXTERNAL_TYPE | (1 << 30), native, std::ptr::null(), requirements.size as usize) };
                    // Unlike an fd, CUDA keeps its own reference; the exported handle stays ours.
                    unsafe { CloseHandle(native); }
                    handle
                };
                let imported = Buffer {engine:device.clone(),handle:handle as usize,bytes:bytes as usize};
                device.check()?;
                ensure!(!handle.is_null(), "OIDN returned null imported buffer");
                Ok(imported)
            })();
            let oidn = match imported {
                Ok(handle) => handle,
                Err(error) => { unsafe { raw.destroy_buffer(buffer, None); raw.free_memory(memory, None); } return Err(error); }
            };
            let native = unsafe { wgpu::hal::vulkan::Buffer::from_raw_managed(buffer, memory, 0, requirements.size) };
            let owned = unsafe { r.device.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(native, &desc) };
            Ok(Self { oidn, buffer: owned, raw: buffer })
        }
        #[cfg(target_os = "macos")]
        {
            use objc2_metal::{MTLDevice, MTLResourceOptions};
            let hal = unsafe { r.device.as_hal::<wgpu::hal::api::Metal>() }.context("OIDN needs Metal")?;
            let native = hal.raw_device().newBufferWithLength_options(bytes as usize,
                MTLResourceOptions::StorageModePrivate | MTLResourceOptions::HazardTrackingModeTracked).ok_or_else(|| OutOfMemory("OIDN Metal allocation".into()))?;
            let handle = unsafe { (device.api.shared_metal)(device.handle as Handle, &*native as *const _ as *mut c_void) };
            let oidn = Buffer {engine: device.clone(), handle: handle as usize, bytes: bytes as usize};
            device.check()?;
            ensure!(!handle.is_null(), "OIDN returned null Metal buffer");
            let native = unsafe { wgpu::hal::metal::Device::buffer_from_raw(native, bytes) };
            let buffer = unsafe { r.device.create_buffer_from_hal::<wgpu::hal::api::Metal>(native, &desc) };
            Ok(Self {oidn, buffer})
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        anyhow::bail!("No OIDN memory sharing backend")
    }
    fn ownership(&self, r: &Renderer, encoder: &mut wgpu::CommandEncoder, release: bool) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            use ash::vk;
            let hal = unsafe { r.device.as_hal::<wgpu::hal::api::Vulkan>() }.unwrap();
            let family = hal.queue_family_index();
            let access = vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
            let barrier = vk::BufferMemoryBarrier::default().buffer(self.raw).offset(0).size(vk::WHOLE_SIZE)
                .src_queue_family_index(if release {family} else {vk::QUEUE_FAMILY_EXTERNAL})
                .dst_queue_family_index(if release {vk::QUEUE_FAMILY_EXTERNAL} else {family})
                .src_access_mask(if release {access} else {vk::AccessFlags::empty()})
                .dst_access_mask(if release {vk::AccessFlags::empty()} else {access});
            unsafe { encoder.as_hal_mut::<wgpu::hal::api::Vulkan,_,_>(|native| {
                hal.raw_device().cmd_pipeline_barrier(native.unwrap().raw_handle(),
                    vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(), &[], &[barrier], &[]);
            }); }
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        let _ = (r, encoder, release); // Tracked Metal resources; completion waits order queues.
    }
}

pub(super) struct Workspace {
    // Filters refer to shared storage; drop them first.
    filters: Vec<Filter>,
    shared: Shared,
    size: [u32;2],
    guides: bool,
}
impl Workspace {
    fn new(r: &Renderer, device: &Arc<Device>, size: [u32;2], guides: bool) -> Result<Self> {
        let bytes = size[0] as u64 * size[1] as u64 * 16;
        let shared = Shared::new(r, device, bytes * 3)?;
        let mut filters = Vec::new();
        let image = |filter: &Filter, name: &CStr, plane: usize| unsafe {
            (device.api.image)(filter.handle, name.as_ptr(), shared.oidn.handle as Handle,
                3, size[0] as usize, size[1] as usize, plane * bytes as usize, 16, size[0] as usize * 16);
        };
        if guides {
            for (name, plane) in [(c"albedo",1), (c"normal",2)] {
                let filter = device.filter()?;
                image(&filter, name, plane); image(&filter, c"output", plane);
                filters.push(filter);
            }
        }
        let filter = device.filter()?;
        image(&filter, c"color",0); image(&filter, c"output",0);
        if guides { image(&filter,c"albedo",1); image(&filter,c"normal",2); }
        unsafe {
            (device.api.boolean)(filter.handle,c"hdr".as_ptr(),true);
            (device.api.boolean)(filter.handle,c"cleanAux".as_ptr(),true);
            (device.api.float)(filter.handle,c"inputScale".as_ptr(),1.);
        }
        filters.push(filter);
        for filter in &filters {
            unsafe {
                (device.api.integer)(filter.handle,c"quality".as_ptr(),OIDN_QUALITY_HIGH);
                (device.api.commit_filter)(filter.handle);
            }
            device.check()?;
        }
        Ok(Self {filters, shared, size, guides})
    }
}

pub(crate) fn entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..5).map(|binding| wgpu::BindGroupLayoutEntry {
            binding, visibility: wgpu::ShaderStages::COMPUTE, count: None,
            ty: wgpu::BindingType::Buffer {ty: if binding==0 {wgpu::BufferBindingType::Uniform}
                else {wgpu::BufferBindingType::Storage {read_only: binding==2}}, has_dynamic_offset:false, min_binding_size:None},
        }).collect()
}

pub(super) struct Stage {
    layout: wgpu::BindGroupLayout,
    pack: wgpu::ComputePipeline,
    unpack: wgpu::ComputePipeline,
}
impl Stage {
    pub(super) fn new(r: &Renderer) -> Self {
        let entries = entries();
        let layout = r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("OIDN preparation"),entries:&entries});
        let pipeline_layout = r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:Some("OIDN preparation"), bind_group_layouts:&[Some(&layout)], immediate_size:0});
        let pipeline = |compiled: &crate::native::Compiled| {
            let module=crate::native::shader_module(&r.device, compiled);
            r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label:Some(compiled.name),layout:Some(&pipeline_layout),module:&module,
                entry_point:Some(compiled.entry),compilation_options:Default::default(),cache:r.cache.as_ref()})
        };
        let pack=pipeline(&crate::native::DENOISE_PACK);
        let unpack=pipeline(&crate::native::DENOISE_UNPACK);
        Self {layout,pack,unpack}
    }
    pub(super) fn run(&self, stage: &super::Stage, r: &Renderer, encoder: &mut wgpu::CommandEncoder,
        samples: usize, input_scale: f32, timing: &mut Timing, segments: &mut crate::command_finish::Segments) -> Result<()> {
        use wgpu::util::DeviceExt;
        let mut engine=r.denoiser.lock().map_err(|_| anyhow::anyhow!("OIDN engine lock poisoned"))?;
        let engine=engine.as_mut().context("OIDN runtime not initialized")?;
        // Drain before replacing storage: a preceding view may still consume it.
        // Same-size frames keep their filters and model scratch allocations.
        let rebuild=engine.gpu_workspace.as_ref().is_none_or(|w| w.size!=stage.size || w.guides!=stage.use_guides);
        if rebuild {
            let start=std::time::Instant::now();
            let mut retired_ms=0.;
            let mut retire = || -> Result<()> {
                let retired=std::time::Instant::now();
                let (submission,finish)=segments.interop(r,encoder,"OIDN workspace allocation");
                timing.finish_ms+=finish;
                let wait=std::time::Instant::now();
                r.wait(submission)?;
                timing.wait_ms+=wait.elapsed().as_secs_f64()*1000.;
                retired_ms+=retired.elapsed().as_secs_f64()*1000.;
                Ok(())
            };
            if engine.gpu_workspace.is_some() { retire()?; }
            engine.gpu_workspace=None;
            let mut first=true;
            engine.gpu_workspace=Some(r.residency.retry(
                || {
                    // Evicted wgpu handles enqueue native destruction. Reap
                    // those releases before retrying OIDN's external allocator.
                    if !first { let _=r.device.poll(wgpu::PollType::Poll); }
                    first=false;
                    Workspace::new(r,&engine.device,stage.size,stage.use_guides)
                },
                |error| {
                    if error.is::<OutOfMemory>() { return true; }
                    #[cfg(any(target_os = "linux", target_os = "windows"))]
                    if error.downcast_ref::<ash::vk::Result>().is_some_and(|e|
                        matches!(*e, ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY | ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY)) { return true; }
                    false
                },
                &mut retire,
            )?);
            let setup_ms=(start.elapsed().as_secs_f64()*1000.-retired_ms).max(0.);
            timing.filter_ms+=setup_ms;
            crate::timing::creation(&r.device,"OIDN GPU filters","denoiser workspace",0.,setup_ms);
        }
        let workspace=engine.gpu_workspace.as_ref().unwrap();
        let count=stage.size[0]*stage.size[1];
        for channel in 0..if stage.spectral {3} else {1} {
        let params=r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {label:Some("OIDN preparation parameters"),
            contents:bytemuck::cast_slice(&[count,samples as u32,stage.guide_alpha as u32,stage.size[0],stage.spectral as u32,channel,input_scale.to_bits(),0]),usage:wgpu::BufferUsages::UNIFORM});
        let buffers=[&params,&stage.accumulation,&stage.guides,&workspace.shared.buffer,&stage.output];
        let entries:Vec<_>=buffers.iter().enumerate().map(|(i,b)|wgpu::BindGroupEntry {binding:i as u32,resource:b.as_entire_binding()}).collect();
        let bindings=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:Some("OIDN preparation"),layout:&self.layout,entries:&entries});
        let dispatch=|encoder:&mut wgpu::CommandEncoder,pipeline:&wgpu::ComputePipeline| {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("OIDN preparation"),timestamp_writes:None});
            pass.set_pipeline(pipeline);pass.set_bind_group(0,&bindings,&[]);
            pass.dispatch_workgroups(stage.size[0].div_ceil(vork_slang::registry::DENOISE_GROUP),stage.size[1],1);
        };
        let start=std::time::Instant::now();
        #[cfg(test)] if channel==0 {
            let color_bytes = u64::from(count)*16;
            encoder.copy_buffer_to_buffer(&stage.accumulation,0,stage.staging.as_ref().unwrap(),0,color_bytes);
            encoder.copy_buffer_to_buffer(&stage.guides,0,stage.staging.as_ref().unwrap(),color_bytes,u64::from(count)*32);
        }
        dispatch(encoder,&self.pack);
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            // wgpu's raw encoding mode cannot share an encoder with validated
            // compute passes. Keep the ownership transfer in its own ordered
            // segment, after the pack pass and its resource transitions.
            segments.push(std::mem::replace(encoder,r.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor {label:Some("OIDN release ownership")})));
        }
        workspace.shared.ownership(r,encoder,true);
        let (submission,finish)=segments.interop(r,encoder,"after GPU OIDN");
        timing.finish_ms+=finish;
        r.wait(submission)?;
        timing.wait_ms+=start.elapsed().as_secs_f64()*1000.;
        let start=std::time::Instant::now();
        for (index,filter) in workspace.filters.iter().enumerate() {
            // Normal filtering is independent of tristimulus channel; retain
            // its first result while albedo and beauty are repacked.
            if channel>0 && stage.use_guides && index==1 {continue;}
            unsafe { (engine.device.api.execute)(filter.handle); }
            engine.device.check()?;
        }
        timing.filter_ms+=start.elapsed().as_secs_f64()*1000.;
        // Synchronous OIDN execution completes writes before Vulkan reacquires
        // ownership or Metal submits the tracked read on the renderer's queue.
        workspace.shared.ownership(r,encoder,false);
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            // wgpu can prepend buffer-state barriers to a command buffer at
            // submission. Acquire in a separate, resource-free submission so
            // none of those barriers touch CUDA-owned memory before acquisition.
            // Waiting also retires this raw command's untracked buffer reference
            // on error paths before the workspace can be destroyed.
            let start=std::time::Instant::now();
            let (submission,finish)=segments.interop(r,encoder,"OIDN ownership acquired");
            timing.finish_ms+=finish;
            r.wait(submission)?;
            timing.wait_ms+=start.elapsed().as_secs_f64()*1000.;
        }
        dispatch(encoder,&self.unpack);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires real CUDA/Metal GPU, Slang and pinned OIDN"]
    fn shared_images_match_gpu_host_reference_across_reuse_and_resize() -> Result<()> {
        use wgpu::util::DeviceExt;
        let r=Renderer::new()?;
        // Exercise non-workgroup widths, alpha, background, no guides, retained
        // filters, resolution changes, and repeated handoffs of shared memory.
        // Force the host-staging fallback too; its pre-exposure must match the
        // same independently filtered working-unit reference.
        for cpu in [false, true] {
        if cpu { *r.denoiser.lock().unwrap() = Some(Engine::new()?); }
        for (size,guided,spectral,scale) in [([73,31],true,false,1_f32),([73,31],true,true,131072.),([19,41],false,true,0.03125),([73,31],true,false,131072.)] {
            let count=(size[0]*size[1]) as usize;
            let samples=4;
            let associated:Vec<[f32;4]>=(0..count).map(|i| {
                let alpha=if i%13==0 {0.} else {0.5};
                let rgb=if spectral {[-0.25,0.5,0.125]} else {[(i%17) as f32/17.,0.4,0.8]};
                [alpha*rgb[0]/scale,alpha*rgb[1]/scale,alpha*rgb[2]/scale,alpha].map(|x| x*samples as f32)
            }).collect();
            let guides:Vec<[f32;4]>=(0..count).flat_map(|i| {
                let a=if i%7==0 {0.} else {2.};
                [[0.3*a,0.2*a,0.1*a,a],[0.6*a,0.,0.2*a,0.]]
            }).collect();
            let make=|data:&[[f32;4]]|r.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label:Some("OIDN interop test"),contents:bytemuck::cast_slice(data),usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST});
            let accumulation=make(&associated);
            let guide_buffer=make(&guides);
            let output=make(&vec![[0.;4];count]);
            let mut stage=super::super::Stage::new(&r,&Denoiser::default(),&accumulation,&output,&guide_buffer,true,guided,size)?;
            stage.spectral=spectral;
            let device=r.denoiser.lock().unwrap().as_ref().unwrap().device.clone();
            ensure!(device.gpu!=cpu,"test must exercise the requested OIDN backend");
            let mut bytes=bytemuck::cast_slice(&associated).to_vec();
            bytes.extend_from_slice(bytemuck::cast_slice(&guides));
            let Inputs {mut color,mut albedo,normals,..}=unpack_inputs(&bytes,samples,count,true);
            let mut reference=Engine {gpu_workspace:None,device,color:None,output:None,albedo:None,normal:None};
            if spectral {
                for rgb in color.iter_mut().chain(albedo.iter_mut()) { *rgb=spectral_color(*rgb,false); }
            }
            for rgb in &mut color {for c in rgb {*c*=scale;}}
            if spectral {reference.denoise_spectral(&Denoiser::default(),size,&mut color,&albedo,&normals,guided)?;}
            else {reference.denoise(&Denoiser::default(),size,&mut color,&albedo,&normals,guided)?;}
            for rgb in &mut color {for c in rgb.iter_mut() {*c/=scale;}}
            if spectral {for rgb in &mut color { *rgb=spectral_color(*rgb,true); }}
            let mut encoder=r.device.create_command_encoder(&Default::default());
            stage.run(&r,&mut encoder,samples,scale,&mut Timing::default(),&mut crate::command_finish::Segments::default())?;
            let readback=r.device.create_buffer(&wgpu::BufferDescriptor {label:None,size:(count*16) as u64,
                usage:wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ,mapped_at_creation:false});
            encoder.copy_buffer_to_buffer(&output,0,&readback,0,(count*16) as u64);
            let submission=r.queue.submit([encoder.finish()]);
            let (tx,rx)=std::sync::mpsc::channel();
            readback.map_async(wgpu::MapMode::Read,..,move |v| {let _=tx.send(v);});
            r.wait(Some(submission))?;rx.recv()??;
            let data=readback.get_mapped_range(..);
            for (i,pixel) in data.chunks_exact(16).enumerate() {
                let actual:[f32;4]=std::array::from_fn(|c| f32::from_ne_bytes(pixel[c*4..c*4+4].try_into().unwrap()));
                let mut expected=associated[i].map(|x|x/samples as f32);
                if guides[2*i][3]>0. {for c in 0..3 {expected[c]=color[i][c]*expected[3];}}
                for c in 0..4 {assert!((actual[c]-expected[c]).abs()*(if c<3 {scale} else {1.})<2e-4,"{size:?} guided={guided} pixel {i}: {actual:?} != {expected:?}");}
            }
            drop(data);readback.unmap();
        }
        }
        Ok(())
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod backend_tests {
    use super::*;

    #[test]
    fn cuda_remains_the_preferred_backend() {
        assert_eq!(GPU_BACKENDS.first().map(|(kind, ..)| *kind), Some(3),
            "NVIDIA hosts must keep selecting CUDA before any other backend");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hip_backend_is_registered_on_linux() {
        assert!(GPU_BACKENDS.iter().any(|&(kind, ..)| kind == 4),
            "AMD hosts need OIDN's HIP device type");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn pci_bdf_formats_the_device_address() {
        assert_eq!(pci_bdf([0, 3, 0, 0]), "0000:03:00.0");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn unset_pci_address_is_not_a_device() {
        assert_eq!(set_pci([0, 0, 0, 0]), None,
            "an unset PCI address must not resolve to 0000:00:00.0");
        assert_eq!(set_pci([0, 3, 0, 0]), Some([0, 3, 0, 0]));
    }
}
