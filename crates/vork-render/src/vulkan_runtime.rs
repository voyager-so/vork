//! Discover container-mounted NVIDIA ICD entry points without mutating loader
//! environment variables or host manifests. The Vulkan loader still owns dispatch.
use anyhow::{Context, Result};
use ash::vk;
#[cfg(target_os = "linux")]
use std::sync::OnceLock;

pub(crate) fn instance(desc: wgpu::InstanceDescriptor) -> Result<wgpu::Instance> {
    // GPU-free commands never call this. Report the loader separately from its ICDs.
    let entry = unsafe { ash::Entry::load() }.context(if cfg!(target_os = "windows") {
        "Vulkan loader unavailable (vulkan-1.dll). Install your GPU vendor graphics driver including its Vulkan runtime; do not download individual DLLs"
    } else {
        "Vulkan loader unavailable (libvulkan.so.1). Install libvulkan1 (Debian/Ubuntu), vulkan-loader (Fedora), or vulkan-icd-loader (Arch)"
    })?;
    #[cfg(target_os = "windows")]
    { drop(entry); Ok(wgpu::Instance::new(desc)) }
    #[cfg(target_os = "linux")]
    nvidia_instance(desc, entry)
}

#[cfg(target_os = "linux")]
fn nvidia_instance(desc: wgpu::InstanceDescriptor, entry: ash::Entry) -> Result<wgpu::Instance> {
    // Respect explicit operator/CI selection before any optional probe work.
    let selected = ["VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES",
        "VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE"]
        .iter().any(|name| std::env::var_os(name).is_some());
    discover(selected, || {
        let extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .context("Vulkan loader could not enumerate instance extensions for direct driver loading")?;
        Ok(extensions.iter().any(|e| e.extension_name_as_c_str()
            .is_ok_and(|name| name == ash::lunarg::direct_driver_loading::NAME)))
    }, |name| {
        // Keep both libraries resident: an initialized instance can retain ICD
        // pointers even when adapter admission fails and we try the next path.
        static EGL: OnceLock<libloading::Library> = OnceLock::new();
        static GLX: OnceLock<libloading::Library> = OnceLock::new();
        let slot = if name == "libGLX_nvidia.so.0" { &GLX } else { &EGL };
        if slot.get().is_none() {
            let library = unsafe { libloading::Library::new(name) }
                .with_context(|| format!("could not load {name}"))?;
            let _ = slot.set(library);
        }
        let library = slot.get().unwrap();
        let get_proc = *unsafe { library.get::<vk::PFN_vkGetInstanceProcAddr>(b"vk_icdGetInstanceProcAddr\0") }
            .context("NVIDIA ICD does not export vk_icdGetInstanceProcAddr")?;
        nvidia_candidate(&desc, get_proc)
    }, || initialized_instance(&desc, None))
}

#[cfg(target_os = "linux")]
fn nvidia_candidate(desc: &wgpu::InstanceDescriptor, get_proc: vk::PFN_vkGetInstanceProcAddr) -> Result<wgpu::Instance> {
    // ash's fallback for a missing required entry point panics. A malformed
    // optional ICD must instead fail discovery and allow the other paths.
    anyhow::ensure!(unsafe { get_proc(vk::Instance::null(), c"vkEnumerateInstanceExtensionProperties".as_ptr()) }.is_some(),
        "NVIDIA ICD is missing vkEnumerateInstanceExtensionProperties");
    // Loader discovery happens before the direct ICD is added. In an image
    // without any normal ICD, that initial extension list omits the driver's
    // properties2 extension, which wgpu needs to query descriptor-array features.
    let driver_entry = unsafe { ash::Entry::from_static_fn(ash::StaticFn { get_instance_proc_addr: get_proc }) };
    let driver_extensions = unsafe { driver_entry.enumerate_instance_extension_properties(None) }
        .context("NVIDIA ICD could not enumerate its instance extensions")?;
    let properties2 = driver_extensions.iter().any(|e| e.extension_name_as_c_str()
        .is_ok_and(|name| name == ash::khr::get_physical_device_properties2::NAME));
    let drivers = [vk::DirectDriverLoadingInfoLUNARG::default()
        .pfn_get_instance_proc_addr(Some(get_proc))];
    let mut loading = vk::DirectDriverLoadingListLUNARG::default()
        .mode(vk::DirectDriverLoadingModeLUNARG::INCLUSIVE).drivers(&drivers);
    let loading = &mut loading;
    let callback: Box<wgpu::hal::vulkan::CreateInstanceCallback<'_>> = Box::new(move |args| {
        args.extensions.push(ash::lunarg::direct_driver_loading::NAME);
        if properties2 && !args.extensions.contains(&ash::khr::get_physical_device_properties2::NAME) {
            args.extensions.push(ash::khr::get_physical_device_properties2::NAME);
        }
        *args.create_info = (*args.create_info).push_next(loading);
    });
    initialized_instance(desc, Some(callback))
}

#[cfg(target_os = "linux")]
fn initialized_instance(
    desc: &wgpu::InstanceDescriptor,
    callback: Option<Box<wgpu::hal::vulkan::CreateInstanceCallback<'_>>>,
) -> Result<wgpu::Instance> {
    let hal_desc = wgpu::hal::InstanceDescriptor {
        name: "vork", flags: desc.flags, memory_budget_thresholds: desc.memory_budget_thresholds,
        backend_options: desc.backend_options.clone(), telemetry: None, display: None,
    };
    // Use the fallible HAL initializer for ordinary discovery too: Instance::new
    // discards the native backend initialization errors needed for diagnosis.
    let hal = unsafe { wgpu::hal::vulkan::Instance::init_with_callback(&hal_desc, callback) }
        .context("Vulkan loader could not initialize the driver")?;
    // from_hal resets core flags to defaults, which re-enables indirect-command
    // validation in release builds. Preserve the same policy as the native HAL.
    let mut core = unsafe { wgpu::wgc::instance::Instance::from_hal_instance::<wgpu::hal::api::Vulkan>("vork".into(), hal) };
    core.flags = desc.flags;
    let instance = unsafe { wgpu::Instance::from_core(core) };
    // Do not publish/cache an instance until the same adapter request used by
    // Renderer succeeds. An ICD can initialize yet expose no usable adapter.
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    })).map_err(|error| {
        let diagnostic = adapter_failure(&instance, &error);
        anyhow::Error::new(error).context(diagnostic)
    })?;
    Ok(instance)
}

pub(crate) fn adapter_failure(instance: &wgpu::Instance, error: &impl std::fmt::Display) -> String {
    let driver = if cfg!(target_os = "windows") {
        "Vulkan loader is available, but no usable driver/device was exposed. Install the vendor Vulkan graphics driver and check device access"
    } else {
        "Vulkan loader is available, but no usable driver/device was exposed. Install the vendor Vulkan ICD and check device access. NVIDIA containers need graphics driver libraries mounted by NVIDIA Container Toolkit, not only compute/utility access"
    };
    let Some(hal) = (unsafe { instance.as_hal::<wgpu::hal::api::Vulkan>() }) else {
        return format!("{driver}; native adapter error: {error}");
    };
    let raw = hal.shared_instance().raw_instance();
    match unsafe { raw.enumerate_physical_devices() } {
        Ok(devices) if !devices.is_empty() => {
            let details: Vec<_> = devices.into_iter().map(|device| {
                let properties = unsafe { raw.get_physical_device_properties(device) };
                let name = properties.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "unnamed Vulkan device".into());
                let graphics = unsafe { raw.get_physical_device_queue_family_properties(device) }
                    .iter().any(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::GRAPHICS));
                format!("{name} (graphics queue: {graphics})")
            }).collect();
            format!("GPU driver enumerated {}, but none could be opened as a graphics adapter. Compute-only hardware cannot gain graphics/ray capabilities from container flags; use supported graphics hardware such as an NVIDIA L4. Native adapter error: {error}", details.join(", "))
        }
        Ok(_) => format!("{driver}; native adapter error: {error}"),
        Err(native) => format!("Vulkan driver failed to enumerate physical devices: {native}; native adapter error: {error}. Check the vendor driver installation and device access"),
    }
}

// Follow NVIDIA's desktop ICD choice: EGL is the fallback when X11 client
// libraries (and thus GLX) are unavailable. EGL can expose an adapter yet crash
// in Xlib surface queries, so adapter admission alone cannot prefer it to GLX.
// Discovery candidates must succeed through adapter admission, not just dlopen.
#[cfg(target_os = "linux")]
const NVIDIA_DRIVERS: [&str; 2] = ["libGLX_nvidia.so.0", "libEGL_nvidia.so.0"];

#[cfg(target_os = "linux")]
fn discover<T>(
    selected: bool,
    probe: impl FnOnce() -> Result<bool>,
    mut candidate: impl FnMut(&'static str) -> Result<T>,
    ordinary: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if selected { return ordinary(); }
    let mut failures = Vec::new();
    match probe() {
        Ok(true) => for name in NVIDIA_DRIVERS {
            match candidate(name) {
                Ok(value) => return Ok(value),
                Err(error) => failures.push((name, error)),
            }
        },
        Ok(false) => {},
        Err(error) => failures.push(("direct driver loading discovery", error)),
    }
    ordinary().map_err(|error| {
        if failures.is_empty() { return error; }
        let diagnostic = DiscoveryFailure { probes: failures, ordinary: format!("{error:#}") };
        error.context(diagnostic)
    })
}

// Context preserves ordinary discovery's native type for callers' downcasts,
// while retaining every optional native error and displaying all cause chains.
#[cfg(target_os = "linux")]
#[derive(Debug)]
struct DiscoveryFailure {
    probes: Vec<(&'static str, anyhow::Error)>,
    ordinary: String,
}

#[cfg(target_os = "linux")]
impl std::fmt::Display for DiscoveryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "No usable Vulkan driver/device")?;
        for (name, error) in &self.probes { write!(f, "; {name}: {error:#}")?; }
        write!(f, "; ordinary Vulkan discovery: {}", self.ordinary)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Vulkan; set VORK_GPU_TESTS=1"]
    fn imported_instance_preserves_indirect_validation_policy() -> Result<()> {
        anyhow::ensure!(std::env::var("VORK_GPU_TESTS").as_deref() == Ok("1"),
            "GPU test requires VORK_GPU_TESTS=1 and a working Vulkan adapter");
        for enabled in [false, true] {
            let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
            desc.flags = wgpu::InstanceFlags::DEBUG;
            desc.flags.set(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL, enabled);
            let instance = initialized_instance(&desc, None)?;
            let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
            let (device, _queue) = pollster::block_on(adapter.request_device(&Default::default()))?;
            let report = device.generate_allocator_report().context("Vulkan allocation report")?;
            let validation = report.allocations.iter().any(|a|
                a.name.contains("Indirect dispatch validation destination buffer"));
            assert_eq!(validation, enabled, "HAL import must retain the requested validation policy");
        }
        Ok(())
    }

    #[test]
    fn broken_optional_driver_does_not_block_ordinary_driver() {
        for stage in ["library load", "symbol lookup", "extension enumeration", "instance initialization", "adapter request"] {
            let mut tried = Vec::new();
            let result = discover(false, || Ok(true), |name| {
                tried.push(name);
                Err(anyhow::Error::new(vk::Result::ERROR_INITIALIZATION_FAILED).context(stage))
            }, || Ok("ordinary adapter"));
            assert_eq!(result.unwrap(), "ordinary adapter", "{stage}");
            assert_eq!(tried, NVIDIA_DRIVERS);
        }
    }

    #[test]
    fn egl_is_tried_after_glx_fails() {
        for stage in ["library load", "symbol lookup", "extension enumeration", "instance initialization", "adapter request"] {
            let mut tried = Vec::new();
            let result = discover(false, || Ok(true), |name| {
                tried.push(name);
                if name == NVIDIA_DRIVERS[0] { anyhow::bail!(stage); }
                Ok("headless EGL adapter")
            }, || panic!("working EGL fallback must win"));
            assert_eq!(result.unwrap(), "headless EGL adapter");
            assert_eq!(tried, NVIDIA_DRIVERS);
        }
    }

    #[test]
    fn desktop_icd_is_preferred_even_when_egl_can_render() {
        let result = discover(false, || Ok(true), |name| {
            assert_eq!(name, "libGLX_nvidia.so.0");
            Ok("GLX adapter")
        }, || panic!("working GLX must win"));
        assert_eq!(result.unwrap(), "GLX adapter");
    }

    #[test]
    fn all_failed_paths_keep_native_causes() {
        let error = discover::<()>(false, || Ok(true), |name| {
            Err(anyhow::Error::new(vk::Result::ERROR_INITIALIZATION_FAILED).context(name))
        }, || Err(anyhow::Error::new(vk::Result::ERROR_INCOMPATIBLE_DRIVER).context("ordinary init"))).unwrap_err();
        let diagnostic = error.to_string();
        for text in NVIDIA_DRIVERS.into_iter().chain(["ordinary init", "Initialization of an object", "Unable to find a Vulkan driver"]) {
            assert!(diagnostic.contains(text), "missing {text}: {diagnostic}");
        }
        let failures = &error.downcast_ref::<DiscoveryFailure>().unwrap().probes;
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].1.downcast_ref::<vk::Result>(), Some(&vk::Result::ERROR_INITIALIZATION_FAILED));
        assert_eq!(failures[1].1.downcast_ref::<vk::Result>(), Some(&vk::Result::ERROR_INITIALIZATION_FAILED));
        assert_eq!(error.downcast_ref::<vk::Result>(), Some(&vk::Result::ERROR_INCOMPATIBLE_DRIVER));
    }

    #[test]
    fn explicit_selection_never_probes_or_falls_back() {
        for works in [false, true] {
            let result = discover(true, || panic!("explicit selection must bypass probe"),
                |_| panic!("explicit selection must bypass candidates"),
                || if works { Ok(()) } else { anyhow::bail!("selected driver failed") });
            assert_eq!(result.is_ok(), works);
            if let Err(error) = result { assert_eq!(error.to_string(), "selected driver failed"); }
        }
    }

    #[test]
    fn unavailable_direct_loading_uses_ordinary_discovery() {
        assert_eq!(discover(false, || Ok(false), |_| panic!("unsupported extension"), || Ok(42)).unwrap(), 42);
        assert_eq!(discover(false, || anyhow::bail!("extension enumeration failed"), |_| panic!("failed probe"), || Ok(42)).unwrap(), 42);
        let error = discover::<()>(false, || anyhow::bail!("extension enumeration failed"),
            |_| panic!("failed probe"), || anyhow::bail!("ordinary failed")).unwrap_err();
        assert!(error.to_string().contains("extension enumeration failed"));
        assert!(error.to_string().contains("ordinary failed"));
    }

    #[test]
    fn native_icd_enumeration_failure_falls_back() {
        unsafe extern "system" fn enumerate(_: *const std::ffi::c_char, _: *mut u32, _: *mut vk::ExtensionProperties) -> vk::Result {
            vk::Result::ERROR_INITIALIZATION_FAILED
        }
        unsafe extern "system" fn get_proc(_: vk::Instance, name: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction {
            if unsafe { std::ffi::CStr::from_ptr(name) } == c"vkEnumerateInstanceExtensionProperties" {
                Some(unsafe { std::mem::transmute::<vk::PFN_vkEnumerateInstanceExtensionProperties, unsafe extern "system" fn()>(enumerate) })
            } else { None }
        }
        let desc = wgpu::InstanceDescriptor::new_without_display_handle();
        let mut attempts = 0;
        let result = discover(false, || Ok(true), |_| {
            attempts += 1;
            let error = nvidia_candidate(&desc, get_proc).err().unwrap();
            assert_eq!(error.downcast_ref::<vk::Result>(), Some(&vk::Result::ERROR_INITIALIZATION_FAILED));
            Err(error)
        }, || Ok("ordinary adapter"));
        assert_eq!(result.unwrap(), "ordinary adapter");
        assert_eq!(attempts, 2);
    }

    #[test]
    fn missing_required_icd_entry_point_is_an_error() {
        unsafe extern "system" fn missing(_: vk::Instance, _: *const std::ffi::c_char) -> vk::PFN_vkVoidFunction { None }
        let error = nvidia_candidate(&wgpu::InstanceDescriptor::new_without_display_handle(), missing).err().unwrap();
        assert!(error.to_string().contains("vkEnumerateInstanceExtensionProperties"));
    }
}
