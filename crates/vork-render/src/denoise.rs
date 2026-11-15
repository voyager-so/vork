//! Optional OIDN boundary. All synchronization, host staging and native calls
//! live here; a null denoiser installs none of these resources.
use crate::*;
use std::{
    ffi::{CStr, c_char, c_void},
    path::PathBuf,
    sync::Arc,
};
use vork_scene::v2::Denoiser;
type Handle = *mut c_void;
const OIDN_QUALITY_HIGH: i32 = 6;
pub(crate) mod gpu;
pub(crate) mod spatial;

// Inputs are bytes; OIDN's maxMemoryMB uses MiB. Camera budget is headroom at
// renderer creation (including any lowering override), so charge only usage
// growth since that same snapshot. Current native headroom also bounds scratch.
fn memory_limit_mb(budget: u64, usage_at_creation: u64, usage: u64, headroom: u64) -> Result<i32> {
    let growth = usage.saturating_sub(usage_at_creation);
    let mb = budget.saturating_sub(growth).min(headroom) / (1024 * 1024);
    if mb == 0 {
        return Err(OutOfMemory("denoising needs more memory; reduce resolution or use --set denoise=false".into()).into());
    }
    Ok(mb.min(i32::MAX as u64) as i32)
}

// OIDN's HDR input processor clamps negative channels. Spectral Rec.709 has
// negative matching lobes, so filter in a white-preserving XYZ basis and invert
// after reconstruction. Alpha and normal vectors never enter this transform.
fn spectral_color(rgb: [f32; 3], inverse: bool) -> [f32; 3] {
    use vork_slang::registry::*;
    let rows = if inverse {
        [[DENOISE_FROM_XYZ_00,DENOISE_FROM_XYZ_01,DENOISE_FROM_XYZ_02],
         [DENOISE_FROM_XYZ_10,DENOISE_FROM_XYZ_11,DENOISE_FROM_XYZ_12],
         [DENOISE_FROM_XYZ_20,DENOISE_FROM_XYZ_21,DENOISE_FROM_XYZ_22]]
    } else {
        [[DENOISE_TO_XYZ_00,DENOISE_TO_XYZ_01,DENOISE_TO_XYZ_02],
         [DENOISE_TO_XYZ_10,DENOISE_TO_XYZ_11,DENOISE_TO_XYZ_12],
         [DENOISE_TO_XYZ_20,DENOISE_TO_XYZ_21,DENOISE_TO_XYZ_22]]
    };
    rows.map(|row| row[0]*rgb[0]+row[1]*rgb[1]+row[2]*rgb[2])
}

// ABI pinned to OIDN 2.5.1's public C header. Function pointers cannot outlive Library.
macro_rules! api {
    ($($(#[$meta:meta])* $name:ident: $symbol:literal ($($arg:ty),*) -> $ret:ty;)*) => {
        struct Api { $($(#[$meta])* $name: unsafe extern "C" fn($($arg),*) -> $ret,)* _library: libloading::Library }
        impl Api {
            unsafe fn load(path: &std::path::Path) -> Result<Self> {
                #[cfg(target_os="windows")]
                let loaded = unsafe { libloading::os::windows::Library::load_with_flags(path, 0x1100) }.map(Into::into);
                #[cfg(not(target_os="windows"))]
                let loaded = unsafe { libloading::Library::new(path) };
                let library: libloading::Library = loaded.with_context(|| format!("load OIDN runtime {} (set VORK_OIDN to its library path)", path.display()))?;
                Ok(Self { $($(#[$meta])* $name: *unsafe { library.get::<unsafe extern "C" fn($($arg),*) -> $ret>(concat!($symbol, "\0").as_bytes()) }?,)* _library: library })
            }
        }
    }
}
api! {
    new_device: "oidnNewDevice" (i32) -> Handle;
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    physical_count: "oidnGetNumPhysicalDevices" () -> i32;
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    physical_int: "oidnGetPhysicalDeviceInt" (i32, *const c_char) -> i32;
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    physical_data: "oidnGetPhysicalDeviceData" (i32, *const c_char, *mut usize) -> *const c_void;
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    new_by_id: "oidnNewDeviceByID" (i32) -> Handle;
    #[cfg(target_os = "macos")]
    metal_supported: "oidnIsMetalDeviceSupported" (Handle) -> bool;
    #[cfg(target_os = "macos")]
    new_metal: "oidnNewMetalDevice" (*const Handle, i32) -> Handle;
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    device_int: "oidnGetDeviceInt" (Handle, *const c_char) -> i32;
    #[cfg(target_os = "linux")]
    shared_fd: "oidnNewSharedBufferFromFD" (Handle, u32, i32, usize) -> Handle;
    #[cfg(target_os = "windows")]
    shared_win32: "oidnNewSharedBufferFromWin32Handle" (Handle, u32, Handle, *const c_void, usize) -> Handle;
    #[cfg(target_os = "macos")]
    shared_metal: "oidnNewSharedBufferFromMetal" (Handle, Handle) -> Handle;
    commit_device: "oidnCommitDevice" (Handle) -> ();
    release_device: "oidnReleaseDevice" (Handle) -> ();
    error: "oidnGetDeviceError" (Handle, *mut *const c_char) -> i32;
    new_buffer: "oidnNewBuffer" (Handle, usize) -> Handle;
    release_buffer: "oidnReleaseBuffer" (Handle) -> ();
    write: "oidnWriteBuffer" (Handle, usize, usize, *const c_void) -> ();
    read: "oidnReadBuffer" (Handle, usize, usize, *mut c_void) -> ();
    new_filter: "oidnNewFilter" (Handle, *const c_char) -> Handle;
    release_filter: "oidnReleaseFilter" (Handle) -> ();
    image: "oidnSetFilterImage" (Handle, *const c_char, Handle, i32, usize, usize, usize, usize, usize) -> ();
    boolean: "oidnSetFilterBool" (Handle, *const c_char, bool) -> ();
    float: "oidnSetFilterFloat" (Handle, *const c_char, f32) -> ();
    integer: "oidnSetFilterInt" (Handle, *const c_char, i32) -> ();
    commit_filter: "oidnCommitFilter" (Handle) -> ();
    execute: "oidnExecuteFilter" (Handle) -> ();
}
/// The denoiser or its device could not allocate; residency eviction retries on it.
#[derive(Debug)]
pub(crate) struct OutOfMemory(String);
impl std::fmt::Display for OutOfMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for OutOfMemory {}

struct Device {
    api: Api,
    handle: usize,
    gpu: bool,
}
impl Device {
    /// Failing to open OIDN is a setup failure; later OIDN errors are device failures.
    fn new() -> Result<Self> {
        Self::open().map_err(crate::failure::setup)
    }
    fn open() -> Result<Self> {
        Self::open_for(None)
    }
    fn load_api() -> Result<Api> {
        let filename = if cfg!(target_os = "windows") {
            "OpenImageDenoise.dll"
        } else if cfg!(target_os = "macos") {
            "libOpenImageDenoise.2.dylib"
        } else {
            "libOpenImageDenoise.so.2"
        };
        let path = std::env::var_os("VORK_OIDN")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let root = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.canonicalize().ok())
                    .and_then(|p| p.parent().map(ToOwned::to_owned))
                    .unwrap_or_default();
                root.join("oidn")
                    .join(if cfg!(target_os = "windows") {
                        "bin"
                    } else {
                        "lib"
                    })
                    .join(filename)
            });
        unsafe { Api::load(&path) }
    }
    fn open_for(renderer: Option<&Renderer>) -> Result<Self> {
        let api = Self::load_api()?;
        let selected = renderer.and_then(|r| gpu::select(&api, r).ok());
        let device = selected.unwrap_or_else(|| unsafe { (api.new_device)(1) });
        Self::adopt(api, device, selected.is_some())
    }
    fn adopt(api: Api, device: Handle, gpu: bool) -> Result<Self> {
        let engine = Self {
            api,
            handle: device as usize,
            gpu,
        };
        engine.check_untagged()?;
        ensure!(!device.is_null(), "OIDN returned no supported device");
        // GPU selection commits while any caller-owned native queue is alive.
        if !engine.gpu { unsafe { (engine.api.commit_device)(device); } }
        engine.check_untagged()?;
        Ok(engine)
    }
    fn check(&self) -> Result<()> {
        self.check_untagged().context(crate::failure::FailureKind::Device)
    }
    fn check_untagged(&self) -> Result<()> {
        let mut message = std::ptr::null();
        let code = unsafe { (self.api.error)(self.handle as Handle, &mut message) };
        if code != 0 {
            let message = if message.is_null() {
                "unknown OIDN error".into()
            } else {
                unsafe { CStr::from_ptr(message) }.to_string_lossy()
            };
            if code == 4 { return Err(OutOfMemory(format!("OIDN ({code}): {message}")).into()); }
            anyhow::bail!("OIDN ({code}): {message}");
        }
        Ok(())
    }
    fn buffer(self: &Arc<Self>, bytes: usize) -> Result<Buffer> {
        let handle = unsafe { (self.api.new_buffer)(self.handle as Handle, bytes) };
        let buffer = Buffer {engine: self.clone(), handle: handle as usize, bytes};
        self.check()?;
        if handle.is_null() { return Err(anyhow::anyhow!("OIDN returned null buffer").context(crate::failure::FailureKind::Device)); }
        Ok(buffer)
    }
    fn filter(self: &Arc<Self>) -> Result<Filter> {
        let handle = unsafe { (self.api.new_filter)(self.handle as Handle, c"RT".as_ptr()) };
        let filter = Filter {
            engine: self.clone(),
            handle,
        };
        self.check()?;
        if handle.is_null() { return Err(anyhow::anyhow!("OIDN returned null RT filter").context(crate::failure::FailureKind::Device)); }
        Ok(filter)
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe {
                (self.api.release_device)(self.handle as Handle);
            }
        }
    }
}
// One renderer owns the native workspace. Captures reuse it sequentially rather
// than retaining one full-size CUDA allocation set per image view.
pub(crate) struct Engine {
    gpu_workspace: Option<gpu::Workspace>,
    device: Arc<Device>,
    color: Option<Buffer>,
    output: Option<Buffer>,
    albedo: Option<Buffer>,
    normal: Option<Buffer>,
}
impl Engine {
    fn new() -> Result<Self> {
        Self::with_device(Device::new()?)
    }
    fn with_device(device: Device) -> Result<Self> {
        Ok(Self {gpu_workspace: None, device: Arc::new(device), color:None, output:None, albedo:None, normal:None})
    }
    fn reserve<'a>(device: &Arc<Device>, slot: &'a mut Option<Buffer>, bytes: usize) -> Result<&'a Buffer> {
        if slot.as_ref().is_none_or(|buffer| buffer.bytes < bytes) {
            *slot = Some(device.buffer(bytes)?);
        }
        Ok(slot.as_ref().unwrap())
    }
    /// Positive tristimulus channels are scalar signals, not RGB chroma.
    /// Replicate each into neutral RGB so the learned RGB colour prior cannot
    /// mix physically unrelated matching functions. Retain the network's local
    /// reconstruction in the caller's pre-exposed units; no energy gain follows.
    fn denoise_spectral(&mut self, settings: &Denoiser, size: [u32;2], color: &mut [[f32;3]],
        albedo: &[[f32;3]], normals: &[[f32;3]], use_guides: bool) -> Result<()> {
        let mut scalar=vec![[0.;3];color.len()];
        let mut guide=vec![[0.;3];albedo.len()];
        for channel in 0..3 {
            for (p,q) in scalar.iter_mut().zip(color.iter()) {*p=[q[channel];3];}
            for (p,q) in guide.iter_mut().zip(albedo) {*p=[q[channel];3];}
            self.denoise_channel(settings,size,&mut scalar,&guide,normals,use_guides,channel==0)?;
            for (p,q) in color.iter_mut().zip(&scalar) {p[channel]=(q[0]+q[1]+q[2])/3.;}
        }
        Ok(())
    }
    fn denoise(
        &mut self,
        _settings: &Denoiser,
        size: [u32; 2],
        color: &mut [[f32; 3]],
        albedo: &[[f32; 3]],
        normals: &[[f32; 3]],
        use_guides: bool,
    ) -> Result<()> {
        self.denoise_channel(_settings,size,color,albedo,normals,use_guides,true)
    }
    // Reuse only within one spectral image; every new image refreshes normals.
    fn denoise_channel(&mut self, _settings: &Denoiser, size: [u32;2],
        color: &mut [[f32;3]], albedo: &[[f32;3]], normals: &[[f32;3]],
        use_guides: bool, prepare_normals: bool) -> Result<()> {
        let bytes = std::mem::size_of_val(color);
        let color_buffer = Self::reserve(&self.device, &mut self.color, bytes)?;
        let output = Self::reserve(&self.device, &mut self.output, bytes)?;
        color_buffer.write(color)?;
        let mut guides = Vec::new();
        if use_guides {
            let buffer = Self::reserve(&self.device, &mut self.albedo, bytes)?;
            buffer.write(albedo)?;
            guides.push((c"albedo", buffer));
        }
        if use_guides {
            let buffer = Self::reserve(&self.device, &mut self.normal, bytes)?;
            if prepare_normals {buffer.write(normals)?;}
            guides.push((c"normal", buffer));
        }
        let quality = OIDN_QUALITY_HIGH;
        {
            for (name, buffer) in &guides {
                if *name==c"normal" && !prepare_normals {continue;}
                let filter = self.device.filter()?;
                filter.image(name, buffer, size);
                filter.image(c"output", buffer, size);
                unsafe { (self.device.api.integer)(filter.handle, c"quality".as_ptr(), quality); }
                filter.run()?;
            }
        }
        let filter = self.device.filter()?;
        filter.image(c"color", color_buffer, size);
        filter.image(c"output", output, size);
        for (name, buffer) in &guides {filter.image(name, buffer, size);}
        unsafe {
            (self.device.api.boolean)(filter.handle, c"hdr".as_ptr(), true);
            // Beauty uses the renderer’s scene-linear working units. Per-image HDR
            // autoexposure would couple pixels to unrelated sampling roundoff.
            (self.device.api.float)(filter.handle, c"inputScale".as_ptr(), 1.);
            (self.device.api.boolean)(filter.handle, c"cleanAux".as_ptr(), true);
            (self.device.api.integer)(filter.handle, c"quality".as_ptr(), quality);
        }
        filter.run()?;
        unsafe { (self.device.api.read)(output.handle as Handle, 0, bytes, color.as_mut_ptr().cast()); }
        self.device.check()
    }
}
struct Buffer {
    engine: Arc<Device>,
    handle: usize,
    bytes: usize,
}
impl Buffer {
    fn write(&self, data: &[[f32; 3]]) -> Result<()> {
        unsafe { (self.engine.api.write)(self.handle as Handle, 0, std::mem::size_of_val(data), data.as_ptr().cast()); }
        self.engine.check()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        if self.handle != 0 {
            unsafe {
                (self.engine.api.release_buffer)(self.handle as Handle);
            }
        }
    }
}
struct Filter {
    engine: Arc<Device>,
    handle: Handle,
}
// Filters are only used under the renderer's denoiser mutex. OIDN permits
// sequential calls from different threads; the device outlives every filter.
unsafe impl Send for Filter {}
impl Filter {
    fn image(&self, name: &CStr, buffer: &Buffer, [w, h]: [u32; 2]) {
        unsafe {
            (self.engine.api.image)(
                self.handle,
                name.as_ptr(),
                buffer.handle as Handle,
                3,
                w as usize,
                h as usize,
                0,
                0,
                0,
            );
        }
    }
    fn run(&self) -> Result<()> {
        self.engine.check()?;
        unsafe {
            (self.engine.api.commit_filter)(self.handle);
        }
        self.engine.check()?;
        unsafe {
            (self.engine.api.execute)(self.handle);
        }
        self.engine.check()
    }
}
impl Drop for Filter {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                (self.engine.api.release_filter)(self.handle);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Timing {
    pub finish_ms: f64,
    pub wait_ms: f64,
    pub pack_ms: f64,
    pub filter_ms: f64,
    pub upload_ms: f64,
}

struct Inputs {
    associated: Vec<[f32; 4]>,
    color: Vec<[f32; 3]>,
    albedo: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    surface: Vec<bool>,
}

fn unpack_inputs(data: &[u8], samples: usize, count: usize, guide_alpha: bool) -> Inputs {
    let color_bytes = count * 16;
    let associated: Vec<[f32; 4]> = data[..color_bytes]
        .chunks_exact(16)
        .map(|p| std::array::from_fn(|c| {
            f32::from_ne_bytes(p[c * 4..c * 4 + 4].try_into().unwrap()) / samples as f32
        }))
        .collect();
    let color = associated.iter()
        .map(|p| std::array::from_fn(|c| if p[3] > 0. { p[c] / p[3] } else { 0. }))
        .collect();
    let guide = |plane: usize| -> Vec<[f32; 3]> {
        (0..count).map(|i| {
            let start = color_bytes + i * 32 + plane * 16;
            // Guides carry associated RGB. Unassociate only after the complete
            // lens/exposure sum, using albedo's coverage for both guide planes.
            let coverage = if guide_alpha {
                let at = color_bytes + i * 32 + 12;
                f32::from_ne_bytes(data[at..at + 4].try_into().unwrap())
            } else {
                samples as f32
            };
            std::array::from_fn(|c| if coverage > 0. {
                f32::from_ne_bytes(data[start + c * 4..start + c * 4 + 4].try_into().unwrap()) / coverage
            } else { 0. })
        }).collect()
    };
    let albedo = guide(0);
    // Length retains camera-sample coverage and normal dispersion.
    // Normalizing this average promotes a rare strand hit to a full guide.
    let normals = guide(1);
    let surface=(0..count).map(|i| {
        let at=color_bytes+i*32+12;
        f32::from_ne_bytes(data[at..at+4].try_into().unwrap())>0.
    }).collect();
    Inputs { associated, color, albedo, normals, surface }
}

pub(crate) struct Stage {
    gpu: Option<gpu::Stage>,
    settings: Denoiser,
    size: [u32; 2],
    accumulation: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: Option<wgpu::Buffer>,
    guides: wgpu::Buffer,
    guide_alpha: bool,
    use_guides: bool,
    pub(crate) spectral: bool,
    feedback: Option<Feedback>,
}
struct Feedback {
    sources: Vec<(wgpu::Buffer, std::ops::Range<u64>)>,
    staging: wgpu::Buffer,
}
impl Feedback {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut offset = 0;
        for (source, range) in &self.sources {
            let bytes = range.end - range.start;
            encoder.copy_buffer_to_buffer(source, range.start, &self.staging, offset, bytes);
            offset += bytes;
        }
    }
    fn rejected(&self) -> bool {
        let data = self.staging.get_mapped_range(..);
        let rejected = data.chunks_exact(4).any(|word| u32::from_ne_bytes(word.try_into().unwrap()) != 0);
        drop(data);
        self.staging.unmap();
        rejected
    }
}
/// Initialize once before building denoising pipelines, allowing callers to fall back.
pub(crate) fn ensure_runtime(r: &Renderer) -> Result<()> {
    if !r.release_preparation {
        let mut engine = r.denoiser.lock().map_err(|_| anyhow::anyhow!("OIDN engine lock poisoned"))?;
        if engine.is_none() {
            let start = std::time::Instant::now();
            let device = Device::open_for(Some(r)).map_err(crate::failure::setup)?;
            crate::timing::creation(&r.device, if device.gpu {"OIDN GPU"} else {"OIDN CPU"},
                "denoiser device", 0., start.elapsed().as_secs_f64()*1000.);
            *engine = Some(Engine::with_device(device)?);
        }
    }
    Ok(())
}

/// Open this adapter's GPU denoiser, or say why it has none, without opening a
/// CPU device the caller may not want. Only one-time device creation costs time.
pub(crate) fn open_fast(r: &Renderer) -> std::result::Result<(), String> {
    let mut engine = r.denoiser.lock().map_err(|_| "OIDN engine lock poisoned".to_string())?;
    if let Some(engine) = engine.as_ref() {
        return if engine.device.gpu { Ok(()) } else { Err("OIDN opened on the CPU".into()) };
    }
    let start = std::time::Instant::now();
    let api = Device::load_api().map_err(|error| format!("{error:#}"))?;
    let handle = gpu::select(&api, r)?;
    let opened = Device::adopt(api, handle, true).and_then(Engine::with_device).map_err(|error| format!("{error:#}"))?;
    crate::timing::creation(&r.device, "OIDN GPU", "denoiser device", 0., start.elapsed().as_secs_f64()*1000.);
    *engine = Some(opened);
    Ok(())
}

/// Apply the default denoising choice for this device at render start. Called only
/// for a library draft or final profile whose call did not ask for denoising itself.
pub(crate) fn resolve_default(r: &Renderer, scene: &mut vork_scene::Scene, profile: &str, level: u8, keep_samples: bool) {
    let Some(reason) = resolve_default_with(&mut scene.render.settings, level, keep_samples, || open_fast(r)) else { return };
    eprintln!("No fast GPU denoiser for {} ({reason}): {profile} renders {} samples without denoising. --set denoise=true uses the slow CPU denoiser anyway; --set samples=N sets the budget.",
        r.adapter_info.name, scene.render.settings.samples);
}

/// Capability-result injection keeps simulated failures out of device discovery
/// and lets tests exercise the same default decision without a GPU or OIDN.
fn resolve_default_with(
    settings: &mut vork_scene::v2::Settings,
    level: u8,
    keep_samples: bool,
    open: impl FnOnce() -> std::result::Result<(), String>,
) -> Option<String> {
    if settings.integrator.denoiser().is_none() { return None; }
    let Err(reason) = open() else { return None };
    settings.without_fast_denoiser(level, keep_samples);
    Some(reason)
}

#[cfg(test)]
mod capability_tests {
    use super::resolve_default_with;
    use vork_scene::v2::{CallOverrides, Profile};

    #[test]
    fn injected_capability_results_preserve_default_and_explicit_budgets() {
        for name in ["std/draft", "std/final"] {
            let profile = Profile::load(name).unwrap();
            for explicit in [false, true] {
                let call = if explicit {
                    CallOverrides::resolve([("samples".into(), serde_json::json!(7))]).unwrap()
                } else { Default::default() };
                let settings = profile.resolve(&call.settings).unwrap();
                let mut available = settings.clone();
                assert!(resolve_default_with(&mut available, profile.level, explicit, || Ok(())).is_none());
                assert_eq!(available, settings);
                let mut unavailable = settings.clone();
                assert_eq!(resolve_default_with(&mut unavailable, profile.level, explicit,
                    || Err("injected unavailable device".into())).as_deref(), Some("injected unavailable device"));
                assert!(unavailable.integrator.denoiser().is_none());
                assert_eq!(unavailable.samples, if explicit {7} else if name == "std/draft" {16} else {32});
            }
            let call = CallOverrides::resolve([("denoise".into(), serde_json::json!(false))]).unwrap();
            let mut disabled = profile.resolve(&call.settings).unwrap();
            let original = disabled.clone();
            assert!(resolve_default_with(&mut disabled, profile.level, false,
                || panic!("explicit denoise=false must not probe capabilities")).is_none());
            assert_eq!(disabled, original);
        }
    }
}

/// The device an open denoiser runs on, for output metadata.
pub(crate) fn device_kind(r: &Renderer) -> Option<&'static str> {
    r.denoiser.lock().ok()?.as_ref().map(|engine| if engine.device.gpu {"gpu"} else {"cpu"})
}

impl Stage {
    pub(crate) fn new(
        r: &Renderer,
        settings: &Denoiser,
        accumulation: &wgpu::Buffer,
        output: &wgpu::Buffer,
        guides: &wgpu::Buffer,
        guide_alpha: bool,
        use_guides: bool,
        size: [u32; 2],
    ) -> Result<Self> {
        ensure_runtime(r)?;
        // Asset preparation constructs our pack/unpack pipelines, but never
        // opens OIDN or runs a filter on the nonexecuting device.
        let gpu = if r.release_preparation { Some(gpu::Stage::new(r)) } else {
            r.denoiser.lock().map_err(|_| anyhow::anyhow!("OIDN engine lock poisoned"))?
                .as_ref().unwrap().device.gpu.then(|| gpu::Stage::new(r))
        };
        let bytes = u64::from(size[0]) * u64::from(size[1]) * 48;
        let host_staging = gpu.is_none() || cfg!(test);
        Ok(Self {
            gpu,
            settings: settings.clone(),
            size,
            accumulation: accumulation.clone(),
            output: output.clone(),
            staging: host_staging.then(|| crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
                label: Some("OIDN host staging"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })),
            guides: guides.clone(),
            guide_alpha,
            use_guides,
            spectral: false,
            feedback: None,
        })
    }
    /// Read small rejection flags before filtering. Collection still owns the
    /// exact retry demand; GPU image data never enters this readback.
    pub(crate) fn set_rejection_feedback(&mut self, r: &Renderer, sources: Vec<(wgpu::Buffer, std::ops::Range<u64>)>) {
        let bytes = sources.iter().map(|(_, range)| range.end - range.start).sum::<u64>();
        self.feedback = Some(Feedback {
            sources,
            staging: crate::gpu_memory::buffer(&r.device, &wgpu::BufferDescriptor {
                label: Some("OIDN rejection feedback"), size: bytes.max(4),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        });
    }
    pub(crate) fn run(
        &self,
        r: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        samples: usize,
        input_scale: f32,
        timing: &mut Timing,
        segments: &mut crate::command_finish::Segments,
    ) -> Result<()> {
        if let Some(gpu) = &self.gpu {
            if let Some(feedback) = &self.feedback {
                // Reject before allocating filters or releasing image ownership.
                // Only small capacity/error flags cross the host boundary.
                let start = std::time::Instant::now();
                feedback.encode(encoder);
                let (submission, finish) = segments.interop(r, encoder, "OIDN rejection feedback");
                timing.finish_ms += finish;
                let (tx, rx) = std::sync::mpsc::channel();
                feedback.staging.map_async(wgpu::MapMode::Read, .., move |result| { let _ = tx.send(result); });
                r.wait(submission)?;
                rx.recv()??;
                timing.wait_ms += start.elapsed().as_secs_f64()*1000.;
                if feedback.rejected() { return Ok(()); }
            }
            return gpu.run(self, r, encoder, samples, input_scale, timing, segments);
        }
        let staging = self.staging.as_ref().context("OIDN CPU staging missing")?;
        let start = std::time::Instant::now();
        let [w, h] = self.size;
        let count = w as usize * h as usize;
        let color_bytes = count as u64 * 16;
        encoder.copy_buffer_to_buffer(&self.accumulation, 0, staging, 0, color_bytes);
        encoder.copy_buffer_to_buffer(
            &self.guides,
            0,
            staging,
            color_bytes,
            count as u64 * 32,
        );
        if let Some(feedback) = &self.feedback { feedback.encode(encoder); }
        let (submission, finish_ms) = segments.interop(r, encoder, "after OIDN");
        timing.finish_ms += finish_ms;
        let (tx, rx) = std::sync::mpsc::channel();
        for buffer in std::iter::once(staging).chain(self.feedback.as_ref().map(|f| &f.staging)) {
            let tx = tx.clone();
            buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                let _ = tx.send(result);
            });
        }
        r.wait(submission)?;
        for _ in 0..1 + usize::from(self.feedback.is_some()) { rx.recv()??; }
        let wait_ms = start.elapsed().as_secs_f64() * 1000.;
        timing.wait_ms += wait_ms;
        if self.feedback.as_ref().is_some_and(Feedback::rejected) {
            staging.unmap();
            return Ok(());
        }
        let pack_start = std::time::Instant::now();
        let data = staging.get_mapped_range(..);
        // Copy bytes rather than casting mapped pointers (their alignment is not an API guarantee).
        let Inputs { mut associated, mut color, mut albedo, normals, surface } =
            unpack_inputs(&data, samples, count, self.guide_alpha);
        drop(data);
        staging.unmap();
        if self.spectral {
            for rgb in color.iter_mut().chain(albedo.iter_mut()) { *rgb=spectral_color(*rgb,false); }
        }
        for rgb in &mut color { for c in rgb { *c *= input_scale; } }
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.;
        let filter_start = std::time::Instant::now();
        let mut engine=r.denoiser.lock().map_err(|_| anyhow::anyhow!("OIDN engine lock poisoned"))?;
        let engine=engine.as_mut().context("OIDN runtime not initialized")?;
        if self.spectral {engine.denoise_spectral(&self.settings,self.size,&mut color,&albedo,&normals,self.use_guides)?;}
        else {engine.denoise(&self.settings,self.size,&mut color,&albedo,&normals,self.use_guides)?;}
        let filter_ms = filter_start.elapsed().as_secs_f64() * 1000.;
        for rgb in &mut color { for c in rgb.iter_mut() { *c /= input_scale; } }
        if self.spectral { for rgb in &mut color { *rgb=spectral_color(*rgb,true); } }
        let upload_start = std::time::Instant::now();
        for ((p, rgb), surface) in associated.iter_mut().zip(color).zip(surface) {
            // An empty guide has no surface transport to denoise. Keep the
            // sampled background rather than replacing it with the network's
            // reconstruction, which need not preserve even a constant World.
            if !surface { continue; }
            for c in 0..3 {
                p[c] = rgb[c] * p[3];
            }
        }
        r.queue
            .write_buffer(&self.output, 0, bytemuck::cast_slice(&associated));
        // Bloom and compositor consume the accumulated sum and divide by samples.
        for p in &mut associated {
            for value in p {
                *value *= samples as f32;
            }
        }
        r.queue
            .write_buffer(&self.accumulation, 0, bytemuck::cast_slice(&associated));
        timing.pack_ms += pack_ms;
        timing.filter_ms += filter_ms;
        timing.upload_ms += upload_start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;
    const MIB: u64 = 1024 * 1024;
    const GIB: u64 = 1024 * MIB;

    #[test]
    fn usage_growth_spans_renderer_and_workspace_creation() -> Result<()> {
        // Existing renderers own 2 GiB before this renderer snapshots 6 GiB free.
        // Later allocations (including shared OIDN planes) bring total use to 6.
        assert_eq!(memory_limit_mb(6 * GIB, 2 * GIB, 6 * GIB, 2 * GIB)?, 2048);
        assert_eq!(memory_limit_mb(6 * GIB, 2 * GIB, 2 * GIB, 6 * GIB)?, 6144);
        // Releasing pre-existing allocations must not increase the stable quota.
        assert_eq!(memory_limit_mb(6 * GIB, 2 * GIB, GIB, 7 * GIB)?, 6144);
        Ok(())
    }

    #[test]
    fn shared_card_does_not_charge_unavailable_capacity_twice() -> Result<()> {
        let heap_budget = 76 * GIB / 10;
        let usage_at_creation = 37 * GIB / 10;
        let headroom = heap_budget - usage_at_creation;
        assert_eq!(memory_limit_mb(headroom, usage_at_creation, usage_at_creation, headroom)?, 3993);
        // Other applications lower the process's native budget, not its usage.
        let shared_heap_budget = heap_budget - 3700 * MIB;
        let camera_budget = shared_heap_budget - 50 * MIB;
        assert_eq!(memory_limit_mb(camera_budget, 50 * MIB, 300 * MIB,
            shared_heap_budget - 300 * MIB)?, 3782);
        Ok(())
    }

    #[test]
    fn exclusive_card_bounds_scratch_by_its_remaining_budget() -> Result<()> {
        let heap_budget = 76 * GIB / 10;
        let camera_budget = heap_budget - 50 * MIB;
        assert_eq!(memory_limit_mb(camera_budget, 50 * MIB, 300 * MIB,
            heap_budget - 300 * MIB)?, 7482);
        Ok(())
    }

    #[test]
    fn oidn_memory_target_uses_remaining_camera_budget() -> Result<()> {
        // Qualification lowers startup headroom; only subsequent growth spends it.
        assert_eq!(memory_limit_mb(5 * GIB, 2 * GIB, 6 * GIB, 2 * GIB)?, 1024);
        assert_eq!(memory_limit_mb(8192 * MIB, 2000 * MIB, 7000 * MIB, 17000 * MIB)?, 3192);
        // Native headroom can shrink after the camera budget was established.
        assert_eq!(memory_limit_mb(8192 * MIB, 2000 * MIB, 7000 * MIB, 600 * MIB)?, 600);
        assert_eq!(memory_limit_mb(24000 * MIB, 0, 8000 * MIB, 16000 * MIB)?, 16000);
        assert_eq!(memory_limit_mb(3 * MIB - 1, MIB, 2 * MIB, 2 * MIB)?, 1);
        assert_eq!(memory_limit_mb(u64::MAX, 0, 0, u64::MAX)?, i32::MAX);
        Ok(())
    }

    #[test]
    fn exhausted_oidn_budget_refuses_with_one_actionable_line() {
        for (budget, initial, usage, headroom) in [(0, 0, 0, 0),
            (6 * GIB, 2 * GIB, 8 * GIB, 0), (3 * GIB, 2 * GIB, 6 * GIB, 2 * GIB),
            (u64::MAX, 0, 0, 0), (MIB - 1, 0, 0, MIB), (8 * MIB, MIB, 8 * MIB + 1, 4 * MIB)] {
            let error = memory_limit_mb(budget, initial, usage, headroom).unwrap_err();
            assert!(error.is::<OutOfMemory>(), "budget pressure must use the existing eviction retry");
            let message = error.to_string();
            assert_eq!(message, "denoising needs more memory; reduce resolution or use --set denoise=false");
            assert_eq!(message.lines().count(), 1);
            // Whichever caller it reaches (a still beside a video, a world in a canvas), it is the device's.
            assert_eq!(crate::failure::exit_code(&crate::failure::scene(error.context("denoise"))), 4);
        }
    }
}

#[cfg(all(test, vork_private_tests))]
include!("denoise_tests_private.rs");

/// Initialize the actual runtime and device, rather than checking only a filename.
pub(crate) fn check_runtime() -> Result<()> {
    Engine::new().map(|_| ())
}
