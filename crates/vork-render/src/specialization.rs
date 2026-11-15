//! CLI-owned, detached path specialization. No scene assets or GPU submissions.
use crate::disk_cache;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{path::{Path, PathBuf}, sync::atomic::{AtomicBool, Ordering}, io::Write};

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Only the CLI promises to implement the private worker entry point.
pub fn enable() { ENABLED.store(true, Ordering::Release); }
pub(crate) fn enabled() -> bool { ENABLED.load(Ordering::Acquire) }

#[derive(Serialize, Deserialize)]
pub(crate) struct Source {
    pub modules: Vec<(String,String)>,
    pub args: Vec<String>,
    pub identities: std::collections::BTreeMap<String,Vec<u8>>,
    pub main: String,
    pub entry: String,
    pub threads: (u32,u32,u32),
    pub metal: bool,
    pub transforms: crate::slang_forward::EntryTransforms,
    pub identity: Vec<u8>,
    pub artifact: PathBuf,
}
#[derive(Serialize, Deserialize)]
struct Request {
    build: String,
    gpu: Option<String>,
    key: [u8;32],
    path: PathBuf,
    source: Source,
    groups: Vec<Option<Vec<wgpu::BindGroupLayoutEntry>>>,
}

pub(crate) fn load(device: &wgpu::Device, path: &Path) -> Option<wgpu::PipelineCache> {
    let data = disk_cache::read_pipeline(path)?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // SAFETY: checksummed cache produced by this build on the same driver/device.
    let cache = unsafe { device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
        label: Some("scene specialization"), data: Some(&data), fallback: false,
    }) };
    if pollster::block_on(scope.pop()).is_some() {
        // A rejected entry must not suppress the worker's next attempt.
        let _ = std::fs::remove_file(path);
        return None;
    }
    Some(cache)
}

pub(crate) fn start(key: blake3::Hash, path: PathBuf, source: Source,
    groups: Vec<Option<Vec<wgpu::BindGroupLayoutEntry>>>) -> Result<Option<std::sync::Arc<AtomicBool>>> {
    use std::process::{Command, Stdio};
    let request = Request { build: env!("VORK_BUILD_ID").into(), gpu: crate::selected_gpu()?.map(|v|v.to_string()), key:*key.as_bytes(),path,source,groups };
    let mut command = Command::new(std::env::current_exe()?);
    command.arg("--specialization-worker").stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null());
    if let Some(path) = std::env::var_os("VORK_SPECIALIZATION_LOG") {
        command.stderr(std::fs::OpenOptions::new().create(true).append(true).open(path)?);
    }
    #[cfg(unix)] {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid is async-signal-safe; no allocation or locks after fork.
        unsafe { command.pre_exec(|| { if libc::setsid() < 0 {return Err(std::io::Error::last_os_error());} Ok(()) }); }
    }
    #[cfg(windows)] {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x00000008 | 0x00000200); // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    }
    let mut child = match command.spawn() { Ok(child)=>child, Err(_)=>return Ok(None) };
    let mut input = child.stdin.take().unwrap();
    let bytes = serde_json::to_vec(&request)?;
    let sent = input.write_all(&bytes).is_ok();
    drop(input);
    // Reap while the CLI lives; after CLI exit the detached child is adopted.
    let ended = std::sync::Arc::new(AtomicBool::new(false));
    let done = ended.clone();
    std::thread::spawn(move || { let _ = child.wait(); done.store(true, Ordering::Release); });
    Ok(sent.then_some(ended))
}

pub(crate) fn wait(device: &wgpu::Device, path: &Path, ended: &AtomicBool) -> Result<wgpu::PipelineCache> {
    let start = std::time::Instant::now();
    loop {
        if let Some(cache) = load(device,path) {return Ok(cache);}
        ensure!(!ended.load(Ordering::Acquire),"detached specialization exited without publishing");
        ensure!(start.elapsed().as_secs()<600,"detached specialization did not publish within 600 seconds");
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Recognized before ordinary CLI parsing, never starts a renderer recursively.
pub fn worker_entry() -> Option<Result<()>> {
    (std::env::args_os().nth(1).is_some_and(|arg|arg=="--specialization-worker")).then(worker)
}
fn worker() -> Result<()> {
    let request: Request = serde_json::from_reader(std::io::stdin())?;
    ensure!(request.build==env!("VORK_BUILD_ID"),"specialization executable changed");
    let root = request.path.parent().context("specialization cache directory")?;
    std::fs::create_dir_all(root)?;
    // Fixed striped locks bound metadata; never unlink a lock inode while another
    // process may have opened it. Kernel locks release on every exit, including kill.
    let lock = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false)
        .open(root.join(format!("specialization-{:02x}.lock",request.key[0])))?;
    lock.lock()?;
    if disk_cache::read_pipeline(&request.path).is_some() {return Ok(());}
    // One private scratch directory per lock stripe. A killed worker's files
    // are reclaimed by its next lock owner; successful/error exits remove them.
    let scratch = root.join(format!("specialization-work-{:02x}",request.key[0]));
    if scratch.exists() { std::fs::remove_dir_all(&scratch)?; }
    std::fs::create_dir(&scratch)?;
    let scratch = Scratch(scratch.canonicalize()?);
    // SAFETY: this private entry runs before CLI initialization and before any
    // thread or renderer is created. All later threads inherit a fixed temp root.
    unsafe {
        #[cfg(unix)] std::env::set_var("TMPDIR", &scratch.0);
        #[cfg(windows)] {
            std::env::set_var("TMP", &scratch.0);
            std::env::set_var("TEMP", &scratch.0);
        }
    }
    // A wedged compiler/driver cannot retain a detached job and its lock forever.
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(600));
        std::process::exit(1);
    });
    let source = &request.source;
    let compiler = vork_slang::Compiler::discover();
    let modules: Vec<_> = source.modules.iter().map(|(n,s)|(n.as_str(),s.as_str())).collect();
    let args: Vec<_> = source.args.iter().map(String::as_str).collect();
    let prepared = compiler.prepare_identified_modules(&modules,&source.main,&[(&source.entry,"compute")],&args,&source.identities)?;
    ensure!(prepared.receipt().identity==source.identity,"specialization inputs changed before worker capture");
    let code = if let Some(code)=disk_cache::read(&source.artifact) {code} else {
        let mut code = prepared.compile()?;
        if source.transforms.localize_blends {code=compiler.localize_native_blends(&code)?;}
        if source.transforms.inline_metal {code=crate::camera_native::inline_metal_functions(&code);}
        if source.transforms.driver_inline {crate::slang_forward::allow_driver_inlining(&mut code)?;}
        disk_cache::write_entry(&source.artifact,&code)?;
        code
    };
    drop(prepared);
    // Slang is finished before opening a device. Driver pipeline creation needs
    // a device, but allocates no scene buffers and submits no rendering work.
    if let Some(gpu)=request.gpu {crate::select_gpu(gpu.parse()?);}
    let r = crate::Renderer::new()?;
    ensure!(r.disk_cache.path("specialization",1,request.key.into())==request.path,"specialization adapter changed");
    ensure!(r.cache.is_some(), "specialization needs persistent driver cache");
    // SAFETY: an empty private cache has no externally supplied opaque bytes.
    let cache = unsafe { r.device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
        label: Some("scene specialization"), data: None, fallback: true,
    }) };
    let module = if source.metal {
        let text=std::str::from_utf8(&code)?;
        vork_slang::metal_source::validate(text)?;
        crate::native::module(&r.device,&source.entry,None,Some(text),source.threads)
    } else {crate::native::module(&r.device,&source.entry,Some(&code),None,source.threads)};
    let groups: Vec<_> = request.groups.iter().map(|entries|entries.as_ref().map(|entries|
        r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:None,entries}))).collect();
    let refs: Vec<_> = groups.iter().map(Option::as_ref).collect();
    let layout = r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label:None,bind_group_layouts:&refs,immediate_size:0});
    let _pipeline = r.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label:Some("scene specialization"), layout:Some(&layout), module:&module, entry_point:Some(&source.entry),
        compilation_options:Default::default(),cache:Some(&cache),
    });
    r.errors.check()?;
    let data=cache.get_data().context("serialize scene specialization")?;
    disk_cache::write_entry(&request.path,&data)?;
    // Renderer compiler never created this pipeline, so Drop cannot overwrite
    // the shared cache with the worker's older snapshot.
    Ok(())
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}
