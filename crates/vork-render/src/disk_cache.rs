//! The renderer's single persistent store. Callers own content keys and payload
//! formats; this module owns location, integrity, atomic publication and eviction.
use crate::constants::{DISK_CACHE_BUDGET, DISK_CACHE_HEADROOM_DIVISOR};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::SystemTime,
};

#[derive(Clone)]
pub struct Store {
    root: PathBuf,
    identity: blake3::Hash,
}
impl Store {
    #[cfg(test)]
    pub(crate) fn for_test(root: PathBuf) -> Self {
        Self { root, identity: blake3::hash(b"isolated-test-store") }
    }
    /// Host preprocessing shares the renderer's store, checksums and eviction.
    pub fn host() -> Self {
        let store=Self {root:std::env::var_os("VORK_CACHE_DIR").map(PathBuf::from).unwrap_or_else(default_root),identity:blake3::hash(b"host-preprocessing")};
        store.trim();store
    }
    pub fn new(adapter: &wgpu::AdapterInfo) -> Self {
        let device_identity = device_identity(adapter);
        let mut identity = blake3::Hasher::new();
        identity.update(env!("VORK_BUILD_ID").as_bytes());
        identity.update(device_identity.as_bytes());
        let store = Self {
            identity: identity.finalize(),
            root: std::env::var_os("VORK_CACHE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(default_root),
        };
        store.trim();
        store
    }
    pub fn path(&self, kind: &str, version: u32, hash: blake3::Hash) -> PathBuf {
        let mut key = blake3::Hasher::new();
        // CPU decoding is independent of renderer builds and adapters. Codec
        // revisions belong in the caller's explicit format version.
        match kind {
            "decoded-texture" | "native-decoded-texture" => {}
            // Slang emits native source/IR, not device machine code. Generated
            // scene text is the caller's key; embedded modules, generators and
            // pinned compiler inputs have their own content identity.
            "slang-variant" => {
                key.update(env!("VORK_SHADER_ID").as_bytes());
            }
            "slang-program" | "slang-resolve" | "aov-resolve" => {
                key.update(env!("VORK_PORTABLE_ID").as_bytes());
            }
            "compressed-texture" => {
                key.update(env!("VORK_TEXTURE_BUILD_ID").as_bytes());
            }
            _ => {
                key.update(self.identity.as_bytes());
            }
        }
        key.update(hash.as_bytes());
        key.update(kind.as_bytes());
        key.update(&version.to_le_bytes());
        self.root
            .join(format!("{}.{kind}-v{version}.entry", key.finalize()))
    }
    /// Opening a store trims it, including what older builds left; every publication makes its own room.
    pub fn trim(&self) {
        network_notice(&self.root);
        let _ = (|| -> std::io::Result<()> {
            let _lock = lock_store(&self.root)?;
            self.make_room(0)?;
            Ok(())
        })();
    }
    fn make_room(&self, incoming: u64) -> std::io::Result<bool> {
        let mut files = Vec::new();
        for entry in fs::read_dir(&self.root)?.flatten() {
            let path = entry.path();
            // Every writer holds the store lock from creating its temporary file to renaming it, and
            // the caller holds that lock now: a temporary file here was left by a process that
            // ended mid-publication (a one-shot exits without waiting for a background save).
            if path.extension().is_some_and(|ext| ext == "tmp")
                && entry.file_name().to_string_lossy().starts_with("store-")
            {
                let _ = fs::remove_file(&path);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "entry") {
                continue;
            }
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if metadata.is_file() {
                files.push((metadata.modified()?, metadata.len(), path));
            }
        }
        let bytes = files.iter().map(|(_, len, _)| len).sum::<u64>();
        let configured = std::env::var("VORK_CACHE_BYTES")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DISK_CACHE_BUDGET);
        let budget = available_space(&self.root).map_or(configured, |free| {
            configured.min(bytes.saturating_add(free) / DISK_CACHE_HEADROOM_DIVISOR)
        });
        if incoming > budget {
            return Ok(false);
        }
        let remaining = self.trim_to_budget(files, bytes, budget - incoming);
        Ok(remaining <= budget - incoming)
    }
    fn trim_to_budget(
        &self,
        mut files: Vec<(SystemTime, u64, PathBuf)>,
        mut bytes: u64,
        budget: u64,
    ) -> u64 {
        if bytes <= budget {
            return bytes;
        }
        files.sort_unstable_by_key(|(used, _, _)| *used);
        for (_, len, path) in files {
            if bytes <= budget {
                break;
            }
            if fs::remove_file(path).is_ok() {
                bytes -= len;
            }
        }
        bytes
    }
}

/// What makes a driver blob reusable: the GPU model and the driver build, never where the card
/// sits. The PCI bus address (and anything else wgpu adds to `AdapterInfo` later) changes from one
/// machine to the next for the same card and driver, so hashing the whole `Debug` output made every
/// cloud or multi-GPU machine miss caches built for an identical card. A blob the driver does not
/// accept is still caught when it is loaded: wgpu validates the pipeline-cache header, and a
/// rejected blob recompiles.
fn device_identity(adapter: &wgpu::AdapterInfo) -> blake3::Hash {
    let wgpu::AdapterInfo { name, vendor, device, device_type, driver, driver_info, backend, .. } = adapter;
    blake3::hash(format!("{name:?}|{vendor}|{device}|{device_type:?}|{driver:?}|{driver_info:?}|{backend:?}").as_bytes())
}

/// Optional captures reserve their peak owned bytes before allocating readback
/// buffers. Outstanding content keys coalesce across producers, including the
/// active writer while it waits for the filesystem. This never limits rendering.
#[derive(Default)]
struct PersistenceState {
    bytes: usize,
    keys: std::collections::HashSet<PathBuf>,
}
struct PersistenceAdmission {
    state: std::sync::Mutex<PersistenceState>,
    capacity: usize,
}
pub(crate) struct PersistenceReservation {
    admission: std::sync::Arc<PersistenceAdmission>,
    key: PathBuf,
    bytes: usize,
}
impl PersistenceAdmission {
    fn reserve(self: &std::sync::Arc<Self>, key: &Path, bytes: usize) -> Option<PersistenceReservation> {
        let mut state=self.state.try_lock().ok()?;
        if bytes==0 || bytes>self.capacity.saturating_sub(state.bytes) || state.keys.contains(key) {return None;}
        state.keys.insert(key.to_owned());state.bytes+=bytes;
        Some(PersistenceReservation {admission:self.clone(),key:key.to_owned(),bytes})
    }
}
fn persistence_admission() -> &'static std::sync::Arc<PersistenceAdmission> {
    static ADMISSION:std::sync::OnceLock<std::sync::Arc<PersistenceAdmission>>=std::sync::OnceLock::new();
    ADMISSION.get_or_init(||std::sync::Arc::new(PersistenceAdmission {
        state:Default::default(),capacity:crate::constants::CACHE_PERSISTENCE_BYTES,
    }))
}
pub(crate) fn reserve_persistence(key: &Path, bytes: usize) -> Option<PersistenceReservation> {
    persistence_admission().reserve(key,bytes)
}
impl Drop for PersistenceReservation {
    fn drop(&mut self) {
        let mut state=self.admission.state.lock().unwrap();
        state.bytes-=self.bytes;state.keys.remove(&self.key);
    }
}
impl PersistenceReservation {
    pub(crate) fn publish(self:std::sync::Arc<Self>, task: impl FnOnce() + Send + 'static) {
        persist(move || {let _reservation=self;task();});
    }
}
#[cfg(test)]
pub(crate) fn persistence_bytes() -> usize {persistence_admission().state.lock().unwrap().bytes}

/// Optional persistence owns its payload after GPU readback, and the pipeline cache is saved
/// here too (`pipelines::Compiler::save_soon`). One process worker serializes publication
/// through the ordinary store lock and eviction policy. Enqueueing never waits for the
/// filesystem; process exit may discard pending work, and leaves no partial entry: an
/// unpublished temporary file is removed by the next writer (`make_room`).
pub(crate) fn persist(task: impl FnOnce() + Send + 'static) {
    type Task = Box<dyn FnOnce() + Send>;
    static WORKER: std::sync::OnceLock<Option<std::sync::mpsc::Sender<Task>>> = std::sync::OnceLock::new();
    if let Some(sender) = WORKER.get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Task>();
        std::thread::Builder::new().name("vork-cache-persistence".into()).spawn(move || {
            for task in receiver { task(); }
        }).ok().map(|_| sender)
    }) {
        let _ = sender.send(Box::new(task));
    }
}

#[cfg(test)]
pub(crate) fn wait_persistence() -> anyhow::Result<()> {
    let (done,wait)=std::sync::mpsc::channel();
    persist(move || {let _=done.send(());});
    Ok(wait.recv_timeout(std::time::Duration::from_secs(30))?)
}

/// Writers and eviction share a persistent advisory lock across processes. Never
/// unlink the lock file: existing waiters must keep locking the same inode.
pub(crate) struct StoreLock(fs::File);
impl Drop for StoreLock {
    fn drop(&mut self) {
        // A forked child may retain the descriptor beyond this scope.
        let _ = self.0.unlock();
    }
}
pub(crate) fn lock_store(root: &Path) -> std::io::Result<StoreLock> {
    fs::create_dir_all(root)?;
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root.join("store.lock"))?;
    file.lock()?;
    let lock = StoreLock(file);
    // Recover the fixed-name reservation used by older writers. No live
    // writer can own it while we hold the store lock.
    match fs::remove_file(root.join("store.tmp")) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    Ok(lock)
}

/// Platform cache location for the same content-addressed store and trim policy.
fn default_root() -> PathBuf {
    #[cfg(target_os = "macos")]
    if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(home).join("Library/Caches/vork");
    }
    #[cfg(windows)]
    if let Some(local) = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
    {
        return local.join("vork");
    }
    #[cfg(not(any(target_os = "macos", windows)))]
    {
        if let Some(cache) = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
        {
            return cache.join("vork");
        }
        if let Some(home) = std::env::var_os("HOME").filter(|value| !value.is_empty()) {
            return PathBuf::from(home).join(".cache/vork");
        }
    }
    // Unusual service accounts without a home retain best-effort caching.
    std::env::temp_dir().join("vork")
}

// Bind the integrity envelope to the requested key as well as the bytes. A
// complete entry returned for a different filename is still an ordinary miss.
fn entry_hasher(path: &Path) -> blake3::Hasher {
    let mut hash = blake3::Hasher::new();
    hash.update(b"vork-cache-entry-v2\0");
    hash.update(path.file_name().unwrap_or_default().as_encoded_bytes());
    hash.update(b"\0");
    hash
}

/// A damaged, absent or oversized entry is an ordinary cache miss.
pub fn read(path: &Path) -> Option<Vec<u8>> {
    read_with_shaders(path, shipped_shaders().as_deref())
}

/// Release assets are an immutable base of the same content-addressed store.
/// Never accept driver blobs, known-pipeline ledgers or preprocessing entries
/// from a different machine. Missing or damaged assets compile on demand.
/// Each asset is its store entry (digest and payload) as one XZ stream, named
/// for the entry with `.xz` appended: the entries repeat each other little
/// enough alone, and a render reads only the few it needs, so each is packed
/// and expanded by itself. The digest is checked after expansion, as for a
/// store entry. A verified asset is then published to the store like any entry
/// this machine compiled, so it is expanded once, not at every read or probe;
/// a store that cannot take it (read-only, over budget) just expands it again.
fn read_with_shaders(path: &Path, shaders: Option<&Path>) -> Option<Vec<u8>> {
    read_bounded(path, usize::MAX as u64).or_else(|| {
        let name = path.file_name()?.to_str()?;
        let (key, kind) = name.split_once('.')?;
        if key.len() != 64 || !key.bytes().all(|b| b.is_ascii_hexdigit())
            || !matches!(kind, "slang-variant-v1.entry" | "slang-program-v2.entry"
                | "slang-resolve-v2.entry" | "aov-resolve-v1.entry" | "slang-closed-input-v1.entry") {
            return None;
        }
        let packed = fs::read(shaders?.join(format!("{name}.xz"))).ok()?;
        let mut entry = Vec::new();
        lzma_rs::xz_decompress(&mut packed.as_slice(), &mut entry).ok()?;
        let (digest, payload) = entry.split_at_checked(blake3::OUT_LEN)?;
        let mut checked = entry_hasher(path);
        checked.update(payload);
        if checked.finalize().as_bytes() != digest {
            return None;
        }
        let _ = write_entry(path, payload);
        Some(payload.to_vec())
    })
}

fn shipped_shaders() -> Option<PathBuf> {
    static DIRECTORY: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    DIRECTORY.get_or_init(|| Some(std::env::current_exe().ok()?.canonicalize().ok()?.parent()?.join("shaders"))).clone()
}

/// Immutable cache entries are atomically published and never modified in place.
/// An entry is trusted only through its digest, never because its file exists.
/// Keep the mapping alive through decoding/upload, including after eviction.
pub(crate) struct MappedPayload(memmap2::Mmap);

impl std::ops::Deref for MappedPayload {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.0[blake3::OUT_LEN..]
    }
}

pub(crate) fn read_mapped(path: &Path) -> Option<MappedPayload> {
    #[cfg(not(windows))]
    let file = fs::File::open(path).ok()?;
    #[cfg(windows)]
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .or_else(|_| fs::File::open(path))
        .ok()?;
    // SAFETY: the store only publishes new files by rename, and eviction unlinks
    // them. Neither operation mutates the opened file backing this mapping.
    let mapped = unsafe { memmap2::Mmap::map(&file) }.ok()?;
    let (digest, payload) = mapped.split_at_checked(blake3::OUT_LEN)?;
    let mut checked = entry_hasher(path);
    checked.update(payload);
    if checked.finalize().as_bytes() != digest {
        return None;
    }
    let _ = file.set_times(fs::FileTimes::new().set_modified(SystemTime::now()));
    Some(MappedPayload(mapped))
}

pub(crate) fn read_bounded(path: &Path, limit: u64) -> Option<Vec<u8>> {
    #[cfg(not(windows))]
    let mut file = fs::File::open(path).ok()?;
    // Windows SetFileTime requires write access. Read-only stores can still hit,
    // even when their last-use timestamp cannot be refreshed.
    #[cfg(windows)]
    let mut file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .or_else(|_| fs::File::open(path))
        .ok()?;
    let length = file.metadata().ok()?.len();
    if !(blake3::OUT_LEN as u64..=limit).contains(&length) {
        return None;
    }
    let mut digest = [0; blake3::OUT_LEN];
    file.read_exact(&mut digest).ok()?;
    let payload_len = usize::try_from(length).ok()?.checked_sub(blake3::OUT_LEN)?;
    let mut payload = Vec::new();
    payload.try_reserve_exact(payload_len).ok()?;
    payload.resize(payload_len, 0);
    file.read_exact(&mut payload).ok()?;
    let mut checked = entry_hasher(path);
    checked.update(&payload);
    if checked.finalize().as_bytes() != &digest {
        return None;
    }
    // Touch only metadata on a hit, never rewrite a large pixel payload.
    let _ = file.set_times(fs::FileTimes::new().set_modified(SystemTime::now()));
    Some(payload)
}

/// Driver blobs obey the shared store budget, just like compiler artifacts.
/// There is no independent entry cap: a complete driver blob cannot be truncated.
pub(crate) fn read_pipeline(path: &Path) -> Option<MappedPayload> {
    // Keep the verified immutable blob alive through driver ingestion without
    // allocating, zeroing and copying another full cache-sized buffer.
    read_mapped(path)
}

/// Report whether publication committed; budget refusal is not a saved entry.
pub(crate) fn write_entry(path: &Path, payload: &[u8]) -> std::io::Result<bool> {
    publish_parts(path, &[payload])
}

pub fn write(path: &Path, payload: &[u8]) -> std::io::Result<()> {
    write_parts(path, &[payload])
}

/// Serialize existing buffers directly, without copying decoded pixel allocations.
pub(crate) fn write_parts(path: &Path, parts: &[&[u8]]) -> std::io::Result<()> {
    publish_parts(path, parts).map(|_| ())
}

fn publish_parts(path: &Path, parts: &[&[u8]]) -> std::io::Result<bool> {
    let root = path.parent().unwrap();
    let _lock = lock_store(root)?;
    let incoming = parts
        .iter()
        .try_fold(blake3::OUT_LEN as u64, |n, p| n.checked_add(p.len() as u64))
        .ok_or_else(|| std::io::Error::other("cache payload overflow"))?;
    let store = Store {
        root: root.to_owned(),
        identity: blake3::hash(b""),
    };
    if !store.make_room(incoming)? {
        return Ok(false);
    }
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let (temporary, mut file) = loop {
        let temporary = root.join(format!("store-{}-{}-{}.tmp", std::process::id(),
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap_or_default().as_nanos(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)));
        match fs::OpenOptions::new().write(true).create_new(true).open(&temporary) {
            Ok(file) => break (temporary, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    };
    // Cleanup below only owns a file after exclusive creation succeeds. A
    // collision must never unlink another client's unpublished file.
    let result = (|| {
        let mut digest = entry_hasher(path);
        for part in parts {
            digest.update(part);
        }
        file.write_all(digest.finalize().as_bytes())?;
        for part in parts {
            file.write_all(part)?;
        }
        // No fsync. Readers only ever see a whole file: the rename is atomic for every process, and
        // the bytes survive this process ending. Only a machine that loses power can keep a renamed
        // entry without its bytes, and the digest above turns that entry into an ordinary miss. A
        // cache needs no more. With the flush, publishing one pipeline entry took 0.003 to 1.07 s on
        // an L4 host, and a short render exited before its entry was published; on macOS the flush
        // is a full device flush.
        drop(file);
        fs::rename(&temporary, path)
    })();
    let _ = fs::remove_file(temporary);
    result.map(|_| true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_lock_releases_retained_descriptor_on_all_exits() {
        let root = std::env::temp_dir().join(format!("vork-store-unlock-{}", std::process::id()));
        for exit in 0..3 {
            let mut retained = None;
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> std::io::Result<()> {
                let lock = lock_store(&root)?;
                retained = Some(lock.0.try_clone()?);
                match exit {
                    0 => Ok(()),
                    1 => Err(std::io::Error::other("injected I/O failure")),
                    _ => panic!("injected unwind"),
                }
            }));
            let file = fs::File::open(root.join("store.lock")).unwrap();
            file.try_lock().expect("store scope must release retained descriptors");
            let _lock = StoreLock(file);
            drop(retained);
        }
        // Recovery failure occurs after acquisition and must release too.
        fs::create_dir(root.join("store.tmp")).unwrap();
        assert!(lock_store(&root).is_err());
        let file = fs::File::open(root.join("store.lock")).unwrap();
        file.try_lock().unwrap();
        drop(StoreLock(file));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn device_identity_ignores_where_the_card_sits() {
        let card = |bus: &str, device: u32, driver_info: &str| wgpu::AdapterInfo {
            name: "NVIDIA L4".into(),
            vendor: 0x10de,
            device,
            device_type: wgpu::DeviceType::DiscreteGpu,
            device_pci_bus_id: bus.into(),
            driver: "NVIDIA".into(),
            driver_info: driver_info.into(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        };
        let here = card("0000:80:02.0", 0x27b8, "610.57.04");
        assert_eq!(device_identity(&here), device_identity(&card("0000:00:07.0", 0x27b8, "610.57.04")));
        assert_ne!(device_identity(&here), device_identity(&card("0000:80:02.0", 0x2bb5, "610.57.04")));
        assert_ne!(device_identity(&here), device_identity(&card("0000:80:02.0", 0x27b8, "610.58.01")));
        let mut other = here.clone();
        other.name = "NVIDIA L40S".into();
        assert_ne!(device_identity(&here), device_identity(&other));
        let mut other = here.clone();
        other.backend = wgpu::Backend::Gl;
        assert_ne!(device_identity(&here), device_identity(&other));
    }

    #[test]
    fn optional_persistence_reservations_bound_bytes_coalesce_and_release() {
        let admission=std::sync::Arc::new(PersistenceAdmission {state:Default::default(),capacity:100});
        let key=Path::new("one");
        let first=admission.reserve(key,60).unwrap();
        assert!(admission.reserve(key,1).is_none());
        assert!(admission.reserve(Path::new("two"),41).is_none());
        let second=admission.reserve(Path::new("two"),40).unwrap();
        assert_eq!(admission.state.lock().unwrap().bytes,100);
        drop(first);
        assert!(admission.reserve(key,60).is_some());
        drop(second);
        assert_eq!(admission.state.lock().unwrap().bytes,0);
        let lock=admission.state.lock().unwrap();
        assert!(admission.reserve(key,1).is_none(),"admission must not block on contention");
        drop(lock);
        assert!(admission.reserve(key,100).is_some());
    }

    #[test]
    fn mapped_payload_rejects_corrupt_and_truncated_entries() {
        let root = std::env::temp_dir().join(format!("vork-mapped-test-{}", std::process::id()));
        let path = root.join("test.entry");
        write(&path, b"pixels").unwrap();
        assert_eq!(&*read_mapped(&path).unwrap(), b"pixels");
        fs::write(&path, [0; blake3::OUT_LEN + 6]).unwrap();
        assert!(read_mapped(&path).is_none());
        fs::write(&path, [0; blake3::OUT_LEN - 1]).unwrap();
        assert!(read_mapped(&path).is_none());
        fs::remove_dir_all(root).unwrap();
    }

    /// Entries are published without a device flush, so after a power loss a renamed entry can be
    /// empty, cut short, or its full length of zeros. Every loader reads each as a miss, never as
    /// an error or as content, and the next publication replaces it.
    #[test]
    fn entry_torn_by_power_loss_is_a_miss_for_every_loader() {
        let root = std::env::temp_dir().join(format!("vork-torn-entry-{}", std::process::id()));
        let path = root.join("driver.entry");
        let payload: Vec<u8> = (0..4096u32).flat_map(u32::to_le_bytes).collect();
        let whole = blake3::OUT_LEN + payload.len();
        let torn: [(&str, Box<dyn Fn(&Path)>); 5] = [
            ("empty", Box::new(|path: &Path| fs::write(path, b"").unwrap())),
            ("digest cut short", Box::new(|path: &Path| fs::OpenOptions::new().write(true).open(path).unwrap().set_len(blake3::OUT_LEN as u64 - 1).unwrap())),
            ("digest without payload", Box::new(|path: &Path| fs::OpenOptions::new().write(true).open(path).unwrap().set_len(blake3::OUT_LEN as u64).unwrap())),
            ("payload cut short", Box::new(move |path: &Path| fs::OpenOptions::new().write(true).open(path).unwrap().set_len(whole as u64 / 2).unwrap())),
            ("zeros of the full length", Box::new(move |path: &Path| fs::write(path, vec![0; whole]).unwrap())),
        ];
        for (case, tear) in &torn {
            assert!(write_entry(&path, &payload).unwrap());
            assert_eq!(read_pipeline(&path).as_deref(), Some(&payload[..]));
            tear(&path);
            assert!(read(&path).is_none(), "{case}: read");
            assert!(read_pipeline(&path).is_none(), "{case}: pipeline blob");
            assert!(read_mapped(&path).is_none(), "{case}: mapped");
            assert!(read_bounded(&path, u64::MAX).is_none(), "{case}: bounded");
            // The known-pipeline ledger loads through the same check: no key is trusted.
            assert!(crate::pipelines::Known::load(path.clone(), true).is_empty(), "{case}: ledger");
        }
        assert!(write_entry(&path, &payload).unwrap());
        assert_eq!(read(&path).as_deref(), Some(&payload[..]));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn portable_artifacts_share_content_across_builds_and_devices() {
        let a = Store {
            root: PathBuf::from("cache"),
            identity: blake3::hash(b"build A device A"),
        };
        let b = Store {
            root: a.root.clone(),
            identity: blake3::hash(b"build B device B"),
        };
        let content = blake3::hash(b"image");
        for kind in [
            "decoded-texture",
            "native-decoded-texture",
            "compressed-texture",
        ] {
            assert_eq!(a.path(kind, 2, content), b.path(kind, 2, content));
        }
        assert_eq!(
            a.path("slang-variant", 1, content),
            b.path("slang-variant", 1, content)
        );
        for kind in ["slang-program", "slang-resolve", "aov-resolve"] {
            assert_eq!(a.path(kind, 1, content), b.path(kind, 1, content));
        }
        assert_ne!(a.path("pipeline", 1, content), b.path("pipeline", 1, content));
        assert_ne!(
            a.path("slang-variant", 1, content),
            a.path("slang-variant", 1, blake3::hash(b"changed shader"))
        );
    }

    /// The bundle's writer is scripts/shader_bundle.py (Python's lzma: LZMA2 at
    /// preset 6 with CRC64), not this crate's encoder, which stores chunks
    /// unpacked and unchecked. The fixture is that writer's output for a 3 MiB
    /// entry, so this reads compressed chunks, a chunk boundary and the check.
    #[test]
    fn a_bundle_entry_written_by_the_release_exporter_is_read() {
        const NAME: &str = "5555555555555555555555555555555555555555555555555555555555555555.slang-variant-v1.entry";
        let payload: Vec<u8> = (0..3u64 << 20).map(|i| if i % 4096 < 4000 { ((i * i >> 7) ^ (i >> 3)) as u8 } else { 0 }).collect();
        let root = std::env::temp_dir().join(format!("vork-exported-shader-{}", std::process::id()));
        let (cache, shaders) = (root.join("cache"), root.join("shaders"));
        fs::create_dir_all(&cache).unwrap();
        fs::create_dir_all(&shaders).unwrap();
        let packed = include_bytes!(concat!("../tests/fixtures/shipped-shader/", "5555555555555555555555555555555555555555555555555555555555555555.slang-variant-v1.entry.xz"));
        fs::write(shaders.join(format!("{NAME}.xz")), packed).unwrap();
        assert_eq!(read_with_shaders(&cache.join(NAME), Some(&shaders)).as_deref(), Some(payload.as_slice()));
        // The same valid stream under another entry's name fails that entry's digest.
        let other = NAME.replace('5', "6");
        fs::write(shaders.join(format!("{other}.xz")), packed).unwrap();
        assert!(read_with_shaders(&cache.join(&other), Some(&shaders)).is_none());
        // A damaged stream is a miss.
        let mut damaged = packed.to_vec();
        let middle = damaged.len() / 2;
        damaged[middle] ^= 0x55;
        let _ = fs::remove_file(cache.join(NAME));
        fs::write(shaders.join(format!("{NAME}.xz")), damaged).unwrap();
        assert!(read_with_shaders(&cache.join(NAME), Some(&shaders)).is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn shipped_shader_fallback_checks_integrity_and_excludes_device_state() {
        let root = std::env::temp_dir().join(format!("vork-shipped-shaders-{}", std::process::id()));
        let store = Store::for_test(root.join("cache"));
        let shaders = root.join("shaders");
        for kind in ["slang-variant", "slang-program", "slang-resolve", "aov-resolve", "slang-closed-input", "slang-input", "pipeline", "pipeline-keys", "shader-warm"] {
            let version = if matches!(kind, "slang-resolve" | "slang-program") {2} else {1};
            let path = store.path(kind, version, blake3::hash(b"fixture"));
            // A shipped asset is the store entry, packed.
            let entry = shaders.join(path.file_name().unwrap());
            write(&entry, b"shipped").unwrap();
            let shipped = shaders.join(format!("{}.xz", path.file_name().unwrap().to_str().unwrap()));
            let mut packed = Vec::new();
            lzma_rs::xz_compress(&mut fs::read(&entry).unwrap().as_slice(), &mut packed).unwrap();
            fs::write(&shipped, &packed).unwrap();
            fs::remove_file(&entry).unwrap();
            let portable = matches!(kind, "slang-variant" | "slang-program" | "slang-resolve" | "aov-resolve" | "slang-closed-input");
            assert_eq!(read_with_shaders(&path, Some(&shaders)), portable.then(|| b"shipped".to_vec()));
            write(&path, b"local").unwrap();
            assert_eq!(read_with_shaders(&path, Some(&shaders)).unwrap(), b"local");
            fs::write(&path, b"corrupt").unwrap();
            assert_eq!(read_with_shaders(&path, Some(&shaders)), portable.then(|| b"shipped".to_vec()));
            // The asset was published to the store: the next read expands nothing.
            assert_eq!(read_bounded(&path, u64::MAX), portable.then(|| b"shipped".to_vec()));
            let _ = fs::remove_file(&path);
            // A stream that expands to other bytes fails the entry's digest.
            let mut other = Vec::new();
            lzma_rs::xz_compress(&mut [0u8; 64].as_slice(), &mut other).unwrap();
            fs::write(&shipped, &other).unwrap();
            assert!(read_with_shaders(&path, Some(&shaders)).is_none());
            fs::write(&shipped, b"corrupt").unwrap();
            assert!(read_with_shaders(&path, Some(&shaders)).is_none());
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn concurrent_process_budget_worker() {
        let Some(root) = std::env::var_os("VORK_CACHE_TEST_ROOT") else {
            return;
        };
        let root = PathBuf::from(root);
        for i in 0..24 {
            let path = root.join(format!("{}-{i}.entry", std::process::id()));
            let payload = vec![i as u8; 128];
            write(&path, &payload).unwrap();
            // Another writer may evict this entry, but a hit must be complete.
            if let Some(hit) = read(&path) {
                assert_eq!(hit, payload);
            }
        }
    }

    #[test]
    fn concurrent_processes_obey_budget_and_recover_interrupted_writer() {
        let root = std::env::temp_dir().join(format!("vork-budget-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("store.tmp"), b"interrupted write").unwrap();
        let mut children: Vec<_> = (0..4)
            .map(|_| {
                std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "disk_cache::tests::concurrent_process_budget_worker",
                    ])
                    .env("VORK_CACHE_TEST_ROOT", &root)
                    .env("VORK_CACHE_BYTES", "1024")
                    .spawn()
                    .unwrap()
            })
            .collect();
        for child in &mut children {
            assert!(child.wait().unwrap().success());
        }
        assert!(!root.join("store.tmp").exists());
        let bytes: u64 = fs::read_dir(&root)
            .unwrap()
            .flatten()
            .filter(|entry| entry.path().extension().is_some_and(|e| e == "entry"))
            .map(|entry| entry.metadata().unwrap().len())
            .sum();
        assert!(bytes > 0 && bytes <= 1024, "{bytes}");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn opened_reader_survives_eviction_and_hits_refresh_lru() {
        let root = std::env::temp_dir().join(format!("vork-reader-test-{}", std::process::id()));
        let store = Store {
            root: root.clone(),
            identity: blake3::hash(b"test"),
        };
        let old = store.path("texture", 1, blake3::hash(b"old"));
        let recent = store.path("texture", 1, blake3::hash(b"recent"));
        write(&old, b"old").unwrap();
        write(&recent, b"recent").unwrap();
        let mapped = read_mapped(&recent).unwrap();
        for path in [&old, &recent] {
            fs::OpenOptions::new()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH))
                .unwrap();
        }
        assert_eq!(read(&old).unwrap(), b"old");
        let mut opened = fs::File::open(&recent).unwrap();
        let files = [&old, &recent]
            .into_iter()
            .map(|path| {
                let meta = fs::metadata(path).unwrap();
                (meta.modified().unwrap(), meta.len(), path.clone())
            })
            .collect();
        store.trim_to_budget(files, 73, 35);
        assert!(old.exists());
        assert!(!recent.exists());
        let mut bytes = Vec::new();
        opened.read_to_end(&mut bytes).unwrap();
        assert_eq!(&bytes[blake3::OUT_LEN..], b"recent");
        assert_eq!(&*mapped, b"recent");
        // SMB defers the server-side unlink while a reader handle is open.
        drop(opened);
        drop(mapped);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pipeline_blob_survives_beyond_the_former_entry_cap() {
        let root = std::env::temp_dir().join(format!("vork-pipeline-large-{}", std::process::id()));
        let path = root.join("driver.entry");
        let payload = vec![7; 64 * 1024 * 1024 + 1];
        assert!(write_entry(&path, &payload).unwrap());
        assert_eq!(read_pipeline(&path).as_deref(), Some(payload.as_slice()));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn pipeline_budget_worker() {
        let Some(root) = std::env::var_os("VORK_PIPELINE_TEST_ROOT") else { return; };
        let path = PathBuf::from(root).join("driver.entry");
        write_entry(&path, b"previous").unwrap();
        assert!(!write_entry(&path, &[0; 1024]).unwrap());
        assert_eq!(read_pipeline(&path).as_deref(), Some(&b"previous"[..]));
    }

    #[test]
    fn pipeline_budget_failure_preserves_previous_blob() {
        let root = std::env::temp_dir().join(format!("vork-pipeline-budget-{}", std::process::id()));
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "disk_cache::tests::pipeline_budget_worker"])
            .env("VORK_PIPELINE_TEST_ROOT", &root).env("VORK_CACHE_BYTES", "128")
            .status().unwrap();
        assert!(status.success());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn atomic_writers_and_corrupt_entry_recovery() {
        let root = std::env::temp_dir().join(format!("vork-store-test-{}", std::process::id()));
        let store = Store {
            root: root.clone(),
            identity: blake3::hash(b"build + adapter"),
        };
        let hash = blake3::hash(b"source");
        let path = store.path("texture", 1, hash);
        assert_ne!(path, store.path("texture", 2, hash));
        assert_ne!(path, store.path("pipeline", 1, hash));
        assert_ne!(
            path,
            store.path("texture", 1, blake3::hash(b"different source"))
        );
        assert!(read(&path).is_none());
        write(&path, b"writer A").unwrap();
        std::thread::scope(|scope| {
            for payload in [b"writer A", b"writer B"] {
                let path = &path;
                scope.spawn(move || {
                    for _ in 0..16 { write(path, payload).unwrap(); }
                });
            }
            let path = &path;
            scope.spawn(move || {
                for _ in 0..128 {
                    if let Some(data) = read(path) {
                        assert!(data == b"writer A" || data == b"writer B");
                    }
                }
            });
        });
        let data = read(&path).unwrap();
        assert!(data == b"writer A" || data == b"writer B");
        let other = store.path("texture", 1, blake3::hash(b"other request"));
        fs::copy(&path, &other).unwrap();
        assert!(read(&other).is_none(), "a valid envelope from another key is a miss");
        assert!(read_mapped(&other).is_none());
        fs::remove_file(&other).unwrap();
        fs::OpenOptions::new().write(true).open(&path).unwrap().set_len(17).unwrap();
        assert!(read(&path).is_none(), "a truncated entry is a miss");
        write(&path, b"repaired after truncation").unwrap();
        let mut broken = fs::read(&path).unwrap();
        *broken.last_mut().unwrap() ^= 1;
        fs::write(&path, broken).unwrap();
        assert!(read(&path).is_none());
        // Process exit during a background publication leaves only an ignored
        // reservation. The next writer recovers it under the store lock.
        fs::write(root.join("store.tmp"), b"partial background artifact").unwrap();
        // So does a writer that ended between creating its temporary file and renaming it.
        fs::write(root.join("store-1-2-3.tmp"), b"partial background artifact").unwrap();
        assert!(read(&path).is_none());
        write(&path, b"recomputed").unwrap();
        assert_eq!(read(&path).unwrap(), b"recomputed");
        assert_eq!(fs::read_dir(&root).unwrap().count(), 2);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn eviction_is_shared_across_artifact_kinds() {
        let root = std::env::temp_dir().join(format!("vork-trim-test-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let store = Store {
            root: root.clone(),
            identity: blake3::hash(b"build + adapter"),
        };
        let old = store.path("pipeline", 1, blake3::hash(b"old"));
        let recent = store.path("hdr", 1, blake3::hash(b"recent"));
        for (path, age) in [(&old, 1), (&recent, 2)] {
            let file = fs::File::create(path).unwrap();
            // Sparse files exercise the real budget without writing a GiB.
            file.set_len(513).unwrap();
            file.set_times(
                fs::FileTimes::new()
                    .set_modified(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(age)),
            )
            .unwrap();
        }
        let files = [&old, &recent]
            .into_iter()
            .map(|path| {
                let meta = fs::metadata(path).unwrap();
                (meta.modified().unwrap(), meta.len(), path.clone())
            })
            .collect();
        store.trim_to_budget(files, 1026, 1024);
        assert!(!old.exists());
        assert!(recent.exists());
        fs::remove_dir_all(root).unwrap();
    }
}

/// Query the filesystem containing the one cache; unavailable accounting skips eviction.
#[cfg(unix)]
fn available_space(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: path is NUL-terminated and stat points to writable storage.
    if unsafe { libc::statvfs(path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    let stat = unsafe { stat.assume_init() };
    (stat.f_bavail as u64).checked_mul(stat.f_frsize as u64)
}
#[cfg(not(any(unix, windows)))]
fn available_space(_path: &Path) -> Option<u64> {
    None
}

#[cfg(windows)]
fn available_space(path: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            available: *mut u64,
            total: *mut u64,
            free: *mut u64,
        ) -> i32;
    }
    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut available = 0;
    // SAFETY: path is NUL-terminated; the output pointer is valid and optional outputs are null.
    let success = unsafe {
        GetDiskFreeSpaceExW(
            path.as_ptr(),
            &mut available,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    (success != 0).then_some(available)
}

/// Advise once per process; cache availability never controls rendering.
fn network_notice(root: &Path) {
    static NOTICE: std::sync::Once = std::sync::Once::new();
    let Ok(root) = std::path::absolute(root) else { return };
    let mut existing = root.as_path();
    while !existing.exists() {
        let Some(parent) = existing.parent() else { return };
        existing = parent;
    }
    if network_filesystem(existing) {
        NOTICE.call_once(|| eprintln!("vork: cache is on a network filesystem; consider a local VORK_CACHE_DIR for reliable, faster caching"));
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn network_filesystem(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else { return false };
    let mut info = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: statfs initializes info on success and path is NUL terminated.
    if unsafe { libc::statfs(path.as_ptr(), info.as_mut_ptr()) } != 0 { return false }
    let info = unsafe { info.assume_init() };
    #[cfg(target_os = "linux")]
    {
        // NFS, SMB/CIFS, SMB2, NCP, 9P, AFS, Ceph and Coda (linux/magic.h).
        matches!(info.f_type as u64, 0x6969 | 0x517b | 0xff534d42 | 0xfe534d42 |
            0x564c | 0x01021997 | 0x5346414f | 0x6b414653 | 0x00c36400 | 0x73757245)
    }
    #[cfg(target_os = "macos")]
    { info.f_flags & libc::MNT_LOCAL as u32 == 0 }
}

#[cfg(windows)]
fn network_filesystem(path: &Path) -> bool {
    #[link(name = "kernel32")]
    unsafe extern "system" { fn GetDriveTypeW(root: *const u16) -> u32; }
    use std::path::{Component, Prefix};
    let Ok(path) = std::path::absolute(path) else { return false };
    let Some(Component::Prefix(prefix)) = path.components().next() else { return false };
    let drive = match prefix.kind() {
        Prefix::UNC(..) | Prefix::VerbatimUNC(..) => return true,
        Prefix::Disk(drive) | Prefix::VerbatimDisk(drive) => drive,
        _ => return false,
    };
    let root: Vec<u16> = format!("{}:\\", char::from(drive)).encode_utf16().chain(Some(0)).collect();
    // SAFETY: root is a NUL-terminated drive root; this API retains no pointer.
    unsafe { GetDriveTypeW(root.as_ptr()) == 4 }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
fn network_filesystem(_: &Path) -> bool { false }
