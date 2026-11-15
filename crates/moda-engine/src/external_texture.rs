//! External-image GPU upload seam.
//!
//! The wasm editor lane keeps decoded raster pixels OUT of the wasm linear
//! heap: the browser decodes an image to an `ImageBitmap` and the engine
//! uploads it straight to a GPU texture via `wgpu::Queue::
//! copy_external_image_to_texture` — the bytes go browser → GPU, never through
//! wasm's 4 GiB-ceilinged heap.
//!
//! That `copy_external_image_to_texture` call is `#[cfg(web)]` (it takes a
//! `web_sys::ImageBitmap`), so it cannot live in this cross-platform crate.
//! Instead the GPU layer defines this trait and calls it from the pattern
//! texture cache at BIND time (where the wgpu device is guaranteed to exist
//! and which survives engine swaps — matching the resilient process-global
//! `AssetStore` design); `moda-wasm` installs the wasm-only implementation
//! that owns the `ImageBitmap` registry and issues the actual copy.
//!
//! On native (the CLI golden lane) no uploader is installed and the store
//! never produces external entries, so this seam is inert.

use std::sync::{Arc, OnceLock, RwLock};

/// The result of an [`ExternalImageUploader::upload`] attempt — three-state so
/// the pattern-texture cache can tell a DELIBERATE skip apart from a genuine
/// absence and treat the (unwritten, transparent) texture accordingly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExternalUploadOutcome {
    /// The source was copied into the texture; cache it normally.
    Uploaded,
    /// The copy was deliberately skipped for THIS render (e.g. the wasm
    /// offscreen snapshot guard avoiding a panic-prone live-bitmap copy). The
    /// texture is unwritten (samples transparent); the cache must NOT durably
    /// retain it — it re-attempts the upload on the next render so a later live
    /// present of the same key renders the real pixels instead of the skipped
    /// placeholder.
    Skipped,
    /// No source was available (no uploader installed, or the registry has no
    /// bitmap for `key`). The texture is left transparent and cached as-is; the
    /// key recovers when its source is (re)installed under a fresh identity.
    Unavailable,
}

/// Uploads a browser-decoded image into a caller-created GPU texture.
///
/// The implementation looks the source up by `key` (the `AssetStore` key the
/// feeder installed it under) and copies it into `texture`, which the caller
/// has already sized to `width`×`height` with
/// `TEXTURE_BINDING | COPY_DST | RENDER_ATTACHMENT` usage (the dest of
/// `copy_external_image_to_texture` is implemented as a render pass and needs
/// `RENDER_ATTACHMENT`). Returns [`ExternalUploadOutcome::Unavailable`] when the
/// source is no longer available (the caller then bakes a transparent fill
/// rather than sampling uninitialized texels).
pub trait ExternalImageUploader: Send + Sync {
    fn upload(
        &self,
        key: &str,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> ExternalUploadOutcome;
}

/// True when a source of `src_w`×`src_h` can be copied into a `dst_w`×`dst_h`
/// texture without the browser's `copyExternalImageToTexture` throwing
/// (a detached/closed source reads 0×0; the copy extent must fit within the
/// source's current bounds).
///
/// The wasm uploader pre-validates with this before issuing the copy: wgpu's
/// webgpu backend `unwrap()`s the JS copy result, so a throw ("Failed to copy
/// content from external image") panics and — with `panic = "abort"` — poisons
/// the whole wasm instance (WEB-NEXTJS-198). The dest texture's extent is frozen
/// at install-time dims while the source (an `ImageBitmap`/`<video>` frame) can
/// shrink or close afterwards, so the fit must be re-checked at copy time.
pub fn external_copy_source_fits(src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> bool {
    src_w >= dst_w && src_h >= dst_h && dst_w > 0 && dst_h > 0
}

fn slot() -> &'static RwLock<Option<Arc<dyn ExternalImageUploader>>> {
    static SLOT: OnceLock<RwLock<Option<Arc<dyn ExternalImageUploader>>>> = OnceLock::new();
    SLOT.get_or_init(|| RwLock::new(None))
}

/// Install the process-global external-image uploader (wasm editor lane:
/// `moda-wasm` installs its `ImageBitmap`-backed implementation once). Idempotent.
pub fn install_external_uploader(uploader: Arc<dyn ExternalImageUploader>) {
    *slot().write().expect("external uploader lock") = Some(uploader);
}

/// The installed uploader, if any. `None` on the native lane (no external
/// textures), so callers fall back to their heap-bytes upload path.
pub fn external_uploader() -> Option<Arc<dyn ExternalImageUploader>> {
    slot().read().expect("external uploader lock").clone()
}

/// Remove the installed uploader (tests).
#[cfg(test)]
pub fn clear_external_uploader() {
    *slot().write().expect("external uploader lock") = None;
}

/// Serialize tests that replace the process-global uploader slot.
#[cfg(test)]
pub fn lock_external_uploader_for_test() -> std::sync::MutexGuard<'static, ()> {
    static TEST_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
    TEST_LOCK
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The slot only stores/returns the trait object; `upload` (which needs a
    // real wgpu device) is exercised in-browser, so a no-op impl is enough here.
    struct NoopUploader;
    impl ExternalImageUploader for NoopUploader {
        fn upload(
            &self,
            _key: &str,
            _device: &wgpu::Device,
            _queue: &wgpu::Queue,
            _texture: &wgpu::Texture,
            _width: u32,
            _height: u32,
        ) -> ExternalUploadOutcome {
            ExternalUploadOutcome::Uploaded
        }
    }

    // The slot is process-global. The `paint::draw` GPU tests also install an
    // uploader, so cross-test isolation relies on the crate's mandated
    // `--test-threads=1` GPU lane (Moda's agent instructions); this test additionally self-defends
    // by clearing the slot before it asserts.
    #[test]
    fn external_copy_source_fits_cases() {
        // Source at least as large as the dest (exact match and larger): fits.
        assert!(
            external_copy_source_fits(256, 256, 256, 256),
            "exact-size source fits"
        );
        assert!(
            external_copy_source_fits(512, 512, 256, 256),
            "larger source fits"
        );
        // Source shrank below the frozen dest extent (a video frame that dropped
        // resolution mid-render): the copy extent overruns the source → reject.
        assert!(
            !external_copy_source_fits(128, 256, 256, 256),
            "narrowed source rejected"
        );
        assert!(
            !external_copy_source_fits(256, 128, 256, 256),
            "shortened source rejected"
        );
        // Zero-dim dest: nothing to copy, and a 0×0 texture is degenerate → reject.
        assert!(
            !external_copy_source_fits(256, 256, 0, 256),
            "zero-width dest rejected"
        );
        assert!(
            !external_copy_source_fits(256, 256, 256, 0),
            "zero-height dest rejected"
        );
        // Detached/closed source reads 0×0 (the poison race) → reject.
        assert!(
            !external_copy_source_fits(0, 0, 256, 256),
            "detached 0x0 source rejected"
        );
    }

    #[test]
    fn install_get_clear_roundtrip() {
        let _test_lock = lock_external_uploader_for_test();
        clear_external_uploader();
        assert!(external_uploader().is_none(), "starts empty");

        install_external_uploader(Arc::new(NoopUploader));
        assert!(
            external_uploader().is_some(),
            "installed uploader is visible"
        );

        // Re-install replaces (idempotent slot, not an accumulate).
        install_external_uploader(Arc::new(NoopUploader));
        assert!(external_uploader().is_some());

        clear_external_uploader();
        assert!(external_uploader().is_none(), "cleared");
    }
}

// A graph-pixel unit handed to the scene renderer as a GPU texture (moda-cli graph_pixels): registered on the render
// thread under its own key for as long as its `CaptureTexture` lives, so the pool texture outlives the passes that
// sample it; nested captures have keys of their own.
thread_local! {
    static CAPTURES: std::cell::RefCell<std::collections::HashMap<String, crate::pixel::pool::PoolTexture>> = Default::default();
}

pub struct CaptureTexture {
    key: String,
    pub upload_id: u64,
}

impl CaptureTexture {
    pub fn new(texture: crate::pixel::pool::PoolTexture) -> Self {
        static SERIAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let upload_id = SERIAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = format!("{}gpu/{upload_id}", crate::effects::PIXEL_SINK_SRC_PREFIX);
        CAPTURES.with(|c| c.borrow_mut().insert(key.clone(), texture));
        Self { key, upload_id }
    }

    pub fn key(&self) -> &str {
        &self.key
    }
}

impl Drop for CaptureTexture {
    fn drop(&mut self) {
        // The registration may outlive the registry during thread teardown. Drop a
        // removed texture outside the registry borrow (it can release GPU resources).
        let removed = CAPTURES.try_with(|c| c.borrow_mut().remove(&self.key));
        drop(removed);
    }
}

pub(crate) fn capture_texture(key: &str) -> Option<wgpu::Texture> {
    CAPTURES.with(|c| c.borrow().get(key).map(|t| t.texture().clone()))
}

#[cfg(test)]
mod capture_teardown_tests {
    use super::*;
    thread_local! {
        static OWNER: std::cell::RefCell<Option<CaptureTexture>> = const { std::cell::RefCell::new(None) };
    }

    #[test]
    fn registration_outlives_thread_registry() {
        std::thread::spawn(|| {
            OWNER.with(|owner| *owner.borrow_mut() = Some(CaptureTexture { key: "teardown".into(), upload_id: 0 }));
            CAPTURES.with(|_| ());
        }).join().unwrap();
    }
}
