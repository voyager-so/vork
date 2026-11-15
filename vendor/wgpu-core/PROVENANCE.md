# wgpu-core 29.0.4

Source: the crates.io `wgpu-core` 29.0.4 package, except for the changes below
and this file. Registry metadata `.cargo-ok` and `.cargo_vcs_info.json`, plus unused
`Cargo.lock` and `Cargo.toml.orig`, are omitted. Upstream: https://github.com/gfx-rs/wgpu. Licensed MIT OR Apache-2.0;
both upstream license files are retained.

Local patch: a typed `SubmissionBoundary` encoder command and matching native
recording entry point. Replay closes the current native buffer. CPU validation and resource tracking continue
in the same core encoder. Trace recording retains the marker. Vulkan submits all
resulting buffers together; this marker no longer creates a native queue submit.
Final-fence and semaphore ownership cover the whole HAL submission.

This avoids one validated core encoder and allocator per cold sample. It does
not estimate unmeasured workloads, inject waits, or reset any resources.

Touched upstream files: `src/command/encoder_command.rs` (typed marker),
`src/command/mod.rs` (record/replay), and `src/device/trace/record.rs` (trace marker).

Local error-reporting extension: propagate HAL `LostWithReason` through the
existing device-loss callback before mapping the public error to device loss.
Queue teardown reports a failed drain (including exhausted timeout retries)
through that path instead of panicking. It quarantines the native queue,
pending writes, in-flight submissions and parent device until process exit:
an aborted Metal buffer does not prove that later buffers released their
unretained resources. Successful drains retain the original destruction order.
The paired HAL patch checks every native buffer before fence completion; frame
collection therefore fails before reading partial pixels or GPU timestamps.

Local GPU placement extension: native TLAS descriptor buffers enter the existing tracked
full-build path with an explicit retained BLAS reference set. Core validates range,
alignment, capacity, usage and device ownership, records initialization and storage/AS
transitions, and retains dependencies through command replay and submission. Trace conversion retains the native buffer and reference set, but recording native
builds in portable traces is refused because native references cannot survive resource
reallocation. Query submissions retain their exact BLAS dependency snapshot even
when later TLAS builds replace the reference set. Metal residency declarations
use that same submission tracker rather than re-reading mutable TLAS dependencies. Native templates and transform/mask
layout facts come from the Metal/Vulkan backend; shaders never synthesize addresses.
No raw-HAL renderer bypass, indirect build or update/refit mode is introduced. Existing
Metal AS fence ownership is preserved. Remove when upstream supplies an equivalent
tracked descriptor-buffer build and backend layout/template contract. Regressions:
core native descriptor range tests, renderer GPU packing/error tests and the real-device
build/query/rebuild lifetime contract in `scripts/check-metal-contracts`.

Local release preparation extension: `src/device/resource.rs` accepts SPIR-V
or MSL passthrough modules on the no-op backend. No-op consumes neither code
representation and retains passthrough's unknown interface. This lets the
release shader catalogue use normal scene/pipeline construction without a GPU
or driver. Vulkan and Metal admission/compilation are unchanged. Regression:
GPU-free `scripts/prepare-shaders`, followed by the installed archive on L4.
Remove when upstream accepts native shader descriptors on no-op.

Local patch: Metal pipeline caches. `src/pipeline_cache.rs` gives Metal an adapter
key from the device name (Metal reports no PCI ids), and `CreatePipelineCacheError::Rejected`
(`src/pipeline.rs`, `src/device/resource.rs`) carries a backend's refusal of cache
data as a validation error without losing the device. Regression: wgpu-hal
`metal::device::binary_archive_tests`. Remove with the wgpu-hal Metal binary archive patch.

Local native geometry extension: axis-aligned bounding-box BLAS descriptions and
build inputs use the existing Vulkan/Metal HAL AABB implementation. Core checks
geometry counts, compatible flags, stride/alignment, buffer range, usage and device
ownership, and tracks initialization and build-input transitions.
