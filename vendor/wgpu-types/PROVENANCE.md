# wgpu-types 29.0.4

Source: the unmodified crates.io `wgpu-types` 29.0.4 package, except for the
local change below and this file. Registry metadata `.cargo-ok` and
`.cargo_vcs_info.json` are omitted. Upstream: https://github.com/gfx-rs/wgpu.
Licensed MIT OR Apache-2.0; both license files are retained.

Local patch: `src/lib.rs`, `PollError::DeviceLost(String)`, returned by the
vendored `wgpu` `Device::poll` when the device is lost (vendor/wgpu/PROVENANCE.md).
Remove together with that patch.

Local declarative ABI extension: `BindingType::metal_resource_slots` owns native
slot consumption for buffers, textures, samplers, external textures and Tier-2
binding arrays. Metal HAL layout allocation and renderer ABI checks share it;
native Apple reflection remains the independent verifier. Remove together with
passthrough ABI consumers if upstream provides an equivalent contract.

Local native geometry extension: axis-aligned bounding-box BLAS descriptions and
build inputs use the existing Vulkan/Metal HAL AABB implementation. Core checks
geometry counts, compatible flags, stride/alignment, buffer range, usage and device
ownership, and tracks initialization and build-input transitions.
