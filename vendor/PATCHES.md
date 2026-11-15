# Local changes to vendored crates

vork vendors patched copies of these crates; the edits are in the vendored source and git history.
Each row says why a patch exists and when it can be removed.

| Patch contract | Packages / source | Regression / removal condition |
| --- | --- | --- |
| No-op passthrough shaders and mock TLAS descriptors | wgpu-core device/resource.rs; wgpu-hal noop/mod.rs | GPU-free `scripts/prepare-shaders` and installed archive L4 acceptance; remove when upstream accepts native descriptors on no-op |
| Actual Vulkan buffer memory properties | wgpu-hal, vulkan/mod.rs | Renderer `buffer_heap_tests`; remove when upstream exposes allocator-selected buffer memory metadata |
| Passthrough MSL invariance | wgpu-hal, metal/device.rs | `scripts/check-metal --gpu`; remove when upstream preserves invariance for passthrough MSL |
| Safe-math Naga MSL (WGSL modules) | wgpu-hal, metal/device.rs | `compute::tests::ray_techniques_match_authored_compute_instances` on Metal; remove when upstream compiles Naga MSL without fast math |
| Safe-math passthrough MSL uses precise library functions | wgpu-hal, metal/device.rs | `pipelines::tests::frame_pixels_do_not_depend_on_variant_compile_timing` on Metal; remove when upstream exposes per-module MTLCompileOptions math settings |
| All Metal completions checked; failed teardown quarantined | wgpu-hal, metal/{completion,mod,device}.rs; wgpu-core device/queue.rs | `metal::completion::tests`, gpu_memory tests; remove when upstream checks every buffer before fence success and reports failed drains without destroying in-flight resources |
| Native resource-limit error | wgpu-hal DeviceError::ResourceLimit and Metal pipeline layout; wgpu-core DeviceError conversion without device loss | `medium::tests::native_buffer_slot_limit_is_a_validation_error_not_device_loss` on Metal; remove when upstream reports per-stage Metal slot exhaustion as a validation error |
| Shadow device-loss reason extension | wgpu-hal DeviceError::LostWithReason and Metal wait; wgpu-core device error conversion, loss delivery and queue teardown | `scripts/check-metal-contracts`, gpu_memory tests; remove when upstream carries native failed-buffer error and command label to the device-loss callback |
| Declarative Metal slot consumption | wgpu-types binding.rs; wgpu-hal metal/device.rs | `native::tests` plus native reflection; remove when upstream exposes an equivalent shared representation contract |
| Public poll device loss | wgpu, wgpu-types | `gpu_memory::tests`; remove when upstream returns loss from poll |
| Duplicate intersection setter | wgpu-hal, metal/conv.rs | `rt::tests`; remove when upstream calls the setter |
| Metal buffer arrays / offsets / residency | wgpu-hal, metal/{adapter,device}.rs | `scripts/check-metal-contracts`, native_argument_prefix_tail_execution; remove when upstream supports Tier-2 buffer arrays |
| Declared argument capacity / initialized tail | wgpu-hal, metal/device.rs | `scripts/check-metal-contracts`, argument_buffer_tests; remove when upstream allocates declared capacity and initializes the tail |
| Flexible MSL argument tables use pipeline capacities | wgpu-hal, metal/{mod,device}.rs | `scripts/check-metal-contracts` small render and descriptor-capacity tests in both API-validation modes, argument_buffer_tests; remove when upstream specializes flexible passthrough tables from the stage's binding layout before Metal compilation |
| Metal pipeline cache as a binary archive | wgpu-hal metal/{adapter,device,mod}.rs, lib.rs PipelineCacheError::Rejected; wgpu-core pipeline_cache.rs Metal adapter key, CreatePipelineCacheError::Rejected | `metal::device::binary_archive_tests` (via `scripts/check-metal-contracts`), `tests::shader_cache_retries_failed_driver_publication` on Metal; remove when upstream persists Metal pipelines through PipelineCache |
| Metal compile admission | wgpu-hal metal/{compile_admission,device,mod}.rs | `metal::compile_admission::tests` (via `scripts/check-metal-contracts`) and a fresh-cache render of a scene with many large material pipelines on Metal; remove when Metal's compiler service bounds its own memory across a client's concurrent compilations |
| Counter resolve before reuse | wgpu-hal, metal/command.rs | `frame::profiling_tests` on Metal; remove when upstream orders both counter dependencies |
| Tracked GPU-written TLAS descriptors | wgpu API; wgpu-core commands/trace; wgpu-hal native descriptor layout | `gpu_tlas::tests`, native descriptor build/query/rebuild contracts, core range tests; remove when upstream provides equivalent tracked buffers, dependency retention and backend ABI authority |
| Submitted BLAS/TLAS update selection | wgpu-core ray_tracing, resource and queue; existing Vulkan UPDATE / Metal refit HAL paths | `refit_generation_tests`, `refits_recorded_ahead_of_submission_preserve_query_results`, `native_blas_refit_tracks_deformation_and_discarded_builds`, `metal_contract_blas_refit_tracks_deformation`; local implementation of `PreferUpdate`, per-command full-build override for native TLAS buffers, immutable index topology, successful-submit source tracking, an earlier compatible build in the same recording as the update source (`dirty_samples_in_one_recording_refit_the_preceding_build`), submission-local compatibility validation across deferred buffers, and update scratch sizing. Upstream reference: [wgpu ray-tracing tracking issue #1040](https://github.com/gfx-rs/wgpu/issues/1040), not a claim that an upstream fix exists. Remove only after equivalent update and failed/discarded submission contracts pass on both backends |
| AS build/query/rebuild ordering | wgpu-hal, metal/{adapter,mod,command}.rs | `scripts/check-metal-contracts`, both cross-buffer AS tests; remove only with equivalent native dependency lowering; never restore the no-op AS barrier |
| Bounded pass segments | wgpu API command_encoder.rs, device.rs | `command_finish::tests::recordings_with_more_passes_than_native_admission_complete_in_order` on Metal; remove when upstream Metal no longer needs a native command buffer per pass before submission |
| Typed recording boundaries / merged Vulkan submission | wgpu API; wgpu-core commands/trace; wgpu-hal Vulkan queue | `command_finish::tests` and repeated L4 clips with sync validation; retain core barriers and one native submit per HAL call, never per-cut relay chains |
| Vulkan read-only depth/stencil stores | wgpu-hal Vulkan adapter/device | Read-only depth cut regression and sync validation; remove when upstream uses STORE_OP_NONE with matching capability admission |
| Omitted HAL examples and example dependencies | wgpu-hal Cargo manifests | `cargo check --workspace --all-targets --locked`; retain while vendoring library-only |
| objc2-metal `MTLFence` feature declared | wgpu-hal Cargo manifest (its Metal backend imports `MTLFence`) | a build where wgpu-hal is compiled without vork-render's objc2-metal features, such as vork-cli's build script through vork-std and moda-engine; retain while the Metal backend uses `MTLFence` |
| Custom header attributes written in name order | exr meta/header.rs `all_named_attributes` | `png_output::exr_tests::multilayer_and_png_control_units_round_trip` (two encodes of one picture are the same bytes; every EXR carries `vork:color_space` and `vork:reference_domain`); remove when upstream writes custom attributes in a fixed order |
| DWA chunk count admission | exr compression/dwa | vendored DWA tests and CLI malformed-image tests; remove when upstream bounds counts before allocation |
| Format-3 sequence context `would_apply` (Moda's patch) | harfrust 0.5.2 hb/ot/contextual.rs; see harfrust/MODA-PATCH.md | Bengali ra-phala in `scenes/greenlight/text/text-bengali-*.json` and `vork-std` std/text tests; remove when cosmic-text takes harfrust 0.14 or later |
| Optional Basis statistics / UASTC hint work | basisu_c_sys encoder/wrapper | compressed_textures tests; remove when upstream offers equivalent opt-outs preserving final ASTC/BC7 blocks |
| Collision sorts that cannot panic (Moda's patch, copied with moda-scene) | flo_curves bezier/path/ray.rs, graph_path/path_collision.rs; see flo_curves/MODA-PATCH.md | moda-scene boolean and Pathfinder paths; remove when upstream sorts collisions without a non-total comparator |

Upstream trackers: [wgpu](https://github.com/gfx-rs/wgpu/issues),
[exrs](https://github.com/johannesvollmer/exrs/issues),
[basisu_c_sys](https://github.com/beicause/basisu_c_sys/issues).
These are local extensions/fixes; no unverified upstream issue numbers are implied.
Native tests require actual hardware; a successful reapplication proves source
reproducibility, not GPU correctness. Upgrades must pass both checks.
