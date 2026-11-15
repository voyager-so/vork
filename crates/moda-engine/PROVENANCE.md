# moda-engine provenance

Ported from Moda's `moda-engine` crate: the 2D engine vork's canvas draws with (frame walk, tessellation, paints,
masks, effects, layer captures, the two-phase GPU executor and readback).



`src/` is copied unchanged except for the changes below. Moda's text drawing (its GPU side and the vector lane's
batch builder) was already ported once, as `crates/moda-text-draw`; this crate uses that copy rather than a
second one, and keeps the rest of Moda's text planning (`text_draw.rs`'s plan side, `text_area.rs`,
`text_metrics.rs`, `text_geometry.rs`, `text_extract.rs`, `text_warp.rs`) unchanged.

## Changes

- `src/scratch.rs`: destructor-time buffer/texture cleanup uses fallible TLS access and
  borrowing; unavailable scratch state leaves release to wgpu handle ownership.
- `src/external_texture.rs`: capture registration cleanup tolerates an already-destroyed
  TLS registry and releases removed textures outside its borrow. Thread-exit regression
  tests cover both cleanup paths.

- `Cargo.toml`: `edition = "2021"` written out (vork's workspace is edition 2024); no `rust-version`;
  `license.workspace = true`; `publish = false`; `wgpu = "29"` (vork's patched wgpu 29) instead of `=29.0.3`;
  `lyon_tessellation = "1"` instead of `=1.0.22` (vork's lockfile resolves 1.0.23, which vork-std already used);
  the `moda-text-draw` dependency added; the `jpeg-parity` feature and its `mozjpeg` dependency (a C build) are
  not ported, so JPEG encodes through the image crate, as Moda builds without the feature do (its `cfg` stays
  declared under `[lints.rust]`); the `moda-paint` and `ttf-parser` dev-dependencies are omitted with the
  integration tests that use them. Unit tests and doctests are enabled. `features.default` is `["native-io"]`.
- `src/lib.rs`: `pub mod depth;` and `pub mod vec_arena_cache;` become re-exports of moda-text-draw's copies
  (`pub use moda_text_draw::{depth, vec_arena_cache}`); `src/depth.rs` and `src/vec_arena_cache.rs` are not
  ported (moda-text-draw holds the same files).
- `src/seams.rs`: `TextQuadBatch` and `DegradedQuad` are re-exported from moda-text-draw
  (`pub use moda_text_draw::seams::{TextQuadBatch, DegradedQuad}`), whose `TextQuadBatch` has no `mask` and no
  `fill_shader` field (text masks and text shader fills are not ported). The rest of the file is unchanged.
- `src/text_draw.rs`:
  - The GPU side (`UNIFORM_ALIGN`, `TextDraw` and its impl, `create_filled_buffer`, `text_blend_factor`,
    `PreparedDraw`, the `VEC_FLAVOR_*` constants, `VectorDrawKind`, `PreparedVectorDraw`, `PreparedBatch`,
    `PreparedText` and its impl) is replaced by `pub use moda_text_draw::text_draw::{PreparedText, TextDraw};`.
    `VECTOR_DRAW_UNIFORM_BYTES`, `vector_emoji_ops` and `emoji_module`, which only it and the batch builder used,
    are removed with it, as is the `GpuContext` import and the now unused `HashMap`, `ByteLruCache` and `TextOp`
    imports. The unit tests (`mod tests`) are removed.
  - `try_build_vector_batch` keeps its signature and its font-substitution, served and fallback reports, and
    builds the batch with moda-text-draw's `try_build_vector_batch` (the same code, taking the store and host);
    a text shader fill is refused with "text shader fills are not ported".
  - `build_degraded_batch` keeps its signature and builds the raster batch with moda-text-draw's
    `build_degraded_batch` (the same code, ported there, taking the degraded engine) on Moda's process-wide
    degraded engine; its camera-build branch (screen-sized rasters and the zoom-limited notes) goes, as there.
  - `patch_cached_batch` no longer sets `fill_shader`; `dump_batches_json` drops the `fillShader` entry; `greek_font_px` no longer reads `mask` and
    `fill_shader`.
  - `TextBuilder`, the `TEXT_BUILDER` thread local and `install_text_builder` are added: a host that lays out its
    own text installs a builder, and `build_richtext_item` hands it every richtext node with the model matrix it
    computes (`richtext_model_matrix`), z, effective clip and render scale. vork's canvas installs one: its text
    is laid out by vork from its own nodes and drawn by moda-text-draw.
  - The host richtext builder uses the editor's text cache key, lookup, placement patch and bounded
    storage; the host records its external spec revision on the node so content changes re-key it.
- `src/frame.rs`:
  - A GPU thread-exit regression retains frame buffers and textures past scratch TLS teardown.
  - `Engine::set_reuse_frame_targets` lets an offscreen caller retain GPU caches while using the existing
    scratch-target path (fresh resolve, reused MSAA/depth). Vork's canvases embedded in 3D worlds use it
    because their consumers can be submitted after subsequent canvas renders; ordinary canvas frames
    reuse the retained target in queue order, as before in Moda.
  - `Engine::new` builds moda-text-draw's `TextDraw` with a moda-text-draw `GpuContext` (the device and queue);
    `Engine::text_gpu` (new) gives that context to the two `TextDraw::prepare` calls.
  - The transparent bucket loop draws text through moda-text-draw's `draw_batch`, which takes a
    `wgpu::RenderPass`: a `CountedPass::Pass` passes its pass; a `CountedPass::Bundle` (recorded only on the
    retained camera lane, which vork does not use) is unreachable.
  - `unmask_captured`, `offset_item_mask` and `mask_subtree_shadows` drop their arms for a text batch's mask.
  - A drawn item at a render-z past what depth orders is refused, not warned about: `build_coverage_plan_inner`
    and `build_canvas_root_items` (now a `Result`) call `depth::ensure_render_z` on `drawn_z_end` (new) of their
    items; the camera plan's "near/past the
    depth budget" warning and `CameraPlanAux::between` are removed. A brush's placed shapes draw in the
    transparent phase (painter's order), as paint layers do.
- `src/camera_slot.rs`: `group_has_masks`, `item_mask_id`, `set_item_mask_id` and the plan digest read no text
  batch mask. The slot holds no between-segment warnings; a splice that leaves a drawing unit past
  `depth::MAX_RENDER_Z` falls back to a full build (which refuses it).
- `src/residency.rs`: `batch_bytes` counts no text shader fill; `items_use_shader_time` and
  `items_shader_time_rethreadable` treat a text batch as reading no clock; `drawitem_mask` reads no text mask.
- Comments only, for vork's public-source export checks: `external_texture.rs` (a test comment names Moda's agent
  instructions rather than their file), `export_job.rs` and `residency.rs` (the export caches' study is not named),
  `gpu_profile.rs` and `tess/tessellator.rs` (their references to Moda's examples, which are not ported, say so).
- Not ported: integration test targets in `tests/`, `examples/`, `SEAMS.md`; the unit-test fixtures below are restored.
- Every `create_render_pipeline` and `create_compute_pipeline` call (23 sites: `src/frame.rs`, `src/isolated.rs`,
  `src/convert.rs`, `src/surface.rs`, `src/nv12.rs`, `src/motion_cache.rs`, `src/backdrop/{blend,exec}.rs`,
  `src/layer_blur/exec.rs`, `src/pixel/pass.rs`, `src/solid/render.rs`, `src/effects/{mask,mask_filter}.rs`) goes
  through `moda_text_draw::device::create_render_pipeline` / `create_compute_pipeline`, so the host's pipeline
  creation (vork's persistent pipeline cache) serves them. The descriptors are unchanged.

- `src/tess/tessellator.rs`: bound dash expansion work before the arc-length walk (including
  gaps and runs too short to emit vertices); return excessive expansion through `stroke_dropped`
  for paths and vector networks instead of hanging. The retained upstream tests unwrap the
  now-fallible dash expansion.

- `src/text_draw.rs`, `src/text_area.rs`, `src/text_geometry.rs`, `src/text_metrics.rs`:
  adapt text-spec construction and mutation to shared HTML and per-code-point arrays.

- `src/frame.rs`, `src/backdrop/blend.rs`: a layer capture takes the fade pairs among its items (a faded
  group's `BlendBegin` / `StackEnd`, a paint-stack group's `GroupBegin` / `StackEnd`) instead of refusing the
  filter: `retarget_isolation_splits` re-aims each pair's region at the capture's own pixels, in
  `divert_subtree_filter` and in the leaf divert. `split_pair_in` keeps a pair that lands nowhere on the page
  target, with an empty region the executor skips, so a capture that reaches it still has it. vork refuses a
  frame Moda warns about, so a blurred group with a fading child could not render.
- `src/frame.rs` `capture_cannot_hold`: one rule says what a layer capture holds, for a leaf's own items and
  a filtered subtree alike, through nested captures: what is exact in a target that starts transparent and
  has its own pixels (fade pairs, normal and screen bodies, source-over, screen and add composites). A
  multiply body or composite, a blend split, a backdrop effect and a page fill decline the filter with a
  warning. Moda captured a fixed-function multiply inside a filtered group (it vanishes over the capture's
  empty parts) and a leaf's blended paint layer (its split kept the page's region). Blend pairs planned for
  the target they land in would lift this; vork's canvas writes no blend mode.
- Filter shadows (`filter-shadow`, `FilterShadow` in moda-scene): CSS `filter: drop-shadow()` as a layer
  filter. The layer's captured picture casts the shadow from its own alpha, where a `drop-shadow` (Figma's) is
  cast from the opaque silhouette. `src/layer_blur/plan.rs` pads the capture by the shadows' reach, writes
  them as `LayerFilters::drop_shadows` (`src/seams.rs`), takes a leaf's drawn bounds when it has no box, and
  never declines a shadow for its size. With no other filter and no capture among the layer's items (`LayerFilters::shadows_alone`) the two divert
  sites in `src/frame.rs` capture a copy of the layer's items and put the shadows' composite before the items,
  which draw directly; under another filter the capture is the picture and `src/layer_blur/exec.rs` draws it
  over its shadows first (`shadows_of`, which also draws a shadow group's shadows as before). Such a capture
  takes the shadow groups' sample count; `src/cull.rs` counts the reach; `src/residency.rs` lists the type
  with the layer filters. vork's canvas writes every CSS drop shadow that way. `drop-shadow` and the lanes
  that draw it are unchanged.
- `src/frame.rs`, the two divert sites, with `src/effects/mask.rs` `clip_rounded_rect_until`: what a layer
  capture draws is cut by the clips inside its root and by the capture's own texture, never by the page or by
  a clipping frame above the root; those cut the composite. `widen_captured_clips` does it for every capture
  and keeps what an item clips by itself: a text batch's own box and a nested composite's rounded clip from a
  frame inside the root. Moda left a capture's items with the page's clip, so a blur did not spread what lies
  past the page edge back onto the page, and a filter that reads a descendant's capture (a shadow over a
  blurred child) lost the part of the child off the page. Still pruned by the page, as in Moda: a text wholly
  outside its clip is not built, a masked text's identity capture, a silhouette shadow group's window, and
  the captures of a fill's or stroke's own effects.
- `src/layer_blur/plan.rs` `params_for_bounds`, `readable_window`: a capture over the texture or area limit is
  first cut to what can be read from the target (the target's canvas rect grown by the read reach of every
  layer filter between the node's picture and the target), where that is sound: every filter in the chain is a
  uniform blur or a filter shadow (`cuts_cleanly`), the view is axis-aligned, and the window itself fits the
  limits at full scale, so no capture in the chain is reduced to fit them and every kernel runs at the size the
  reach measures. Elsewhere the whole capture is taken at a reduced scale, as in Moda, which did so always, so
  content far off the page softened a filtered group on it. A capture under the limits is not cut.
  `LayerBlurGroup::extent_canvas` (`src/seams.rs`) keeps what the picture occupies apart from the stored
  region, and `shadow::all_items_canvas_bounds` measures a nested capture by it, so an enclosing filter's
  bounds, and anything it places by them, do not follow a descendant's cut.
- `src/residency.rs`, `src/text_draw.rs` (with `TextQuadBatch::content_key` in moda-text-draw): a text batch a
  plan cache keeps carries its content key (`name_built_text`), and a capture's content key names a text batch
  by that key and its placement (`fold_item_geometry`) instead of its `Arc` address. A cache hit is a copy in a
  new allocation, so by address an unchanged text looked new on every rebuilt plan and a capture holding it was
  rendered again each time.
- `src/text_draw.rs`: `extract_node_effects` is crate-private again: vork's richtext builder no longer reads a
  text node's effects (a canvas text's filter is on a group around it).

## Unit-test restoration

The fixtures below are copied byte-for-byte from the upstream commit recorded above:

- `tests/fixtures/stress-500.json`: `native/fixtures/residency/stress-500.json`.
- `tests/fixtures/chart-bar.json`: `fixtures/parity/design/scenes/generated/chart-bar.json`.
- `tests/fixtures/fonts/`: `catalog.json`, `MANIFEST.json`, `families-index.json`,
  `inter/Inter_24pt-Regular.ttf`, `inter/Inter_24pt-ExtraBold.ttf`,
  `noto-sans/NotoSans-Regular.ttf`, `emoji/*.svg` and `emoji-noto.LICENSE` from
  `native/fixtures/oracle/fonts/`. Only the font faces exercised by these tests are included.
- Font licenses: `packages/editor/public/fonts/inter/OFL.txt` and
  `packages/editor/public/wasm-degraded-fonts/OFL-NotoSans.txt`, beside their respective fixture fonts.

Fixture paths in `frame.rs` and `residency.rs` point inside this crate. Test builds of
`text_draw::discover_fonts_dir` use the bundled directory before initializing the process-wide
host. Test-only calls before degraded-engine access (`text_draw.rs`, `effects/mask.rs`,
`boolean_operands.rs`) initialize the vector and degraded fixture font directories once, through
`test_fonts` in `lib.rs`. The `clip_bounds_scene` helper no longer mutates the environment during parallel tests.
Normal builds retain Moda's discovery. Test batch helpers omit the unported `mask` and
`fill_shader` fields, matching the production batch type already used here.

Removed tests requiring behavior vork does not ship:

- `frame::tests::prepared_text_survives_an_edit_that_leaves_text_alone` and
  `frame::tests::prepared_frame_hit_matches_full_encode`: retained-camera text render bundles.
  Vork draws text through moda-text-draw's render-pass API; the bundle arm is explicitly
  unreachable in this port. The other prepared-frame and split-plan tests remain.
- `residency::tests::batch_bytes_counts_shader_fill_payload`: text shader fills, already refused
  by the port. Only the text-shader-fill subcases of
  `mixed_plan_with_any_non_rethreadable_consumer_is_not_rethreadable` and
  `seed_gate_for_rethreadable_shader_is_superset_of_old_gate` are also removed; their body-fill
  and plain-text assertions remain.
- Only the text-mask assertion in `frame::tests::camera_plan_combines_pages_and_sets_projection`
  is removed, because text batches have no mask field here; body-mask and ancestor rebasing
  assertions remain.

`richtext_clip_text_to_bounds_narrows_only_when_flagged` and
`richtext_cache_hit_repatches_the_narrowed_clip` remain, using the bundled fonts.
Upstream ignored diagnostic/performance and gallery tests retain their ignored status.

- `src/text_draw.rs`, `src/effects/mask.rs`: host text bounds come from the installed builder's actual batch during rendered-mask planning, preserving overflowing and transformed glyphs; rendered captures align to the canvas pixel grid to avoid resampling their coverage.

- `src/frame.rs`, `src/layer_blur/exec.rs`, `src/residency.rs`: retain nested filtered
  subtrees as recursive captures, include their regions in cache keys, and diagnose
  subtree capture refusals instead of silently dropping filters.
- `src/layer_blur/plan.rs`: preserve every finite positive projected transform scale
  when sizing and validating layer filters.
- `src/layer_blur/plan.rs`: measure both projected singular stretches before
  blur visibility/limit checks; refuse visible anisotropic layer blur with a
  uniform-transform remedy instead of silently dropping a transformed axis.

- `src/solid/outline.rs`: flattening removes exact repeated points without an absolute pixel-space cleanup threshold; consumers own output-precision cleanup in their coordinate units.

- `src/frame.rs`, `src/effects/mask.rs`, `src/layer_blur/exec.rs`, `src/paint/draw.rs`, `src/surface.rs`,
  `src/residency.rs`: `Engine::new_with_color` makes an engine that blends in another encoding than sRGB
  (`moda_shaders::working_color`, re-exported as `working_color`). Authored colours stay sRGB and are converted
  where they enter: the engine, its float capture twin, its text, its filter shadows and its mask coverage
  compile their modules for the encoding; the pass's clear colour is converted; the pattern cache uploads a
  decoded image in the encoding, as half floats with a floating-point mip chain (`MipGen` takes the texture
  format; the floating-point filter covers any extent), from the image's straight colour where the asset store
  kept it (`DecodedImage::straight`); such an image past the drawing device's texture limit refuses the frame. `RenderOptions::working` names the encoding to the planner, which plans a luminance mask of such a plan as its
  rendered picture unless the mask is one layer of one solid colour (`PlanMasks::in_working_color`).
