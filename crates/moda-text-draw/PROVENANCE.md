# moda-text-draw provenance

Ported from Moda's `moda-engine` crate. The text drawing files, excerpted:
the vector glyph lane's plan step and the GPU side that draws its batches.



`src/lib.rs` and `Cargo.toml` are vork's (module list and dependencies).

## `src/text_draw.rs` (from `src/text_draw.rs`)

Copied unchanged: the module header and `model_min_axis_scale`, `VECTOR_DRAW_UNIFORM_BYTES`,
`vector_emoji_ops`, `emoji_module`, `TextBuildOutcome`, and the whole GPU side (`TextDraw`, `new`,
`ensure_pipeline`, `vector_emoji_bind_group`, `create_texture_bind_group`, `prepare`, `draw_batch`,
`create_filled_buffer`, `text_blend_factor`, `PreparedDraw`, `PreparedVectorDraw`, `PreparedBatch`,
`PreparedText`), except for these changes:

- Not ported: the plan side that reads Moda scene nodes (`build_richtext_item`, `build_shape_text_item`,
  `plan_connector_text`, the node field extractors, the residency cache
  keys), the process-wide text host and font-directory discovery (`HOST`, `set_fonts_dir`,
  `install_text_host`, `text_host`, `vector_store_and_host`, the `VECTOR_STORE` thread local), the JSON dumps and
  the tests. The imports of `moda_scene`, `RawNode`, `Scene`, `Value`, `parse_gradient_stops`, `GradientSpec`,
  `DegradedQuad` and `DrawItem` go with them.
- `try_build_vector_batch`: public; takes the vector font store and text host as arguments instead of reading
  the process-wide host; drops the `fill_shader` and `fall_label` arguments and the resource-report events
  (`moda_assets::report`: font substitution, vector lane served, fallback class); returns
  `Ok(Err(reason))` when the vector lane declines a node instead of recording the fallback and returning
  `Ok(None)` (a comment in that arm says so); builds the batch without `mask` and `fill_shader`.
- `vector_or_degraded` and `build_degraded_batch`: public; take the degraded engine as an argument instead of
  the process-wide `moda_text::degraded::engine()`, and `vector_or_degraded` the vector font store and text
  host (for `try_build_vector_batch`, called with `VectorParity` and without the fill shader and lane label).
  `build_degraded_batch` drops the camera-build branch (`degraded_screen_scale`, `render_node_on_screen` and the
  zoom-limited notes, `enter_degraded_camera` not ported) and always rasterises with `render_node_at`; it builds
  the batch without `mask` and `fill_shader`.
- `vector_emoji_bind_group` returns a `Result` and keeps only drawn cells: an emoji no face draws in colour, or a
  missing font engine, is an error from `prepare` naming the text node and the emoji (`emoji_image`, vork's),
  where Moda skipped the draw and cached the blank.
- `TextBuildOutcome::Item` holds the `Arc<TextQuadBatch>` instead of a boxed `DrawItem::Text`.
- Text shader fills and agent fills are not ported: the `shader_fill_frame_bgl`, `shader_fill_storage_bgl` and
  `vec_shader_curve_bgl` layouts, `layout_vec_shader_fill`, the `<id>.vector-text-fill` pipeline arm, the agent
  fill lookup in `ensure_pipeline`, the shader-fill instance arena and bind groups in `prepare`, the
  shader-fill branch of `draw_batch` and the `shader*` fields of `PreparedText` and `PreparedVectorDraw` are
  removed; a `VectorOpPipeline::AnalyticShaderFill` op is refused with "text shader fills are not ported".
  vork never asks for one (`TextNodeSpec::shader_fill` stays false). In `prepare`, the analytic arm's
  `let Some(curve_idx)` becomes `let Some(_curve_idx)`, its only other reader having been the shader-fill arm.
- The GPU profiler counters (`crate::gpu_profile::count_texture_write`, `count_buffer_write`) and the emoji miss
  report (`moda_assets::report::record_emoji_miss`) are removed; `draw_batch` takes a `wgpu::RenderPass`
  instead of `gpu_profile::CountedPass`.

## `src/seams.rs` (from `src/seams.rs`)

`TextQuadBatch` and `DegradedQuad`, unchanged except that `TextQuadBatch` drops the `mask` (`MaskRef`) and
`fill_shader` (`TextShaderFill`) fields, which belong to effects and shader fills that are not ported. A new
module header and a re-export of `moda_text::ops::{TextOp, TextOpPipeline}` replace the rest of the file.

## `src/depth.rs` (from `src/depth.rs`)

Vork's. Moda's `linear_depth` spreads 50,000 render-z slots over [0.1, 0.9] and has no lower bound, so slots past
56,250 take depths behind the near plane and are clipped. Vork's takes 40 levels of the 24-bit depth attachment
per slot from the same far depth, which orders 335,544 slots (`MAX_RENDER_Z`), and adds `ensure_render_z`, the
refusal past that. `LINEAR_DEPTH_MAX` and `SHAPE_DEPTH_RANGE` are gone; `stroke_depth_offset` is 5/8 of the new
slot, the ratio Moda's has to its slot. The order of any two depths is Moda's. Tests are vork's.

## `src/device.rs` (from `src/device.rs`)

`GpuContext` with only its `device` and `queue` fields, which are all the text drawing reads; the adapter
probing, sample-count and surface fields and the constructors are not ported.

Added by vork: `PipelineHost`, `set_pipeline_host`, `forget_pipeline_host`, `create_render_pipeline` and
`create_compute_pipeline`. The host of a device (vork's renderer) creates Moda's pipelines, so they use its
persistent pipeline cache; `src/text_draw.rs` `ensure_pipeline` creates its pipeline through them.

## `src/vec_arena_cache.rs` (from `src/vec_arena_cache.rs`)

Unchanged.

- `src/seams.rs`: `TextQuadBatch::content_key` names what a batch draws (the content key a plan cache keeps it
  under); the builders here leave it `None` and moda-engine sets it when it caches a batch.
- `src/seams.rs`: `TextQuadBatch::local_bounds` bounds the actual vector quads, glyph transforms and degraded raster padding for picture captures.
- `src/text_draw.rs`: `TextDraw::new` takes the owning engine's working colour (`moda_shaders::working_color`),
  compiles its text modules for it, and uploads a text raster's or an emoji's texture in it (half floats; the
  crate gains `half`).
