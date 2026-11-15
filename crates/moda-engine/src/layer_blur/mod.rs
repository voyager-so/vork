//! layer_blur/ — the layer filters: layer blur (uniform or progressive),
//! noise and texture, which all post-process a node's OWN rendered output.
//!
//! Native-only (no TS-renderer counterpart — the WebGPU fallback renders the
//! node un-blurred), so unlike backdrop/ nothing here is byte-pinned to a
//! production TS source. Spec: `docs/roadmap/canvas/layer-blur-spec.md`.
//!
//! Split mirrors backdrop/:
//! - **plan** (`plan.rs`): eligibility (an active layer filter, no ancestor
//!   mask, normal blend — mirrored by the TS export gate
//!   `utils/export/layerBlurEligibility.ts`; a group or a frame with children
//!   captures its whole subtree, `frame.rs` `finish_subtree_capture`), padded-region math in
//!   canvas units, and the capture-scale reduction that keeps a uniform
//!   blur's sigma ≤ SIGMA_CAPTURE_MAX capture px for large radii.
//! - **exec** (`exec.rs`): per-group isolation capture via a recursive mini
//!   `CoveragePlan` render (transparent clear, sample_count 1), the filter
//!   passes in a fixed order — texture (`layer-filter-texture`), blur (the
//!   shared separable-Gaussian passes, `backdrop-blur` + `backdrop::kernel`'s
//!   `backdrop_blur_passes` ladder, into float targets,
//!   or two `progressive-blur` passes), every noise (`layer-filter-noise`) —
//!   and the engine-owned `layer-blur-composite` quad pipeline the main pass
//!   draws at the node's z. The grain is an integer hash of cell ids, so it is
//!   the same on every backend (`noise_cell_hash` is its CPU twin).
//! - **shadow** (`shadow.rs`): capture shadows — drop and inner shadows of
//!   every shape the SDF shadow lane can't draw, from a capture of the node's
//!   silhouette through the same pre-pass (`layer-filter-shadow` +
//!   `progressive-blur`); the result is the shadows alone, composited next to
//!   the node's own items rather than in place of them. A group's (or an
//!   unfilled frame's) silhouette is its whole subtree's
//!   (`shadow::is_subtree_capture_node`). Every shape's and text's glows ride
//!   the same capture: outer glows are offset-free drop passes shown behind,
//!   inner glows offset-free inner passes (from the edge, or the shape itself
//!   from the centre), each group composited with its glow mode (screen,
//!   normal, multiply or linear dodge; `seams::CaptureBlend`).
//!
//! Every composite is clipped like the node's body by the clipping frames
//! above it: their boxes (the effective clip) and the innermost rounded one's
//! corners (`effects::clip_rounded_rect`, in `layer-blur-composite.wgsl`).
//!
//! Unlike backdrop, layer blur is a PRE-pass (the mask-atlas slot), not a
//! mid-pass split: it reads nothing from the target, so pages keep their
//! MSAA sample count and `--transparent` exports work.

pub mod cache;
pub mod exec;
pub mod plan;
pub mod shadow;

pub use exec::LayerBlurRenderer;
pub use plan::{
    canvas_root_has_active_layer_filter, compute_layer_blur_params, has_active_layer_filter,
    page_has_active_layer_filter, LayerBlurParams,
};
