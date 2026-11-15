//! backdrop/ — backdrop effects (background blur / pixelate / dither / glass, and the filter kinds grain /
//! vignette / glaze / displace / grade: `BackdropOp::Filter`, one `backdrop-filter.wgsl` pass). OWNED BY M3e.
//!
//! Implements both sides of the SEAMS §7 pass-split contract:
//! - **plan** (`plan.rs`): eligibility + target-pixel param mapping with the
//!   production clamps (blur ≤ 200, pixelate ≤ 400, dither cell ≤ 256),
//!   sampling-region math (visible rect ∩ clip, expanded
//!   `ceil(blur+refr+disp)+1` for glass / `ceil(blur)` for background-blur / 0 for
//!   pixelate+dither, floor/ceil quantized), and coverage geometry for
//!   star/polygon/path silhouettes — emitting `seams::BackdropSplit`.
//! - **exec** (`exec.rs`): `copy_texture_to_texture` of the region from the
//!   resolved target, then (coverage →) (glass refraction →)
//!   blur | pixelate | dither, then the SDF/coverage-masked composite —
//!   sampling only already-drawn content (painter's algorithm). The serial
//!   error-diffusion dither stays a single `@workgroup_size(1)` dispatch
//!   (copy the bug).
//! - **kernel** (`kernel.rs`): the σ = reach/3 (a node's radius R reaches 1.5 R), kernel ≤ 31, folded-weight CPU
//!   math (`WebGPUBlurUtils.ts:36-117`).
//! Coverage geometry for star/polygon silhouettes goes through M2's
//! `tess::shape_paths` converters — the same ones the body draw and
//! `effects/mask.rs` use. The coverage-only duplicate that used to live in
//! `star_polygon.rs` was deleted with the move to the shared corner
//! rounder; mirroring the rounding in two places was exactly the failure mode.
//!
//! All effect WGSL comes from the extracted `backdrop-*` modules; the
//! composite blend state is manifest data (DECISION §1.3).
//!
//! - **blend** (`blend.rs`): exact `overlay` via a `BlendBegin`/`BlendEnd`
//!   split pair around the node's body (fixed-function blend state cannot
//!   express it). Those pages go single-sample too, but keep transparency.
//!
//! Pages with any **active** backdrop effect render with sample_count = 1 and
//! opaque white clear; `--transparent` / `--exclude` are unsupported there
//! (`WebGPUExportCoordinator.ts:786-795`, `WebGPURenderer.ts:2548-2583`;
//! quirk Q-02). The node draws every visible backdrop entry of `effects`, in
//! list order, one split each (`moda_scene::effect::backdrops_json`): they
//! stack.

pub mod blend;
pub mod exec;
pub mod kernel;
pub mod plan;

pub use blend::{
    canvas_root_has_split_blend, canvas_root_has_split_blend_over, compute_blend_splits, compute_group_splits, compute_stack_splits, page_has_split_blend,
    page_has_split_blend_over, page_splits_are_stacks_only,
};
pub use exec::BackdropRenderer;
pub use plan::{
    canvas_root_has_active_backdrop, compute_backdrop_split, compute_backdrop_splits, isolated_backdrop_conflict,
    page_has_active_backdrop, PlanCtx,
};
