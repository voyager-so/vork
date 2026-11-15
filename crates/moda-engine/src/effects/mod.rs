//! effects/ — drop/inner shadows, masks, blend modes, opacity. OWNED BY M3d.
//!
//! The seam this module implements (SEAMS.md §4/§5 + change log):
//! - `seams::DrawItem::Shadow` carrying `seams::ShadowInstance` (160-byte
//!   `RectangleShadowData` ABI of the extracted `path-shadow` module);
//!   shadow-first ordering inside EXACT (fractional) z buckets; inner
//!   shadows at z + `seams::INNER_SHADOW_Z_OFFSET`,
//! - `seams::MaskRef`/`seams::MaskPlan` + the engine mask bind group
//!   (r8unorm coverage, dpr-1 pinned, linear sample, outside-UV discard,
//!   nested ancestor compose),
//! - blend modes consumed FROM `moda_shaders::manifest().blend_modes`
//!   (overlay = src-alpha color factor — extracted data, not prose; wired
//!   by M1's executor, asserted in seams.rs tests),
//! - group opacity: a group, or a frame with children, at an opacity between 0 and 1 composites its subtree as one
//!   unit (`moda_scene::scene::is_opacity_group`): `frame.rs` opens a subtree capture for it and wraps everything
//!   it drew in a `BlendBegin` / `StackEnd { mode: 0 (source-over), opacity }` split, and its layers draw at their
//!   own opacity inside (`Scene::ancestor_opacity_product_except` skips the open fade). Elsewhere (no split context)
//!   an ancestor's opacity is multiplied into each leaf, as before,
//! - drop shadows ONLY for rect/ellipse-derived silhouettes (Q-05), one quad
//!   per drawing drop / inner shadow in `effects` order (back to front),
//!   multiSample 9-tap falloff pinned in the extracted WGSL.
//!
//! Split: `shadow.rs` = pure instance packing (plan side); `mask.rs` = pure
//! mask planning (plan side) + `MaskAtlas` (executor side). frame.rs calls
//! into these at the integration points recorded in SEAMS.md's change log.

pub mod mask;
mod mask_filter;
pub mod shadow;

pub use mask::{
    clip_rounded_rect, clip_rounded_rect_until, clip_to_node_box, container_clip_aabb, effective_page_clip,
    effective_page_clip_until, has_non_box_clip_mask, MaskAtlas, MaskCache, PlanMasks, SurfaceViews,
    PIXEL_SINK_SRC_PREFIX,
};
pub use shadow::{
    drop_shadow_instances, has_drop_shadow_quad, has_inner_shadow_quad, inner_shadow_instances,
    inner_shadow_quad_count,
};
