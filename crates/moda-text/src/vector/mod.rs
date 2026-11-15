//! The vector text engine's layout lane (`--text-engine vector`).
//!
//! Numbers-before-pixels port of the browser's vector engine
//! (`renderers/webgpu/text/vector/**`, landed 2026-06-09 behind
//! `?text=vector`): the SHARED rich-text layout core (already ported for
//! MSDF) driven by a kerned, ligature-aware measurer over opentype.js-exact
//! shaping (`otjs.rs`), plus the CJK wrap segmenter and coverage-split
//! fallback (`cjk.rs`), the measurer/store (`measure.rs`), and the per-node
//! dump (`node_layout.rs`) that Gate 1 compares against
//! `fixtures/oracle-vector/*/linebox.json` at 1e-3 px.

pub mod alternates;
pub mod cjk;
pub mod curves;
pub mod extract;
pub mod extrude;
pub mod hb;
pub mod mask_raster;
pub mod measure;
pub mod node_layout;
pub mod ops;
pub mod otjs;
pub mod script_fallback;
pub mod unicode_scripts;
pub mod varfont;

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod character_tests;

pub use curves::GlyphQuadratics;
pub use mask_raster::render_mask_raster;
pub use measure::{
    font_url_label, manifest_files, VectorFontMiss, VectorFontStore, VectorMeasurer,
};
pub use node_layout::{
    dump_rich_text_node_vector, dump_rich_text_node_vector_with_placement, layout_rich_text_node_detail,
    layout_rich_text_node_vector,
};
pub use ops::{
    accept_vector_layout, build_vector_node_ops, layout_counts, run_color, structured_effects_active, LaidLine, VectorDecline,
    VectorEngineDecision, VectorLayout, VectorNodeOps,
    EmojiEffect, VectorEmoji, VectorOp, VectorOpPipeline, VectorPlacement, EMOJI_CELL_ASCENT_EM,
    EMOJI_CELL_HEIGHT_EM, EMOJI_CELL_WIDTH_EM, EMOJI_EFFECT_TEXELS_PER_EM,
};
pub use otjs::{OtjsFont, ShapeError};
