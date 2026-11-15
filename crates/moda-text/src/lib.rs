//! moda-text — the text layout core: rich-text parsing, runs, variant snapping, wrapping, list/bullet positions
//! and baselines, shared by the vector glyph lane (`vector`) and the degraded raster tier (`degraded`).
//!
//! ```text
//! htmlContent
//!   └─ parse (html5ever, parser.ts semantics)                 parser.rs
//!       └─ runs + emoji split (parserRuns.ts/emoji.ts)        runs.rs / emoji.rs
//!           └─ variant snapping (resolveRunVariant.ts)        runs.rs / catalog.rs
//!               └─ greedy wrap (layout.ts)                    wrap.rs / measure.rs
//!                   └─ baselines/lists/bullets/positions      line_metrics.rs / positioning.rs
//!                       └─ per-node layout + glyph outlines   vector/
//! ```

pub mod area;
pub mod bidi_visual;
pub mod catalog;
pub mod character;
pub mod cjk_spacing;
pub mod color;
mod collection;
pub mod complex_script;
pub mod composer;
pub mod decorations;
pub mod degraded;
pub mod emoji;
pub mod ext_pict;
pub mod family_normalization;
pub mod flags;
pub mod font_features;
pub mod font_variations;
pub mod font_weight;
pub mod host;
pub mod hyphenate;
pub mod jsnum;
pub mod lang;
pub mod line_metrics;
pub mod list_markers;
pub mod lru;
pub mod measure;
pub mod node_layout;
pub mod ops;
pub mod paragraph_direction;
pub mod paragraph_format;
pub mod paragraph_layout;
pub mod parser;
/// Moved to the `moda-plain-text` crate (moda-motion needs it without the text engine); re-exported here.
pub use moda_plain_text as plain_text;
pub mod positioning;
pub mod ruby;
pub mod run_fill;
pub mod runs;
pub mod sea_words;
pub mod small_caps;
pub mod text_case;
pub mod text_path;
pub mod tab_stops;
pub mod text_wrap;
pub mod truncation;
pub mod ttf;
pub mod types;
pub mod vector;
pub mod vertical;
pub mod wrap;

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod test_fixtures;
/// Paths of the world-type faces. The public tree has neither the faces nor
/// their paths; no test that reads one is built there.
#[cfg(all(test, vork_private_tests))]
#[path = "world_type_fonts_private.rs"]
mod world_type_fonts;
#[cfg(all(test, not(vork_private_tests)))]
#[allow(dead_code)]
mod world_type_fonts {
    pub(crate) const WORLD_TYPE_LATO: &str = "absent-from-the-public-tree/Lato-Regular.ttf";
    pub(crate) const WORLD_TYPE_HEBREW: &str = "absent-from-the-public-tree/NotoSansHebrew-Regular.ttf";
    pub(crate) const WORLD_TYPE_THAI: &str = "absent-from-the-public-tree/NotoSansThai-Regular.ttf";
    pub(crate) const WORLD_TYPE_DEVANAGARI: &str = "absent-from-the-public-tree/NotoSansDevanagari-Regular.ttf";
}

pub use catalog::Catalog;
pub use complex_script::{codepoint_needs_complex_shaping, text_needs_complex_shaping};
pub use host::TextHost;
pub use node_layout::RichTextNodeSpec;
pub use ops::{
    parse_gradient_stops, GlyphQuadOut, GradientSpec, TextDropShadow, TextEffectSpec, TextGradientUniforms,
    TextNodeOps, TextNodeSpec, TextOp, TextOpPipeline,
};
pub use types::{FontKey, InlineStyle, Line, LineSegment, Paragraph};
