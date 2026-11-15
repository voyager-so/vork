//! moda-text-draw — Moda's text drawing on the GPU, ported from moda-engine: the vector glyph lane's plan step
//! (`text_draw::try_build_vector_batch`: a laid-out text node to a `TextQuadBatch`), the degraded tier's
//! (`text_draw::build_degraded_batch`: a cosmic-text raster quad for complex scripts) and their GPU half
//! (`text_draw::TextDraw`: the extracted `vector-text-*`, `text-decoration` and `text-emoji` pipelines). The
//! source files and every change to them are listed in PROVENANCE.md.

pub mod depth;
pub mod device;
pub mod seams;
pub mod text_draw;
pub mod vec_arena_cache;

pub use device::GpuContext;
pub use seams::TextQuadBatch;
pub use text_draw::{build_degraded_batch, build_vector_batch, try_build_vector_batch, vector_or_degraded, PreparedText, TextBuildOutcome, TextDraw};
