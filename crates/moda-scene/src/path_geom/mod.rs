//! Shared path geometry helpers used by both the tessellator (via moda-engine)
//! and the boolean path solver, plus curve fitting for the brush tools and the
//! Path menu's Offset Path and Simplify (shared-foundation F4.2).
pub mod arc;
pub mod fit;
pub mod offset_path;
pub mod round_corners;
pub mod simplify;
pub mod vec2;
