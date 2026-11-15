//! moda-color: the colour tables and deterministic maths shared by moda-paint (destructive adjustments on a paint
//! layer's pixels) and moda-engine (the non-destructive adjustment backdrop kinds), so both give the same bytes for
//! the same parameters (docs/design/adjustment-layers.md §3; motion-model joint decision 9: one effect set).
//!
//! Determinism: no platform transcendentals (they differ between native and wasm libms), no hash maps, clock or
//! concurrency. `tests/lint.rs` checks the source.

pub mod fmath;
pub mod tables;
