# moda-wgsl provenance

Ported from Moda's `moda-wgsl` crate (agent WGSL: the std prelude, placement ABI and checks moda-engine builds its
pixel kernels with).



`src/` is copied unchanged, except `src/hosts/fill-head.wgsl`: `STROKE_DEPTH_OFFSET` is vork's
(`crates/moda-shaders/PROVENANCE.md`).

Changes:

- `Cargo.toml`: `edition = "2021"` written out; no `rust-version`; `license.workspace = true`; `naga = "29"`
  (vork's wgpu 29 line) instead of `=29.0.3`.
