# moda-plain-text provenance

Ported from Moda's `moda-plain-text` crate.



`src/` is copied unchanged.

Changes:

- `Cargo.toml`: `edition = "2021"` written out instead of inherited from Moda's workspace (vork's workspace is
  edition 2024); no `rust-version`; `license.workspace = true` (vork's licence); the description no longer names
  moda-motion, which is not ported.
