# moda-assets provenance

Ported from Moda's `moda-assets` crate: local image sources, decode, ICC, SVG raster and the asset store
moda-engine reads image paints from.



`src/` is copied unchanged except for the test-only SVG font fixture path, which uses vork's bundled Noto Sans.

Changes:

- `Cargo.toml`: `edition = "2021"` written out; no `rust-version`; `license.workspace = true`; `publish = false`;
  `wgpu = "29"` instead of `=29.0.3`; `image` without its default features, with `png` and `jpeg` (the formats
  vork's canvas reads; the other default formats, AVIF and its encoder among them, are left out); the
  `jpeg-parity` feature and its `mozjpeg` dependency (a C build) are not ported, so JPEG decodes through the image
  crate as Moda builds without the feature do (its `cfg` stays declared under `[lints.rust]`);
  unit tests are enabled with the `png` dev-dependency; doctests remain disabled.
- `src/decode.rs`, `src/fetch.rs`, `src/lib.rs`, `src/engine_seam.rs`: a raster's decoded texels, for a host whose
  engine blends in another encoding than sRGB and converts, associates and resamples the image itself
  (`moda_shaders::working_color`). `natural_key(src)` names them in the store; the loader prepares a request for
  that key with `prepare_raster_natural` (no sizing policy, no 8-bit association), and the entry says its colour is straight (`PreparedImage::straight`,
  `ResolvedAsset::straight`; `PreparedImage::from_straight` for a host's own frames). `prepare_raster` and the
  plain key are unchanged.
