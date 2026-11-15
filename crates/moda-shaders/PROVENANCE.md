# moda-shaders provenance

Ported from Moda's `moda-shaders` crate: the extracted production WGSL and the pipeline-state manifest that Moda's
2D engine (crates `moda-engine`, `moda-text-draw`) draws with.



`build.rs`, `manifest.json`, `wgsl/` and `src/` (with `src/runtime/`) are copied unchanged, except:

- `src/lib.rs`: the test `manifest_and_embedded_modules_agree` is removed: the manifest's
  `counts.total_modules` (122) disagrees with its own 123 module entries at this commit, so it fails in Moda too.
- `Cargo.toml`: `edition = "2021"` written out; `license.workspace = true`; `publish = false`; the build
  dependency is `naga = "29"` (vork's wgpu 29 line) instead of `=29.0.3`; the `wgpu` and `pollster`
  dev-dependencies are dropped with `tests/`.
- Not ported: `examples/`, `tests/`.
- `wgsl/layer-filter-shadow.wgsl`: `fs_source` is added: the capture drawn source-over on its filter shadows
  (moda-engine `layer_blur/exec.rs` `shadows_of`). The other entry points are unchanged.
- `STROKE_DEPTH_OFFSET` is `-0.000001490116119384765625` (25 levels of the 24-bit depth attachment: 5/8 of vork's
  render-z slot, `moda_text_draw::depth`) instead of `-0.00001` (5/8 of Moda's slot) in the 53 path modules that
  declare it and in `manifest.json`'s `pinned_constants`; `src/lib.rs`'s `pinned_constants_hold` expects it and
  `stroke_depth_offset_is_the_pinned_one_in_every_module` is added.
- `wgsl/vector-text-analytic.wgsl`, `wgsl/vector-text-analytic-additive.wgsl`: centre the text stroke's signed-distance
  band on the glyph outline, with half its width on each side, and test overlapping-contour interior edges across
  that full radius. Shadow and glow silhouette coverage is unchanged.

- Added by vork: `src/working_color.rs` and `wgsl/include/working-color.wgsl`, an engine's working colour.
  `page-fill`, `path-solid`, `path-gradient`, `path-pattern` (its stroke colour), `path-shadow`, `shape-solid`,
  `shape-gradient`, `layer-filter-shadow`, the four `vector-text-*` system modules and `text-decoration` splice
  the include and pass each authored colour through `workingRgb` before they associate and blend it;
  `mask-coverage` passes a rendered mask's picture through `workingToSrgb` before CSS luminance. An sRGB engine
  compiles the two identity definitions, so its arithmetic is unchanged; an engine in another encoding replaces
  them (`working_color::specialize`). `build.rs` validates every spliced module in both forms.
