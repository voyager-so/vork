# moda-text provenance

Ported from Moda's `moda-text` crate.



`src/` (every module, the hyphenation patterns under `src/hyph/` included) is copied unchanged, except for the
changes listed below.

Changes:

- `Cargo.toml`: `edition = "2021"` written out instead of inherited from Moda's workspace (vork's workspace is
  edition 2024); `license.workspace = true`; `publish = false`; the dependency comments that cite Moda's planning
  documents are shortened.
- `Cargo.toml`: unit tests are enabled with a `moda-scene` dev-dependency; doctests and library benchmarks
  remain disabled. Unit tests use shipped fonts where equivalent and explicitly ignore unported Moda fixtures.
  Test-only fixture loaders and assertions are adapted to the shipped fonts and current stroke paint order.
- `Cargo.toml`: `icu_segmenter` is pinned to `=2.3.0`, the release Moda's lockfile resolves; `src/sea_words.rs`
  names that release's data markers, and vork's lockfile otherwise resolves 2.4.
- Not ported: Moda's integration tests against its fixtures and oracle dumps. The `tests/` directory here
  contains vork regression tests and their small bundled fixtures.
- Not ported: Moda's workspace patch of `harfrust` (`native/vendor/harfrust`, one `would_apply` fix for Indic
  below-base forms). cosmic-text, which the degraded tier shapes with, resolves crates.io `harfrust` here.

- `src/degraded.rs`: added, so the degraded tier's text can be geometry (std/text extrudes right-to-left and
  complex-script text with it): the pub `DegradedEngine::glyph_outlines`, which runs `render_node`'s draw walk and
  returns each drawn glyph's outline (`GlyphOutline`, `OutlineSegment`) unhinted from swash
  (`SwashCache::get_outline_commands_uncached`), placed in the node by `raster_offset`; the private `recording`
  field of `EngineInner` (`GlyphRecording`, `RecordedGlyph`, `None` in `from_parts`) that the walk fills while it
  runs: `draw_shaped_run` records each glyph's face, size, weight, flags and pen, and `render_node_scaled` the
  y of the lines' top in the raster (`margin_px / scale`). Drawing is unchanged.

- `src/degraded.rs`: layout and raster entry points validate shaped glyph coverage and name uncovered
  characters instead of drawing glyph zero. They refuse unsupported variation settings and nonzero
  letter spacing on contextually shaped scripts before measurement or drawing. The regression tests in
  `tests/degraded_contract.rs` use vork's bundled fonts with system font discovery disabled.

The hyphenation patterns are from hyph-utf8 under MIT, BSD-3-Clause or LPPL-or-MIT terms, as each file's notice
states; none is GPL or LGPL (the Italian file's notice records its move from LGPL to LPPL in 2011, and it is
available under MIT).

- `src/paragraph_direction.rs`: route paragraphs with odd bidi levels through shaping even when their base
  direction is LTR or undeclared (including Unicode RTL overrides).
- `src/wrap.rs`: emergency grapheme breaking requires `word-break: break-all` or `line-break: anywhere`;
  normal unbreakable words and NBSP groups overflow rather than splitting.
- `src/font_weight.rs`, `src/runs.rs`, `src/vector/measure.rs`, `src/vector/node_layout.rs`,
  `src/vector/extract.rs`: preserve oblique style through normalization, face selection and measurement;
  synthesize the existing slant when an oblique request resolves to an upright face.

- `src/degraded.rs`: paragraph wrapping and intrinsic measurement use the existing CJK/kinsoku/SEA
  segmenter (`vector::cjk::segment_text_for_wrap_with`), preserving CJK character breaks in shaping
  paragraphs; the SEA-only segmenter omitted those opportunities. `tests/degraded_contract.rs`
  covers mixed Han/Arabic, including styled and list paragraphs, using the test-only OFL Han
  font subset documented in `tests/fonts/README.md`.

- `src/runs.rs`: resolve CSS `font-weight` to the variable `wght` coordinate before face matching, retaining
  explicit `font-variation-settings` precedence; static faces ignore absent axes through the existing instance resolver.
- `src/vector/ops.rs`: submit the legacy text stroke's band after the fill, as SVG paints it.
  Keep structured shadows, glows, echoes and extrusion behind the fill. `tests/vector_css_contract.rs` covers
  variable outlines/layout, axis overrides/clamping and fill/stroke ordering with a shipped Noto Sans variable face.

- `src/emoji.rs`, `src/runs.rs`: classify presentation over each complete paragraph, retaining source
  code-point ranges to project text back onto styled runs. Emoji clusters keep their base run's style;
  selectors and joiners across styling boundaries make the same decision as unstyled text.

- `src/ops.rs`: share immutable HTML and per-code-point arrays between text specs;
  `unit_range` selects code points after full-layout positioning and unit pivots.
  `src/vector/ops.rs` omits unselected glyph/emoji records before allocating draw geometry.
  Constructors and mutable callers use owned-to-shared conversion and copy-on-write.

- `src/vector/measure.rs`: default runs use HarfBuzz OpenType defaults through the same dispatch as explicit
  feature overrides, including GPOS extension kerning and default GSUB substitutions.
  `tests/harfbuzz_defaults.rs` compares glyphs, advances and layout widths against independent HarfBuzz results.

- `src/types.rs`, `src/ops.rs`, `src/node_layout.rs`, `src/wrap.rs`, `src/paragraph_layout.rs`,
  `src/vector/{ops,node_layout}.rs`, `src/degraded.rs`: an internal line-height policy preserves Moda's
  whole-pixel Figma default while vork canvas selects fractional CSS line heights. Drawing, wrapping,
  empty lines, text measurement and glyph extraction use the same policy; shaping and baseline metrics
  are unchanged. `tests/line_height_policy.rs` and `tests/degraded_contract.rs` cover fractional line
  advances across hard breaks, soft wraps and empty paragraphs, including actual glyph draw positions.
