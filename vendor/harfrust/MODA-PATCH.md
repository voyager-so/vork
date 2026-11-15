# harfrust 0.5.2, patched

This is `harfrust` 0.5.2 from crates.io (MIT, `LICENSE` from the upstream repository's 0.5.2 tag; the published crate
leaves it out), used by `native/Cargo.toml`'s `[patch.crates-io]`. It is not a workspace member. `Cargo.lock` and
`Cargo.toml.orig` from the published crate are left out. Everything else is unchanged apart from the patch below.
cosmic-text 0.19 (moda-text's complex-script tier) shapes through it, and asks for `^0.5`.

## The patch: `would_apply` of a format-3 sequence context

`src/hb/ot/contextual.rs` `SequenceContextFormat3::would_apply` (GSUB/GPOS lookup type 5, format 3) wanted one glyph
more than the lookup's coverage array and matched glyph `i + 1` against coverage `i`. A format-3 coverage array
includes the first glyph, so the sequence is as long as the array and glyph `i` matches coverage `i`, as in HarfBuzz
and in harfrust's own `ChainedSequenceContextFormat3`. harfrust 0.14 fixed it the same way.

The Indic shaper asks `would_apply` whether `blwf`, `pstf`, `half`, … would substitute a consonant and virama, to
find each consonant's position. A font whose `blwf` is a format-3 context lookup therefore never formed below-base
forms: Noto Sans Bengali 3.011 (the Google Fonts release) drew ra-phala and ba-phala as a visible virama and a full
consonant (অনুগ্রহ as অনুগ্‌রহ, প্রিয় as প্‌রিয়), in every export and the preview alike.

The call site is marked `MODA PATCH`. `moda-text` `degraded::tests::format3_context_blwf_forms_ra_phala` (the font in
`native/fixtures/fonts-indic`) is the regression test.

## Updating

Drop this directory, and its `[patch.crates-io]` line, once cosmic-text takes harfrust 0.14 or later. Until then keep
the `version` in `Cargo.toml` equal to the one cosmic-text resolves (0.5.x): otherwise `[patch]` silently stops
applying, and the regression test fails.
