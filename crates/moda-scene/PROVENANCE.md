# moda-scene provenance

Ported from Moda's `moda-scene` crate: the scene layer moda-engine plans from (document hydration, the display
list, paints, strokes, effects, masks and path geometry).



The initial import copied `src/` unchanged except for the test-only changes below.
Subsequent local code changes are recorded in Git history.

Changes:

- `Cargo.toml`: `edition = "2021"` written out; no `rust-version`; `license.workspace = true`; `publish = false`;
  unit tests and doctests are enabled, with the original `png`, `proptest` and
  `serde_json` test dependencies; the `code` feature and its `moda-code` git dependency
  (live generators through Moda's code sandbox) are not ported; its `cfg` stays declared under `[lints.rust]`.
- `flo_curves` resolves to the copy Moda patches, vendored as `vendor/flo_curves` and named in the workspace's
  `[patch.crates-io]`, as Moda's workspace does.
- Not ported: integration test targets in `tests/` and `examples/`; the unit-test fixtures below are restored.

## Unit-test restoration

Copied byte-for-byte from the upstream commit recorded above into `tests/fixtures/`:

- `chart_number_locales.json` and `illustrator_warp.json` from
  `native/crates/moda-scene/tests/fixtures/`.
- `table.json` from `fixtures/parity/design/scenes/tables/`.
- `chart-{bar,line,area,scatter,combo}.json` from `fixtures/parity/design/scenes/generated/`.
- `structured-{glow,echo,extrude,combo,animated,non-adopter}.json` from
  `fixtures/parity/design/scenes/effects/`.

Fixture paths in `expansion.rs`, `assess.rs`, `charts/mod.rs` and `charts/motion_parts.rs`
now point to this crate's fixtures. The two tests removed are
`defaults::tests::default_chart_config_matches_the_typescript` and
`charts::tests::ported_registry_matches_production_chart_universe`: they compare against the
production Moda editor's TypeScript sources, which vork does not ship. A frozen copy of those
sources would not guard editor/native drift. All other unit tests remain, including their
original ignored diagnostic/performance and gallery cases.
