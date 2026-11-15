# flo_curves 0.8.1, patched

This is `flo_curves` 0.8.1 from crates.io (Apache-2.0, `LICENSE`), used by
`native/Cargo.toml`'s `[patch.crates-io]`. It is not a workspace member. The
logos, `Cargo.lock` and `Cargo.toml.orig` from the published crate are left
out. Everything else is unchanged apart from the patch below.

## The patch: collision sorts that cannot panic

`bezier/path/ray.rs` `ray_collisions` sorts ray collisions with a comparator
that is not a total order. It orders by position along the ray, but
collisions within `SMALL_DISTANCE` of each other on overlapping edges are
ordered by edge priority instead, and "within `SMALL_DISTANCE`" is not
transitive. Since Rust 1.81, `slice::sort_by` may panic when it notices this.
Moda's wasm build is `panic = "abort"`, so that panic took down the whole
engine. It happened in 6.5–12.5% of boolean and Pathfinder runs on Blob Brush
outlines, and 0.5–2% on shapes combined with a near copy of themselves
(`moda-scene/src/path_bool/panic_fuzz_tests.rs`).

`sort_by_never_panics` (end of `ray.rs`) replaces that sort. It is a stable
bottom-up merge sort that never panics, whatever the comparator answers:

- For a comparator that is a total order, it gives exactly what `sort_by`
  gives, because a stable sort's result is unique.
- Otherwise it gives some order, which is what `sort_by` did before 1.81.

The same function also replaces the `t`-ordering sort in
`bezier/path/graph_path/path_collision.rs`, whose comparator calls NaN equal
to everything.

Both call sites are marked `MODA PATCH`. The panic fuzz above is the
regression test.

## Updating

Replace `src/` with the new release's, then re-apply the two `MODA PATCH`
hunks, or drop this directory if upstream has fixed the sort. Keep the
`version` in `Cargo.toml` equal to the pinned `=0.8.1` in
`crates/moda-scene/Cargo.toml`. Otherwise `[patch]` silently stops applying,
and the panic fuzz will fail.
