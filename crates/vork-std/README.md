# JSON standard library

Unstable, internal Rust crate; everything in it is free to change.

`vork std FUNCTION 'JSON'` writes one JSON value to stdout; `-` reads the argument from stdin.
The CLI and scene loader depend on this crate. Geometry definitions use
`{"std":"SHAPE",...options}` inline or in the geometry table; the loader reuses the
CLI evaluator. This crate does not depend on the scene or renderer crates.

The implementation is generated directly from `packages/ts/src/lib/*.ts` with
`node packages/ts/generate-std.mjs` (the SDK development dependencies for generation only). The generated file is
committed so building and running vork needs neither Node nor the SDK. QuickJS-ng (rquickjs) evaluates
only that fixed implementation and JSON data, with no host filesystem/network bindings.
This keeps winding, defaults, and validation shared with the SDK instead
of maintaining a second mathematical implementation. `npm run check:generated --prefix packages/ts` checks bundle freshness;
`cargo test -p vork-std` also checks all committed generated artifacts. It requires Node,
`npm ci --prefix packages/ts`, Python 3 and `cargo fetch --locked` for all-platform licence sources.

Build without the `vork std` command: `cargo build -p vork-cli --no-default-features`.
Scene geometry builders remain available.



`text` is a native std loader: Moda's text engine (`crates/moda-text`) lays out the
supplied TrueType/OpenType font's text and gives the glyph outlines, which it
tessellates with MIT/Apache-2.0 Lyon. It emits
ordinary buffer/geometry fragments, with counters, kerning, multiline alignment,
extrusion and rounded bevels. The QuickJS runtime keeps no host bindings.
`text(name, options)` in the Node SDK calls the installed `vork std text` loader;
set `VORK_BIN` to select the executable. No additional font runtime is needed.


The Python package in `packages/python` uses `vork std` through a thin wrapper;
its input TypedDicts are generated from the Rust-emitted JSON Schema by
`packages/python/generate.py`, invoked by the existing schema generation checks.
