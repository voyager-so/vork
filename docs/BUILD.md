# Build from source

Run these commands from the checkout root. Source builds need Rust, a native
C/C++ toolchain, Python with safe tar extraction filters, and the pinned Slang
compiler. Cargo does not download Slang. The supported Unix toolchain archives
are Apple Silicon macOS and x86-64 Linux (see `dist/slang.json`); there is no
pinned Intel Mac or Linux ARM archive.

## Prerequisites

The pinned Slang 2026.21 macOS binaries target macOS 26. Older macOS versions
have not been qualified with this toolchain.

On macOS, install Apple's command line tools, then install Homebrew from
https://brew.sh if it is not already available:

```sh
xcode-select --install
```

Finish that installer before continuing. Use Homebrew Python explicitly:
stock macOS Python 3.9 may lack the extraction filters required by the Slang
installer. No pip packages are needed.

```sh
brew install python@3.12
PYTHON="$(brew --prefix python@3.12)/bin/python3.12"
```

On Ubuntu 24.04 x86-64 (the Linux baseline), install:

```sh
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libclang-dev curl git python3 ca-certificates
PYTHON=python3
```

Basis texture bindings require libclang (`libclang-dev` on Ubuntu; supplied
by Apple's command line tools on macOS). If bindgen cannot find it, set
`LIBCLANG_PATH` to the directory containing libclang.

Other Linux distributions need equivalent C/C++ build tools, libclang and Python 3.12+
(or a Python release with backported `tarfile.data_filter`). Rendering also
needs a working Vulkan driver; building and GPU-free commands do not need a GPU.
For your own NVIDIA or AMD Linux machine, a [Docker runtime](REMOTE-GPU.md) is available.

Install Rust 1.96, matching CI, through rustup (https://rustup.rs):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/vork-rustup.sh
sh /tmp/vork-rustup.sh -y --default-toolchain 1.96
. "$HOME/.cargo/env"
```

## Install Slang and build

Clone the repository if needed, then enter it.

Choose a new, empty destination outside the checkout. The installer downloads
and verifies the archive against `dist/slang.json`, and refuses to overwrite
an existing directory. Keep the entire toolchain, including its libraries,
modules and notices, together.

```sh
"$PYTHON" scripts/install-slang "$HOME/.local/share/vork/slang-2026.21"
export VORK_SLANGC="$HOME/.local/share/vork/slang-2026.21/bin/slangc"
"$VORK_SLANGC" -version
cargo +1.96 build --release --workspace --locked
./target/release/vork --help
./target/release/vork help
```

The build compiles shaders for both Vulkan and Metal on either host. It checks
the Slang version against the pin. `VORK_SLANGC` must be an absolute executable
path; alternatively the build can find the pinned `slangc` on `PATH`. Export
`VORK_SLANGC` again in each new shell, or add the export to your shell startup
file. A `cannot start slangc` error means this setup is missing; a version
mismatch means a different compiler was selected.

The renderer and its `vork-slang-filesystem` companion must stay together when copied or packaged. Building the companion requires a C++17 compiler and the full pinned Slang headers; it loads the matching shared library at runtime.

The binary is `target/release/vork`, or `$CARGO_TARGET_DIR/release/vork` when
that environment variable is set. These commands do not install it globally.
Runtime program compilation also needs Slang; retain the export when running
the binary from the build directory.

## Rendering and test dependencies

Start with `vork help` and the scenes in `examples/`. macOS uses Metal; Linux
uses Vulkan. Install FFmpeg/ffprobe for video commands and encoding tests
(`brew install ffmpeg` on macOS, `sudo apt-get install ffmpeg` on Ubuntu).

The public workspace includes CPU unit tests and small fixtures. The text tests
use bundled Liberation Sans and Noto Sans Arabic/Devanagari fonts under SIL OFL
1.1; their licences are in `crates/vork-std/tests/fonts/OFL.txt` and
`crates/vork-text/tests/fonts/OFL.txt`. Renderer fixtures are synthetic scene,
shader and glTF inputs shipped with the source. No asset downloads are needed to
run the default tests. Corpus-dependent tests that require unshipped assets are
excluded from the public source; they are not part of its test coverage.

Renderer GPU tests require the explicit runtime opt-in `VORK_GPU_TESTS=1`;
without it they print a skip diagnostic (visible with `-- --nocapture`).
CPU and numeric tests remain enabled. Tests marked ignored explain their additional requirements. GPU tests need a
supported Metal/Vulkan device, including ray-query support; a CPU-only test run
does not validate rendering. To run ignored tests after installing their stated
dependencies, set `VORK_GPU_TESTS=1` and use
`cargo test -p vork-render -p vork-cli -- --include-ignored --nocapture`.

OIDN is loaded at runtime, not required to compile. For denoised renders and
GPU tests that exercise denoising, install the pinned runtime with the same Python:

```sh
"$PYTHON" scripts/install-oidn "$HOME/.local/share/vork/oidn"
# macOS:
export VORK_OIDN="$HOME/.local/share/vork/oidn/lib/libOpenImageDenoise.2.dylib"
# Linux (use this instead):
export VORK_OIDN="$HOME/.local/share/vork/oidn/lib/libOpenImageDenoise.so.2"
```

Workspace tests additionally need Node.js/npm for generated SDK checks.
Install the TypeScript dependencies and fetch the locked crate sources for the
licence notice check, then check and test every public workspace target:

```sh
npm ci --prefix packages/ts
cargo +1.96 fetch --locked
cargo +1.96 check --workspace --all-targets --locked
cargo +1.96 test --workspace --locked
```

Use `-- --nocapture` to see test diagnostics, including any GPU test skipped when
adapter discovery finds no device. Missing fixtures, missing required tools and
failed assertions are failures. An ignored or skipped GPU test supplies no GPU
coverage.

See [README installation](../README.md#install) for binary installation and runtime
setup, and `vork help programs --all` for authored Slang programs. Node/npm and the
SDKs are not prerequisites for the Rust workspace build.



## Public workflows

Distribution archives are built privately and attached to public GitHub releases.
For source validation, run the workspace tests above and `npm ci && npm test`
in `packages/ts`. Rendering checks need the runtime dependencies above;
run the examples with a supported GPU and inspect the resulting frames.

## Release shader assets

Hosted distribution jobs run `scripts/prepare-shaders` with the final binary
(after macOS signing). It invokes the hidden `prepare-release-shaders` command:
the ordinary setup catalogue and a bounded set of starter/example scenes run
against a no-op wgpu device with Slang. The scene set in `scripts/prepare-shaders`
uses normal installation at `std/final` and `std/path`, capturing installed material
unions and linked cameras that isolated library-program jobs cannot cover.
Shared portable artifacts deduplicate by their existing content keys; arbitrary
authored programs and uncommon configurations still compile on demand. Preparation runs
without opening a GPU driver, OIDN device or executing shaders. Denoiser
preparation constructs only the renderer's pack/unpack pipelines. macOS emits Metal source;
Linux and Windows emit SPIR-V. The bundle excludes all driver cache entries
and is bound to the packaged binary's SHA-256. `scripts/package-release` refuses
to package without this verified bundle. `vork setup` on the user's real GPU
then warms its own driver cache from those portable assets.

The shipped generic GI and path entries enable the same existing Slang function
boundaries as their specialized variants. Without these boundaries, Slang's
SPIR-V optimization expands the generic lighting/transport graph. The official
specialization fix reduces duplicate work; these boundaries further bound the
optimizer's memory use.

On Linux, measure individual compiler invocations with
`scripts/toolchains/measure-slang-memory.py` as `VORK_SLANGC`, setting
`VORK_SLANG_MEMORY_COMPILER` to the real pinned compiler and
`VORK_SLANG_MEMORY_REPORT` to a JSONL file outside the checkout. Use a fresh
Cargo target directory and `VORK_SHADER_TARGETS=spirv,metal`; disable compiler
artifact caching (`VORK_VM_SHADER_CACHE=0` on the shared VM) while retaining
admission guards. Each record contains the target, entry, command, exit status,
CPU time and peak process RSS in KiB. Wall time includes admission waits and is
not a compiler speed comparison. Compare both pins on the same renderer sources
and generated inputs; include failed invocations in the inventory.

`check-release --skip-setup --skip-render` verifies installation, compiler and
portable assets on GPU-free hosted runners. Installed rendering, video and
denoising still require the real-device acceptance check.

## Opt-in CPU rendering

On Linux, install recent Mesa Vulkan drivers (`mesa-vulkan-drivers` on
Debian/Ubuntu; `mesa-vulkan-drivers` or `vulkan-swrast` elsewhere), then run
`vork setup --gpu cpu` (or `VORK_GPU=cpu vork setup`) and `VORK_GPU=cpu vork render scene.json --out frame.png`.
The driver must expose ray queries and storage buffer arrays. CPU rendering is
slow and never selected automatically. Setup warms the default starter. CPU shader
warm-up has no time limit. First-run shader compilation can take several minutes. vork's driver
pipeline cache is disabled because it can trigger a lavapipe compiler crash.
Portable shader artifacts and Mesa's own shader disk cache can still speed later runs.

macOS requires Apple Silicon; CPU mode is unavailable through Metal. Windows
requires a Vulkan loader and a recent Mesa lavapipe driver with its ICD manifest;
CPU rendering there has not been qualified.
