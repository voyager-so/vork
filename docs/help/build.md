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

The build compiles shaders for both Vulkan and Metal on either host. Linux builds
that only need Vulkan may set VORK_SHADER_TARGETS=spirv to skip Metal compilation. It checks
the Slang version against the pin. `VORK_SLANGC` must be an absolute executable
path; alternatively the build can find the pinned `slangc` on `PATH`. Export
`VORK_SLANGC` again in each new shell, or add the export to your shell startup
file. A `cannot start slangc` error means this setup is missing; a version
mismatch means a different compiler was selected.

The binary is `target/release/vork`, or `$CARGO_TARGET_DIR/release/vork` when
that environment variable is set. These commands do not install it globally.
Runtime program compilation also needs Slang; retain the export when running
the binary from the build directory.

## Windows x64

Install Rust 1.96 through rustup, Visual Studio Build Tools with Desktop development
with C++, LLVM (libclang for bindgen), and Python 3.12+. In a Developer PowerShell,
from the checkout root:

```powershell
python scripts/install-slang "$env:LOCALAPPDATA/vork/slang-2026.21"
$env:VORK_SLANGC = "$env:LOCALAPPDATA/vork/slang-2026.21/bin/slangc.exe"
cargo +1.96 build --release --workspace --locked
.\target\release\vork.exe help
```

Rendering requires a supported NVIDIA Vulkan driver on Windows and Linux, an AMD
Vulkan driver with ray queries on Linux (Mesa RADV), or Apple Silicon Metal on
macOS. For binary installation on Windows use install.ps1;
see README.md (Install). The compiled help and scene JSON are identical on every platform.

## Runtime dependencies

Linux/Windows video output and video decoding/fallbacks need FFmpeg on PATH;
video inputs also need ffprobe. Native Mac H.264 output uses VideoToolbox.
Install the pinned denoiser with `python scripts/install-oidn DESTINATION`.
Set VORK_OIDN to its lib/libOpenImageDenoise.so.2 on Linux,
lib/libOpenImageDenoise.2.dylib on macOS, or bin/OpenImageDenoise.dll on Windows.

Node/npm and the SDKs are not prerequisites for the Rust workspace build.
For SDK setup use `vork help sdk`; for authored Slang use `vork help programs`.

## Workspace tests

Workspace tests additionally need Node.js/npm for generated SDK checks.
Install the TypeScript dependencies and fetch the locked crate sources for the
licence notice check. After installing Slang and FFmpeg/ffprobe, run from the
checkout root:

```sh
npm ci --prefix packages/ts
cargo +1.96 fetch --locked
cargo +1.96 check --workspace --all-targets --locked
cargo +1.96 test --workspace --locked
```

The public workspace tests use shipped synthetic fixtures and the bundled
Liberation Sans and Noto Sans Arabic/Devanagari fonts (SIL OFL 1.1; licences in
crates/vork-{std,text}/tests/fonts/OFL.txt). No additional asset downloads are needed.
Corpus-dependent tests requiring unshipped assets are excluded from the public
source and are not part of its coverage.

Renderer GPU tests require the explicit runtime opt-in `VORK_GPU_TESTS=1`;
without it they print a skip diagnostic (visible with `-- --nocapture`). CPU and
numeric tests remain enabled. Ignored tests state their extra requirements.
With a supported GPU and the runtime dependencies installed:

```sh
VORK_GPU_TESTS=1 cargo test -p vork-render -p vork-cli -- --include-ignored --nocapture
```

In PowerShell set `$env:VORK_GPU_TESTS = "1"` before the Cargo command.
Use `-- --nocapture` on the default tests to see diagnostics, including skips
when adapter discovery finds no GPU. Ignored and skipped tests supply no GPU
coverage. Denoising tests also need the pinned OIDN runtime described above.
