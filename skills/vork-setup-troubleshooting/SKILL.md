---
name: vork-setup-troubleshooting
description: Install, build, run and debug vork - binary install on Mac/Linux/Windows, GPU support, source build with pinned Slang/Rust, Docker on your own NVIDIA host, `vork setup`, known limits, exit codes, upgrading 0.1 to 0.2, and fixing failed or wrong renders. Use when vork won't start, a render fails or aborts, or a scene from an older release breaks.
---

# Setup and troubleshooting

References: `vork help limits`, `vork help build`, `vork help errors`, `vork help upgrade`, `docs/BUILD.md`, `docs/REMOTE-GPU.md`, README "Install".

## Supported hardware

- **Mac**: Apple Silicon with Metal, macOS 26 for the bundled Slang toolchain (older unqualified). Intel Macs unsupported. H.264 via VideoToolbox.
- **Linux**: x64, glibc 2.35+ (Ubuntu 22.04+), recent NVIDIA with **hardware ray-tracing** Vulkan driver. A100/H100/H200/B200/B300 lack it. Containers need working NVIDIA Vulkan discovery.
- **Windows**: x64 10/11, recent NVIDIA with ray-tracing Vulkan.
- Unsupported: AMD, Intel, CPU/software rendering. Authoring and `vork help` need no GPU.
- No suitable GPU: run on your own NVIDIA Linux box with Docker (`docs/REMOTE-GPU.md`). Hosted rendering is not available yet.

## Install and check

Binary: `install.sh` (Mac/Linux, from GitHub Releases; verifies SHA256SUMS; installs under `$HOME/.local/lib/vork`, links `$HOME/.local/bin/vork`, runs `vork setup`), `install.ps1` (Windows). Linux MP4 needs `ffmpeg` even with NVENC. Keep the whole install folder together (Slang, OIDN, libraries, examples, SDKs).

```sh
vork --version
vork setup          # GPU report, shader warmup for the default profile
vork setup --all    # warm all bundled programs and profiles
```

Authored programs compile on first use; cold compile can take minutes.

## Source build (see `docs/BUILD.md`)

```sh
python3.12 scripts/install-slang "$HOME/.local/share/vork/slang-2026.20"   # Python >= 3.12, new empty dir
export VORK_SLANGC="$HOME/.local/share/vork/slang-2026.20/bin/slangc"      # absolute path, re-export per shell
cargo +1.96 build --release --workspace --locked
./target/release/vork help
```

- `cannot start slangc` = `VORK_SLANGC` missing; version mismatch = wrong compiler on PATH.
- Needs libclang (Basis textures), FFmpeg/ffprobe for video, OIDN for denoise (`scripts/install-oidn DEST`, set `VORK_OIDN`).
- Tests: `npm ci --prefix packages/ts`, `cargo +1.96 test --workspace --locked`; GPU tests need `VORK_GPU_TESTS=1`.

## Failure triage

Read the log first (`>run.log 2>&1`), then the exit code: **2** usage - **3** invalid scene/program (message names the field; use `vork help field PATH`) - **4** device/GPU memory/encoder/setup - **5** filesystem IO.

| Symptom | Action |
|---|---|
| `unknown field` | Remove it or fix the name; unknown fields are errors |
| unknown-program error for `std/fog`, `blender/principled_volume`, `blender.volume` | Removed in 0.2; delete those definitions (even unused ones) |
| Exit 4 "Impacting Interactivity" (Apple GPU), solid glass / heavy interior | Retry; `geometry_thin_walled` for panes; `std/path`; NVIDIA for repeats. Lower resolution is not a reliable fix |
| Device lost with dense transparency or imported transmitting geometry | Try `std/path`; test the full shot before batching; unqualified on Metal |
| Per-buffer binding limit on big point scenes | Follow the error's advice; simplify buffers |
| Interiors too dark in final | `std/path` (final approximates indirect) |
| Noisy or fireflies | `--set samples=N`; keep denoise; no fast GPU denoiser = 16/32 undenoised samples, `--set denoise=true` forces slow CPU |
| `WARNING: REQUESTED EFFECT DROPPED` | The listed feature wasn't applied (hair displacement, extra environment lights, unsupported light masks, motion channels with curved propagation...). Remove or replace it |
| Video output or video input fails | Install ffmpeg on PATH (Linux/Windows output); ffprobe too for video inputs |
| Cache issues | `VORK_CACHE_DIR`, `VORK_CACHE_BYTES=0` to disable the disk cache |

## Upgrading 0.1 -> 0.2 (`vork help upgrade` is authoritative)

- Keep the 0.1 binary and scenes; install 0.2 to a separate `PREFIX` to keep both; select with `VORK_BIN`.
- Before upgrading, run scatter scripts once and save instance columns and buffers; scatter builders are gone.
- `post/vignette|lens_distortion|glare` -> local Slang from `examples/compositor-effects/`.
- Fog/volume programs removed; omitted mesh material roughness default is now 0.3 (was 0.5).
- TS: package `@voyager-so/vork`, camelCase `sky`/`environmentImage` options; `renderArguments` -> `render()`.
- Visible changes: denoising filters in display-exposure units; opposing mesh normals are flipped to winding; `std/path` brightness changed for finite-source null-solid boundaries. Re-check representative frames.
- `$schema` is rejected by 0.1: remove before rollback.
- Don't parse stdout by line position or treat stderr as failure; use exit status and sidecars.
