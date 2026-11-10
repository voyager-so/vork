---
name: vork-render-outputs
description: The `vork render` command and its outputs - stills, frame sequences, multilayer EXR, MP4/MOV video, depth/normal/motion/object-index/AOV passes, sidecar meta.json, exposure summary, diagnostics and exit codes. Use when choosing output format or channels, rendering a clip, exporting passes for compositing, or benchmarking and debugging a render.
---

# Render and outputs

Fields: `vork help render --all`, `vork help outputs --all`, `vork help diagnostics`, `vork help errors`.

```sh
vork render SCENE --out PATH [--at SECONDS | --frame-start N --frame-end N] [--profile P]
  [--set KEY=VALUE ...] [--channels LIST] [--exr-codec CODEC] [--diagnostics]
```

`--out` repeats; the extension decides the output:

| `--out` | Result |
|---|---|
| `x.png` | one RGBA still (t=0 unless `--at`); alpha kept with `render.background: "transparent"` |
| `dir/` (no extension) | every frame `NNNNNN.png` plus channel PNGs and `meta.json` |
| `dir/%06d.exr` | one multilayer OpenEXR per frame, scene-linear, premultiplied, before exposure |
| `x.mp4` | H.264 |
| `x.mov` | ProRes 4444 RGBA (needs `background: transparent` for alpha) |

PNG filename frame patterns (`###`, `%d`) are usage errors: use a directory or `%06d.exr`.

`--set KEY=VALUE` takes `resolution` (`WxH`, `W,H` or JSON), `fps`, `seed`, `samples` (1..4096), `denoise`. `--set seed=42` makes sampling repeatable on the same backend. `--profile` may also be a profile file (`{"version":1,"level":1,"settings":{...}}`, levels 0 draft, 1 final, 2 path, 3 previs).

## Channels

`--channels color` (default), `depth`, `object_index`, `view_normal`, `motion`, `camera`, `aov:NAME`; `all` includes `render.aovs`. Applies to every `--out`.

- PNG: depth/object_index grayscale16 (depth metres per unit in `meta.json` `frames[].depth_unit_m`); view_normal RGB8 `(v+1)/2`; motion RGB16 clamped to +-1 of image extent; shadow_catcher RGB16 clamped 0..1; aov 16-bit clamped 0..1.
- EXR: `R,G,B,A`; `Z` metres (background 1e10); `view_normal.X/Y/Z`; `object_index.X`; `motion.X/Y` pixels, +Y down, current minus previous; `shadow_catcher.R/G/B` unclamped; aov `<name>.X` or `.R/.G/.B`; float32. Prefer EXR for precise data.
- `--exr-codec none|rle|zips|zip|piz|pxr24|b44|b44a|dwaa|dwab` (default `zip`; `pxr24`/`dwa*` are lossy).
- Differences from Blender: `view_normal` is camera space; `motion` is current-minus-previous with +Y down.

## Sidecars

`OUT/meta.json` (directories) or `OUT.meta.json` (files): fps, per-frame camera and depth unit, `object_ids` (pass_index -> node), exposure summary, `warnings [{code,requested,dropped}]`. `camera.json` when camera is a channel (always for video). The exposure summary reports clipping, near-black share and mean luma; transparent renders measure covered pixels only.

**Always read `warnings`**: success does not mean every requested effect was applied.

## Display

`render.display` is a display program (default `std/agx`; also `std/standard`, `blender/filmic`); inputs `exposure` (stops), `gamma`, `dither_intensity`, AgX `look`. EXR output stays scene-linear. White balance is a compositor step (`vork-compositing-vfx`).

## Cost and diagnostics

- `--diagnostics` writes `NAME.timings.json` (adapter, cache, GPU, encoder timings, non-finite pixel counts). With it, `--no-write` skips files and `--gpu-only` skips readback, for benchmarking.
- `vork setup` reports the GPU and warms default shaders; `vork setup --all` warms the bundled catalog. Authored programs compile on first use (cold compile can take minutes).
- Env: `VORK_CACHE_DIR`, `VORK_CACHE_BYTES` (default 16 GiB; 0 = no disk cache), `VORK_SLANGC`, `VORK_OIDN`, `VORK_BIN`.
- Progress/logs go to stderr; capture with `>run.log 2>&1`. Elapsed time is not an ETA.
- Linux/Windows video output needs FFmpeg on PATH; native Mac H.264 uses VideoToolbox.

## Exit codes

0 success - 2 usage - 3 invalid scene or program - 4 device, GPU memory, encoder or setup - 5 filesystem IO. A consumer closing stdout early exits 0.

## Delivery recipe

```sh
vork render shot.json --profile std/previs --frame-start 0 --frame-end 5 --out check/   # block out
vork render shot.json --profile std/draft  --at 1.0 --out look.png                       # inspect
vork render shot.json --profile std/final  --out shot.mp4                                # deliver
```

Test the full shot at final quality for a couple of frames before committing a long batch, especially with glass or dense transparency.
