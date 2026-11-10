---
name: vork-compositing-vfx
description: Compositor graphs and VFX in vork - render.compositor entries, std/bloom, authored glare/vignette/chromatic dispersion, post/white_balance, display programs, shadow catcher and holdouts, and compositing rendered 3D over real footage with a tracked camera. Use for post effects, colour look, or adding CG to a plate.
---

# Compositing and VFX

Fields: `vork help compositor --all`, `vork help post --all`, `vork help std --all`. Worked examples: `examples/compositor-effects/`, `examples/vfx-artifact/`, `examples/vfx-neon/`.

## Compositor graph

`render.compositor`: ordered `material-name | {material, name?, resolution?}`; **the last entry is the output**. It runs after shutter accumulation/denoising and before exposure/display. No history between frames.

- `resolution`: `[w,h]` or a positive scale (floor, min 1); omitted = output size.
- Bind inputs: `{"node": earlier-name}`, `{"image": name}`, `{"channel": color|depth|view_normal|object_index|motion|shadow_catcher}`, `{"aov": name}`. Names are local to the graph.
- Slang entry: `float4 composite(CompositeContext, Inputs)`, scene-linear premultiplied RGBA; `compositor_sample(uv)` reads the previous result (the first entry reads beauty).

## Bloom (shipped)

One level:

```json
"compositor":[
 {"name":"highlights","material":{"program":"std/bloom","inputs":{"stage":"prepare","original":{"channel":"color"}}}},
 {"material":{"program":"std/bloom","inputs":{"stage":"combine","original":{"channel":"color"},"source":{"node":"highlights"}}}}
]
```

Tune `threshold` on prepare, `strength` on combine. Multi-level: chain `downsample` toward smaller images, `upsample` back, `combine` with the final source. Stages require their bindings; missing ones are scene errors even at strength 0. Follows Blender bloom arithmetic.

## Authored looks (glare, vignette, chromatic dispersion)

`post/vignette`, `post/lens_distortion`, `post/glare` were removed. **Copy the entire `examples/compositor-effects/` directory** into your project and use `{"program":{"file":"vignette.slang"},"inputs":{"factor":0.5}}`. Keep `highlight.slang` beside `glare.slang`. Glare returns an *additive* image: keep the original and merge it later (see `glare-streaks.json`). `chromatic-dispersion.slang` shifts RGB channels; it is not geometric distortion.

## Colour

`post/white_balance`: `temperature` 6500 K (1800..100000), `tint` 10 (-500..500); preserves alpha, linear outputs. Display transform and exposure: `render.display` (see `vork-render-outputs`). AgX is the default; `blender/filmic` for Filmic looks.

## Shadow catcher and holdouts

Channel `shadow_catcher` follows Cycles: catcher material light with CG / without CG, incl. shadows, bounce and reflections. EXR carries an unclamped multiplicative ratio; PNG clamps 0..1. Multiply the plate by the ratio, add the CG. A zero baseline is neutral 1; light *added* to an otherwise unlit catcher is not representable. Holdouts are transparent black on transparent film, opaque black on opaque. Combined output treats a catcher as transparent black on transparent film.

## CG over real footage

1. Track the plate externally; build a camera that matches (animate the camera node, or use the extracted K/R/t). Match fps, resolution, lens.
2. Add proxy geometry for occlusion/shadows/bounce; keep it off camera with `visible_camera: false` where it should only catch light.
3. Light with an HDR/environment matching the plate (`vork std environment`).
4. Render with `render.background: "transparent"`, channels `all` or `shadow_catcher`, `depth`, `object_index`, to EXR frames (`%06d.exr`).
5. Composite in the compositor (footage as an Image sequence with `image.time`, mattes as images) or outside.
6. Footage as an input: `.mov/.mp4/.mkv` or `%04d` sequences, needs ffmpeg/ffprobe; time zero is the first on-disk frame; end frames hold. `["images",name,"time"]` retimes.

`examples/vfx-artifact/README.md` is the base-composite recipe; `examples/vfx-neon/` adds match-move, wet-street reflections and reveal wipes.
