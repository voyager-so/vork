---
name: vork
description: Entry point for vork, the headless GPU 3D/VFX renderer driven by scene JSON. Use when asked to make, render, preview or iterate on a 3D scene, shot, animation or VFX with vork, or when a `vork` binary, `scene.json` with `"version":2`, or `vork help` is mentioned. Covers the first image and the render-inspect-refine loop, and routes to the other vork skills.
---

# vork

vork renders a scene described as JSON (or generated from Python/TypeScript) on a GPU:
Apple Silicon (Metal) or recent NVIDIA (Vulkan with ray queries). No editor. You write the
scene, render, **look at the image**, revise.

## The loop

```sh
vork init my-shot && cd my-shot          # scene.json + README.md, no GPU/network needed
vork render scene.json --out shot.png --profile std/draft
```

1. Render a cheap preview. **Open the PNG and inspect it** (framing, scale, shadows, colour). Never judge from the JSON.
2. Change one thing (camera, light position, material colour), re-render.
3. Raise quality only when the composition is right: `std/final` (default), `std/path` for glass, interiors, caustics.
4. For motion, preview a few frames (`--frame-start/--frame-end`, or `--at SECONDS`) before the whole clip.

Profiles: `std/previs` (blocking, no indirect), `std/draft` (4 spp), `std/final` (8 spp, default), `std/path` (4096 spp, accurate). Scene content and render quality are separate: switch profile with `--profile`, don't edit the scene.

## Rules that save time

- **Ask the binary, not memory.** `vork help` (overview), `vork help TOPIC --all` (full topic), `vork help field PATH` (one field, e.g. `vork help field node.target`). Unknown fields are errors; omit anything at its default.
- Units: metres, seconds, radians, linear Rec.709. Right-handed, **Y-up**, camera looks down local -Z. Paths are relative to `scene.json`.
- Colours are **linear** RGB in 0..1. Convert an sRGB picker value to linear first.
- Area lights emit along -Z; aim with `target`. Without an environment light the world is dim grey; an RGB `render.background` is only a backdrop, it does not light the scene.
- Read warnings. A render can succeed and still print `WARNING: REQUESTED EFFECT DROPPED`; the same list is in the `meta.json` sidecar.
- Read stderr/exit code on failure: 2 usage, 3 invalid scene/program, 4 device/GPU/encoder/setup, 5 filesystem IO.

## Minimal scene

```json
{"version":2,"nodes":[
  {"type":"mesh","geometry":{"std":"sphere","radius":1},
   "materials":[{"inputs":{"base_color":[0.12,0.35,0.65],"specular_roughness":0.3}}]},
  {"type":"mesh","translation":[0,-1,0],"geometry":{"std":"grid","size":[10,10]},
   "materials":[{"inputs":{"base_color":[0.35,0.35,0.35]}}]},
  {"type":"light","shape":"rectangle","translation":[-3,5,4],"target":[0,0,0],
   "energy":700,"size":4,"color":[1,0.92,0.8]},
  {"type":"camera","translation":[3,2,5],"target":[0,0,0]}
],"render":{"resolution":[640,480]}}
```

## Which skill next

| You need | Read |
|---|---|
| Nodes, shapes, imported glTF, instancing, buffers | `vork-scene-authoring` |
| Materials, glass, textures, lights, sky, HDRI | `vork-materials-lighting` |
| Camera, DoF, motion blur, animation | `vork-cameras-animation` |
| Output formats, passes, EXR, video, sidecars | `vork-render-outputs` |
| Bloom, glare, grading, compositing over footage | `vork-compositing-vfx` |
| Custom shaders, GPU compute, custom cameras | `vork-slang-programs` |
| Liquids | `vork-fluid-simulation` |
| Python / TypeScript authoring | `vork-sdk` |
| Importing from Blender | `vork-blender-bridge` |
| Install, build, GPU errors, limits, upgrades | `vork-setup-troubleshooting` |

## Not supported (do not promise)

Participating media (fog, smoke, volumes) are absent in 0.2. AMD, Intel and CPU rendering are unsupported. Scatter builders were removed (author instance columns instead).

## Prompt to give an agent

> Run `vork init my-shot`, read `vork help`, render a preview and inspect the image. Refine the camera and lighting before rendering at final quality. Read the relevant help topic when you need a feature the first page doesn't explain.
