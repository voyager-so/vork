---
name: vork-blender-bridge
description: Bring Blender scenes into vork - headless .blend export (export.py), glTF geometry, material graph lowering to Slang, world/HDR translation, display/white balance, shot.json and shot.profile.json, coordinate and convention conversion, and what Blender features are unsupported. Use when importing a .blend or matching a Blender look.
---

# Blender bridge

`integrations/blender/` is a **separate GPL-3.0-or-later component**; it exchanges scene, geometry, shader and image files with vork and is not imported by the renderer. Keep that boundary (do not copy its code into the vork Preview-licensed crates). Full detail: `integrations/blender/README.md`, `examples/kitchen/` (a worked import).

## Export

```sh
blender -b --disable-autoexec source.blend \
  --python integrations/blender/export.py -- --out /tmp/vork-scene
```

Flags: `--animation` (scene-sampled clips, camera/light controls), `--no-apply-modifiers` (keeps native shape keys, reports omitted modifiers), `--bake-curves` (RGB curve lookup textures, fewer shader compiles). Requires Blender 5.2.1 (the standards vork follows).

Outputs: `shot.json` (scene), `shot.profile.json` (level 1 for EEVEE, 2 for Cycles, plus seed/denoise), `export-report.json` (every material variant and substitution, missing geometry/transport features, baked fields, source sample counts). **Read the report**: it lists what was substituted or dropped. No EEVEE rendering runs in export.

```sh
vork render shot.json --profile shot.profile.json --out shot.png
```

The profile owns the sample budget, not the source file's samples.

## Pieces

- Geometry: `geometry.export_gltf(**options)`; subdivision is evaluated by Blender at export (creases become evaluated surface, no `crease_edge` sharpness enters the scene).
- Materials: `material_export.MaterialWriter(directory)`; `writer.write(material)` returns an override ID; merge `writer.images/programs/materials` into the scene. Strict mode raises `UnsupportedMaterial` naming material/node/socket; `best_effort=True` allows Glossy and substitutions. Output is vork's OpenPBR response, **not** identical Blender BSDFs.
- World: `world.translate(world)` -> `WorldImport` (image, environment light node, background, materials). Supports Background with constant strength; constant RGB, Blackbody, or equirect Environment Texture; Mapping rotations; Is Camera Ray mixes. `UnsupportedWorld` names the node. Volume Scatter/Emission/Principled Volume have **no 0.2 renderer support** (participating media absent).
- Display: `display.translate(scene)` -> `render.display` (`std/standard`, `std/agx`, `blender/filmic`) plus white balance via `post/white_balance`. ACES, non-sRGB devices and custom curves raise named errors; compositor node trees are not translated.

## Convention conversion (by hand)

| Blender | vork |
|---|---|
| Z-up `(x,y,z)` | Y-up `(x,z,-y)` |
| quaternion `w,x,y,z` | `[x,y,z,w]` |
| Euler XYZ about Blender axes | same order about vork axes; a Blender Z turn is vork Y |
| frame f (1-based) | `t=(f-1)/fps` seconds |
| lights/cameras aim local -Z | same; identity is horizontal, aim with `target` |
| ease interpolation | `CUBICSPLINE` with `[t,v]` keys |
| Principled glass/SSS tint | `transmission_color` / `subsurface_color` (base_color tints neither) |
| Specular IOR Level 0.5 | `specular_weight 1` |
| `view_normal` world space | camera space in vork |
| `Vector` pass previous-minus-current | `motion` is current-minus-previous, +Y down |
| Filmic/AgX | `blender/filmic`, `std/agx` (AgX is default) |

Blender lights are matched by vork's normalized lights; sky: `vork std sky` (Nishita) gives a sun light that casts shadows. RGB background is camera-only.

## Workflow

1. Export, read `export-report.json`.
2. Render `std/draft` with the generated profile; compare against a Blender render of the same frame.
3. Fix substitutions in order of visual weight (camera, light, key materials) rather than all of them.
4. Unsupported volume/fog scenes: remove them; there is no replacement.
