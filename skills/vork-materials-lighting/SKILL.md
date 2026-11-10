---
name: vork-materials-lighting
description: Materials, textures and lighting in vork - std/openpbr inputs, metals, glass, thin-walled panes, subsurface, normal maps, image inputs, area/sun/environment lights, std/sky, HDR environments, light linking, and choosing a profile for the lighting. Use when a vork render looks wrong in colour, shading, reflections, glass or illumination.
---

# Materials and lighting

Fields: `vork help field material.inputs`, `vork help materials --all`, `vork help lights --all`, `vork help images --all`.

## Materials

A material is `{program, inputs, double_sided, opacity}`; `program` defaults to `std/openpbr` (OpenPBR Surface 1.1.1).

- `base_color` is linear RGB. `base_metalness: 1` = metal. `specular_roughness` sets highlight width.
- Other groups: `specular_*`, `transmission_*`, `subsurface_*`, `coat_*`, `fuzz_*`, `thin_film_*`, `emission_luminance`/`emission_color`, `geometry_opacity`, `alpha_cutoff`, `geometry_normal`, `geometry_tangent`, `geometry_coat_normal`, `transmission_dispersion_*`.
- Glass tint is `transmission_color`; scattering tint is `subsurface_color`. `base_color` tints neither. Blender mapping: `specular_weight 1` = Specular IOR Level 0.5; `thin_film_thickness` in micrometres.
- **Window/pane glass**: `transmission_weight: 1`, `geometry_thin_walled: true`. **Solid glass** needs closed meshes; model hollow vessels as a shell. With `transmission_depth=0` a closed solid shows the authored colour; each thin sheet applies the full tint.
- Leaves, paper, curtains: `geometry_thin_walled: true`, `subsurface_weight > 0`, set `subsurface_color` and `subsurface_scatter_anisotropy` (positive favours transmission). Radius controls apply to thick objects.
- Emissive surfaces: `emission_luminance` + `emission_color`. Check in the image whether the emitter lights its surroundings at your profile; use `std/path` if indirect light matters.
- Opacity: `geometry_opacity` with `alpha_cutoff` (constant or texture). `opacity: coverage|transmittance` chooses the model.
- Numeric inputs animate with material tracks (`["materials",name,"inputs",input]`).

## Textures

Declare in `images`, bind as `{"image": name}` (or inline). The program owns channels and UV transforms. `color_space`: `sRGB` | `Linear Rec.709` | `Non-Color`; null means hdr/exr/ies linear, others sRGB. **Data maps (normal, roughness, masks) must be `Non-Color`.** Formats: png jpg tif hdr exr ies mov mp4 mkv or `%04d` sequences (video needs ffmpeg/ffprobe). 8-bit may be block-compressed; use 16-bit for exact data. Normal maps: tangent-space +Y. Render-to-texture via `image.render`: `vork help images --all`. `image.time` (seconds) retimes footage.

Procedural looks: write a Slang surface program (`vork-slang-programs`), `import std.procedurals`; don't hunt for a sample scene.

## Lights

`type: light`, `shape`: `sun` | `point` | `spot` | `rectangle` | `ellipse` | `environment`. `energy`: sun W/m², local W, environment multiplier. `size`: rectangle side / ellipse diameter. Area lights emit along -Z: use `target`. Normalized like Blender 5.2.1 lights.

- Exactly **one environment light** for sky/HDR lighting; extras are ignored with a warning.
- HDR: `vork std environment` ({file, output, name?, threshold? default 10}) preprocesses into an environment plus a sun; or bind `std/environment_image` to an Image (lat-long, +Y up).
- Sky: `vork std sky` returns Nishita atmosphere plus a declared sun that lights and casts shadows. Sun direction `[cos(el)*sin(rot), sin(el), -cos(el)*cos(rot)]`, rot 0 = -Z, pi/2 = +X.
- Light linking: `illuminates`, `shadowed_by` (`[]` = no shadows).
- IES: `ies` image files (LM-63 Type C).

## Choosing a profile for lighting

- `std/previs`: direct only; broad-light highlights can mislead. Never judge lighting here.
- `std/draft`/`std/final`: scene-ray lighting that **approximates** indirect; interiors and glossy bounces can darken. More samples do not restore omitted transport.
- `std/path`: use for interiors, glass, caustics, dispersion. Finite bounces and clamping still apply.
- Refracted caustics require `std/path`. Final's glass shadows are straight-line only; roughness does not spread them.

## Symptoms

| Looks like | Try |
|---|---|
| Dark interior in final | `--profile std/path` |
| Flat/grey scene | add an environment light; RGB background doesn't light |
| Washed or crushed colours | check linear vs sRGB inputs; display is AgX by default (`render.display`) |
| Glass opaque or black | closed mesh for solid, or thin-walled for panes; use path |
| Texture looks wrong | `color_space`, `Non-Color` for data, v=0 top |
| Noisy | `--set samples=N`, keep denoise on (default) |
| Glass aborts, exit 4 "Impacting Interactivity" (Metal) | retry, thin-walled, path, or NVIDIA |
