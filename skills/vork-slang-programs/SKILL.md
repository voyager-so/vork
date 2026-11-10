---
name: vork-slang-programs
description: Write custom Slang programs for vork - surface, displacement, AOV, lit, environment, composite, camera, sensor, display and GPU compute stages, program Inputs and bindings, procedural textures, simulations, custom light paths and spectral/dispersion. Use when the built-in std/openpbr, std/camera or compositor cannot express the effect.
---

# Slang programs

Use built-in materials first. Reference: `vork help programs --all`, `vork help light-paths --all`, `vork help std --all`, `vork help blender --all`. Study `examples/black-hole/` (custom propagation), `examples/murmuration/` (compute flock), `examples/water/`.

## Declaring a program

Inline or file: `"program": {"file":"my.slang"}` (path relative to the scene) or a named entry in `programs`. A module does `import vork;` and declares one `struct Inputs { ... }` **whose initializers are the defaults**. Scene `inputs` bind to its fields.

Stages (a mesh material needs `surface`; `lit`/`displace` supplement it):

| Stage | Signature | Notes |
|---|---|---|
| surface | `Closure\|Blend\|Hair surface(ShadingContext, Inputs)` | `Closure c = {};` = OpenPBR defaults; `blend(a,b,w)`, `add(a,b)`, `mix_closure(a,b,w)` |
| displace | `float3 displace(DisplacementContext, Inputs)` | object-local offset after deformation; topology fixed; hair unsupported |
| aov | `void aov(ShadingContext, Inputs, inout AovOutputs)` | `write_aov(output, value)` |
| lit | `Closure lit(ShadingContext, Inputs, Lighting)` | return `unlit(rgb)` |
| environment | `float3 environment(EnvironmentContext, Inputs)` | radiance, on `light.material` |
| composite | `float4 composite(CompositeContext, Inputs)` | see `vork-compositing-vfx` |
| camera | `CameraRay camera(CameraContext, Inputs)` | on `camera.material` |
| sensor | `float3 sensor(SensorContext, Inputs)` | per arriving contribution |
| display | `float3 display(DisplayContext, Inputs)` | `render.display` |
| compute | `void compute(ComputeContext, Inputs)` | buffers `<binding>_load/_store/_count`; grids `<binding>_dims` |

## Inputs binding

Numbers, bools, enum tokens bind directly. `Image` <- `{"image"}`, `Aov` <- `{"aov"}`, `FloatGrid/VectorGrid` <- `{"grid"}`, `Attribute<float..float4>` <- `{"attribute"}` (read with `read_attribute(inputs.a, ctx)`; names/types must match every bound geometry or instance column). `[InputRange(min,max,min_closed,max_closed)]` validates (does not clamp). Numeric inputs animate via material tracks.

Restrictions: global resources and mutable globals are refused (use Inputs or compute buffers); initialized read-only tables and `static const` arrays are fine. `Closure.geometry_normal` is world space.

## Procedurals

`import std.procedurals;` for noise and ramps (wood, marble, stone). `import blender.procedurals;` for wave/brick/magic; `blender.nodes`, `blender.texture_coordinate`, `blender.color` mirror Blender nodes. Prefer `std/openpbr` for new scenes.

## Custom cameras and light paths

`camera()` returns `CameraRay` (origin, direction, `weight`, `wavelength_scale`, `t_min/t_max` in metres). `sensor()` gets `path_length` in optical metres, detector `wavelength` (nm) and returns a Rec.709 multiplier; a gate is `start <= path_length < end`, animatable through ordinary inputs. Spectral transport turns on automatically only when reachable programs use wavelength or dispersion. Curved propagation selects path transport; motion channels are then zero with a warning. Relativistic camera: wavelength scale D, weight D^5. Details: `vork help light-paths --all`.

## Compute and simulation

`compute` programs mutate retained buffers each frame; seeking replays history (so motion blur multiplies cost). Computed placement writes `TRANSLATION/ROTATION/SCALE` for opaque unskinned instances; `instances.count` selects a live prefix. Computed geometry with skinning or transparency uses the authored snapshot; displacement on computed/implicit geometry is ignored.

## Workflow

1. Copy the closest shipped `.slang` from `examples/` or `libraries/`.
2. Render tiny and cheap first (`std/previs`, 320x180, one frame). **First compile of an authored program can take minutes**; cache is `VORK_CACHE_DIR`.
3. Compile errors surface as exit code 3 with the Slang diagnostic; fix and re-run.
4. Inspect the image and `warnings` (unsupported custom resources fall back to OpenPBR with a warning).
5. `VORK_SLANGC` must point to the pinned `slangc` for source builds.
