# Python SDK

Requires Python 3.11+. Install the SDK from PyPI:

```sh
pip install vork
```

The SDK contains no renderer. For `std()` and `render()`, install it separately:

```sh
npm install -g @voyager-so/vork
```

Keep npm's optional dependencies enabled and use matching SDK and renderer versions.
Or use the [curl installer or platform archive](https://github.com/voyager-so/vork#install).
The renderer must be on PATH, or set `VORK_BIN` to the vork executable.
Types and scene writing work without the renderer.

`from vork import Scene, std, write_scene`
exposes the scene type, the shared CLI engines and scene writing:

```python
from vork import Scene, std, write_scene

shape = std('rounded_box', name='block', size=[2,1,1], radius=0.1)
scene: Scene = {'version': 2, **shape,
                'nodes': [{'type': 'mesh', 'geometry': 'block'},
                          {'type': 'camera', 'translation': [3, 2, 5], 'target': [0, 0, 0]}]}
write_scene(scene, 'block.json')  # block.json + its immutable buffer sidecar
```

`write_scene(scene, path, base_dir=None)` mirrors the TypeScript `writeScene`: resource paths
resolve against `base_dir` (default the working directory) and are written relative to the
scene; inline buffers of 4 KiB or more (std shapes, bulky mesh data) go to one
`PATH.buffers.UUID.bin` sidecar. The input is not modified.

`std(function, argument)` or `std(function, **options)` invokes `vork std` using JSON.
Set `VORK_BIN` to choose the executable. Maths, tracks and settings are plain Python and scene data.

`render('shot.json', 'still.png', at=0, profile='std/previs')` runs `vork render` with render-time
options (`at`, `profile`, `set`, `channels`); the scene file is unchanged.

Regenerate with `scripts/generate-docs`; check with `python3 packages/python/generate.py --check`.

Custom Slang programs declare vertex inputs as `Attribute<float>` through
`Attribute<float4>`. The generated `AttributeBinding` type keeps JSON bindings
such as `{"attribute": "COLOR_0", "channel": "rgb", "factor": [1, 0.5, 1]}`.
`read_attribute(inputs.tint, ctx)` returns value and screen footprints (`dx`, `dy`)
in raster and ray shading. Installation checks names and types on every bound mesh.

Scene writes complete the sidecar before atomically replacing the JSON, then remove
obsolete generated sidecars referenced by the replaced JSON. Referenced resources and
unrelated files are preserved. Serialize saves to the same destination and finish reading
its buffers before regenerating it; copy the whole scene tree for a lasting snapshot.
Fresh files use ordinary permissions (0666 masked by umask); replacement JSON preserves
its existing permissions. Failed publication removes the new sidecar and keeps the old scene.

`render()` prints the CLI exposure lines and returns `{exposure: [{output, exposure}], warnings}`.
`warnings` contains `{code, requested, dropped}` records for requested effects
omitted or substituted by the renderer, also printed prominently by the CLI and
written into each render metadata sidecar. Check this array before accepting a render.
Each summary contains `frames`, `coverage` (mean alpha), `clipped`, `near_black`, `mean`, and
`worst_clipped`/`worst_near_black` (`share`, absolute `frame`). EXR and control-only
outputs have no entry. Statistics weight display RGB by alpha; empty coverage gives
null metrics and worst-frame entries. Clip averages exclude empty frames. `clipped` is a display
channel at 99% or more, where a view ran out of range; a canvas with no world, view or
compositor shows authored colours as they are and gives null `clipped` and `worst_clipped`. Values
retain full precision from the output metadata.

Surface programs author emitted light once with OpenPBR `emission_color` and
`emission_luminance`; hits and light samples share that evaluation. Use `lit`
for authored glow that should not become a sampled light.

Scene JSON may include `"$schema": "https://voyager.so/vork/schema/scene.schema.json"` for editor
completion and validation. This metadata does not affect rendering and is never
fetched by vork. `vork init` includes it automatically. SDK writers preserve an explicit `$schema`
but do not insert one. SDK render helpers require the current completion metadata
contract, including its `warnings` list.
