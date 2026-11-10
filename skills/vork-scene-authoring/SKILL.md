---
name: vork-scene-authoring
description: Author vork scene JSON - nodes, transforms, targets, std shape builders, inline and buffer geometry, glTF/GLB assets, instances, implicit surfaces, computed placement and visibility. Use when building or editing the structure of a vork scene, importing a model, placing many objects, or converting coordinates.
---

# Scene authoring

Top-level `Scene`: `version: 2`, `nodes`, `geometry`, `materials`, `images`, `buffers`, `programs`, `animation`, `render`. Resources accept a **name or an inline definition**; named tables are `{name: definition}`. Names exclude `/`. Add `"$schema": "https://voyager.so/vork/schema/scene.schema.json"` for editor validation (no fetch happens).

Exact fields: `vork help field node.translation`, `vork help nodes --all`, `vork help geometry --all`, `vork help buffers --all`.

## Nodes

`type`: `node` | `mesh` | `asset` | `camera` | `light`. Type fields are flat siblings. Common: `name` (globally unique, imported nodes included), `translation`, `rotation` (`[x,y,z,w]` quaternion or `{"euler":[x,y,z]}` Blender XYZ order), `scale`, `children`, `visible`, `visible_camera`, `target`.

- `target`: a node name or `[x,y,z]`; aims local -Z with world Y up; `rotation` is an offset on top. It is structural, **not animatable**: aim at a named node and animate that node's translation.
- Any zero world-scale component hides the geometry, children included.
- Visibility per ray type is ANDed with ancestors. `pass_index` is inherited; otherwise object IDs are assigned stably and listed in `meta.json` (0 = background).
- A painted backdrop card: disable `visible_shadow`, and other ray visibilities if it should show only to the camera.

## Meshes and std shapes

`{"type":"mesh","geometry":{"std":"sphere","radius":1},"materials":[...]}`. Omitted materials = `std/openpbr` with roughness 0.3.

Builders (shapes centred, metres, round shapes use Y as axis): `box`, `sphere`, `grid` (XZ plane facing +Y), `cylinder`, `cone`, `torus`, `capsule`, `rounded_box`, `revolve` (profile `[radius,height][]`), `text` (needs a TTF/OTF `font` path), `extrude` (XY outline + holes, optional bevel). Full signatures: `vork help std --all`.

CLI: `vork std NAME JSON` returns a result (`-` reads stdin); `vork std expand` emits buffer geometry.

A rod from A to B: `cylinder` with `height=|B-A|`, centre at `(A+B)/2`, rotate local +Y onto `normalize(B-A)`; reject coincident endpoints.

## Raw geometry

Small meshes inline; large arrays in binary files (tightly packed, little-endian, `vork help buffers --all`). Attributes follow glTF semantics (`POSITION`, `NORMAL`, `TEXCOORD_n`, `COLOR_n`, indices). `COLOR_n` is **linear** float 0..1, not bytes; bind it explicitly: `base_color: {"attribute":"COLOR_0"}` (only imported glTF materials multiply by `COLOR_0` automatically). Polygons, subdivision creases and morph targets: `vork help geometry --all`.

Implicit surfaces: `topology: "implicit"` with one f32 dense buffer attribute (`dims:[x,y,z]` each >= 2). Surface is the zero of the trilinear field; negative inside; boundary samples strictly positive. Use surface materials without subsurface or displacement.

## Assets (glTF)

`{"type":"asset","name":"model","file":"model.glb"}`. Keep `.gltf` with its buffers/textures. Imported names: `asset/node-name` or `asset/#index`. Overrides via `materials` and `nodes` on the asset node (`vork help nodes --all`). Imported joints cannot be target, focus or attachment addresses. Skins and morph targets supported. Fit camera target/distance to the asset's size.

## Instances and computed placement

Many copies: author columns on the mesh, e.g.
`"instances":{"attributes":{"TRANSLATION":{"type":"vec3","data":[[0,0,0],[2,0,0]]}}}` plus equal-length `ROTATION` (XYZW) and `SCALE`; custom columns can drive material attributes. **Scatter builders no longer exist** - generate the columns yourself (seeded in your script). For GPU-driven placement (simulation, flocks), use compute programs writing `TRANSLATION/ROTATION/SCALE`; `instances.count` selects a live prefix. Computed placement is for opaque, unskinned geometry only. See `vork-slang-programs`.

## Coordinates

Right-handed, Y-up. From Blender: `(x,y,z) -> (x,z,-y)`; quaternions `[x,y,z,w]`; Blender frame f -> t=(f-1)/fps. Normal maps tangent-space +Y; TEXCOORD v=0 is the image top. Details in `vork-blender-bridge`.

## Workflow

1. Block out with std shapes and a grid ground; `std/previs`.
2. Fix framing and scale first (nothing else matters if the subject is mis-sized).
3. Swap in assets/materials; re-render.
4. Validate: scenes are checked at render time; exit code 3 means invalid scene, message names the field.
