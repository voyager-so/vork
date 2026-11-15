EXPORT — evaluated glTF 2.0 assets and native animation (0.5)

vork export SCENE --out ASSET.glb|ASSET.gltf [--root NODE]
            [--at SECONDS] [--frame-start FRAME] [--frame-end FRAME]
            [--allow-loss CODE]...

The default snapshot is time zero. --at also accepts rational seconds.
Omitting --root selects the whole scene. A root selects its subtree and retains
its evaluated world placement. Without a frame range animation is frozen at the
requested time. Supplying either inclusive frame endpoint delivers one composed
clip at scene FPS; omitted endpoints use zero/the last scene frame. --at conflicts
with frame ranges. Clip time starts at zero, and the report retains source frames.
Export needs no camera or GPU for native triangles and skin/TRS delivery.

Animation preserves native joint nodes, inverse bind matrices, JOINTS_0 and all
four WEIGHTS_0 influences over rest geometry. External joints and their ancestors
are included as transform dependencies for subtree skins; unrelated object
payloads outside the selection are omitted. Flat skinned meshes
omit NORMAL so shading remains flat after deformation. More influence sets,
morphs, post-skin compute/subdivision/displacement and imported asset expansion
remain refused; influences are never silently truncated.

Uncomplicated single-layer LINEAR/STEP TRS preserves source breakpoints and
interpolation. Layered, aimed, Euler-interpolated and cubic motion samples the
renderer evaluator at source frames, retains quaternion sign continuity and
requires --allow-loss animation_resampling: between-frame motion may differ.
There is no error-bounded key reduction.
Cameras and punctual lights animate through node TRS. Core glTF cannot animate
optics, light intensity, visibility, materials or layer policy; keyed properties
outside TRS/layer weights are refused instead of frozen in a clip. No animation
pointer extension is delivered. Exports exceeding 100001 source frames or one
million node samples fail with a smaller-range/subtree remedy.

Native triangles and standard shapes preserve hierarchy, mesh sharing, material
ranges, normals, UVs and authored tangents. Static instance TRS columns become
nodes sharing meshes; zero-scale instances stay hidden. Live counts, time-offset
instances and compute-written columns are refused. Native unbound colors are preserved
as _VORK_COLOR_0 rather than introducing glTF's implicit color multiplication.
std/openpbr core factors and still base-color/emission PNGs are supported.
Metallic/roughness images pack into G/B; sizes and samplers must match. Normal
images require authored tangents. HDR emission factors and clearcoat use required
Khronos material extensions.
OpenPBR and glTF BRDFs can produce different highlights. Other active inputs,
custom programs and unevaluated geometry stages fail with an authoring fix.

Standard projective cameras and punctual lights are delivered where representable.
Environment/area lighting, finite emitter extent, light links and lens blur are
reported losses. Whole-scene export also reports the default environment.
Read NAME.export.json and fix each issue before accepting an asset.
For an intentional loss, repeat --allow-loss CODE for its exact diagnostic code.
Refusals cannot be acknowledged away. light_representation omits area/environment
lights; environment_lighting acknowledges the missing default environment.
audio acknowledges omitted node sound; glTF sound export is not built yet.
Consumer lighting is outside the asset contract.

GLB is one self-contained file. glTF writes JSON plus a relative content-addressed
binary resource and PNG files. NAME.export.json records selection,
time/FPS, source hash, counts, required extensions and per-source dispositions.
No asset is published after an unacknowledged loss or refusal.

TS: exportAsset(scenePath, {out, root?, at?, frameStart?, frameEnd?, allowLosses?}) returns the report.
Python: export_asset(scene_path, out, root=None, at=None, frame_start=None, frame_end=None, allow_losses=None).
These call the CLI; writeScene/write_scene still write authored scene JSON.
