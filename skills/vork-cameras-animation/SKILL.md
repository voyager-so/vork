---
name: vork-cameras-animation
description: Cameras, depth of field, motion blur and animation in vork - std/camera physical optics, focus_object, camera cuts, animation layers/tracks/glTF clips, interpolation, loops, frame timing, and camera.json export for tracking. Use when framing a shot, animating anything, or matching a camera to footage.
---

# Cameras and animation

Fields: `vork help cameras --all`, `vork help animation --all`, `vork help render --all`, `vork help camera-json --all`.

## Camera

`{"type":"camera","translation":[3,2,5],"target":[0,0,0]}`. `render.camera` picks which camera node (null = the only authored one, else first imported; a cut track may select).

- Default program `std/camera` follows the Blender physical camera. Optics are **material inputs** on `camera.material` (lens/sensor in mm, clip/focus in metres, `aperture_rotation` in radians). An explicit material must name `program: "std/camera"`. Exact lookup: `vork help field std/camera.lens`.
- Depth of field: sampled, enable with `use_dof` on `std/camera`. Focus via a focus distance, or `focus_object: <node>` for sampled camera-axis depth (follows animation).
- Orthographic is supported (`ortho_scale`).
- Custom lens/projection (fisheye, relativistic, gated sensors): `vork-slang-programs` (camera/sensor stages).
- Frame check first: previs at low resolution, then inspect scale and horizon.

## Motion blur

`render.motion_blur: {"shutter": 0.5}` (frames). Non-null enables it. Costs more, and simulation replay (fluid) multiplies cost.

## Frame timing

`render.fps` (default 24), `render.duration` (seconds, default 10.4167), `render.resolution`. Frames are 0-based absolute indices; `--frame-start/--frame-end` are inclusive; `--at SECONDS` renders a still. `--set fps=30`/`resolution=1280x720` override at the call.

## Animation

Ordered `scene.animation` layers, each with `tracks` or a glTF `clip`:

```json
"animation":[{"tracks":[{"path":["nodes","ball","translation"],
  "keys":[[0,[0,0,0]],[2,[2,0,0]]],"interpolation":"LINEAR"}]}]
```

- Paths are **segments**, not dotted strings: `["nodes",NAME,"translation"]`, `["materials",NAME,"inputs",INPUT]`, `["images",NAME,"time"]`. Name your nodes.
- Key times are seconds, increasing. `STEP`, `LINEAR` (numeric default), `CUBICSPLINE` (`[t,v]` keys). Strings/bools default STEP. Tracks hold their end values. Euler keys interpolate angles; quaternions take the short arc.
- Layers: weight, loops (repeat the layer span), `add` composes rotations. Bools and camera names switch at effective weight >= 0.5. Full layer/clip fields: `vork help animation --all`.
- Animatable: numeric non-structural scene fields including material and program inputs, display inputs, `visible` (step). **Not animatable**: settings, topology, `target`.
- Animated aim: `"target":"aim"`, add `{"name":"aim","translation":[0,0,0]}`, track `["nodes","aim","translation"]`.
- Coincident evaluated aiming origins (camera on its target) are errors naming node and sample time.
- glTF clips retarget onto imported assets (follows glTF 2.0 interpolation).

## Preview an animation

```sh
vork render scene.json --profile std/previs --frame-start 0 --frame-end 23 --out frames/
vork render scene.json --profile std/draft --at 1.0 --out mid.png
```

Inspect several frames, not one.

## Camera data for tracking / compositing

Video and EXR/multi-channel outputs write a camera sidecar (`OUT.camera.json`, or `camera.json` in directories): OpenCV `K`, `R`, `t`, `dist`, `image_size`, per-frame arrays, optional `orthographic`. Pinhole only; uncalibrated custom cameras omit K. Request `--channels camera`. `vork help camera-json --all`.
