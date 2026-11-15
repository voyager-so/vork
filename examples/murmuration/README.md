# Murmuration over a dusk lagoon

A camera skims through **one million birds** simulated with scene-owned Slang
compute programs. The camera and nine foreground birds are authored; the
simulated million do not flap. A bounded seven-neighbour approximation steers
the flock. Nearby birds use a compact mesh stream; distant birds use silhouettes.

```sh
python3 examples/murmuration/prepare_launch.py /tmp/vork-birds
vork render /tmp/vork-birds/scene.json --profile std/previs --at 12 --set resolution=320x180 --out /tmp/birds.png
vork render /tmp/vork-birds/scene.json --profile std/final --frame-start 216 --frame-end 407 --out /tmp/birds.mp4
```

On Apple silicon, frame ranges of this scene can lose the GPU and stop with "frame discarded" ([issue #10](https://github.com/voyager-so/vork/issues/10)); there is no workaround yet and a fix is planned.

The shot starts nine seconds into the simulation; seeking replays that history.
The README stills came from an earlier draft profile. These commands use the
shipped profiles and do not reproduce their quality or timing.
The preview omits the flock's reflections. Keep [CREDITS](CREDITS) with renders.
No video or downloaded setting assets are committed.

## Birds and flight

The flock uses [Crow](https://sketchfab.com/3d-models/crow-d5a9b0df4da3493688b63ce42c8a83e2)
by [Alexei Ostapenko](https://sketchfab.com/alexanders823),
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/): 2,208 triangles, feather
alpha and a normal map. The flight pose comes from frame 210 (8.75 seconds) of
`TakeOff`; the first part of that clip is perched motion.

Close birds combine the **Starling** museum scan (CC0) with crow-derived flight
wings. The scan is welded, decimated, cropped to remove its perch, pitched into
flight, and given a selected-to-active tangent-normal bake. Detached scan fragments
are removed by connected surface identity. This is an authored hybrid flight
model; the original scan has folded wings and no flight rig.

The close birds use native upstroke/downstroke blend shapes with independently
shifted pose keys; their animated parent motion stays attached to the body.
Placement banks each bird using lateral acceleration; headings follow velocity.

`integrations/blender/prepare_birds.py` records the offline Blender 5.2.1 asset build. Run with the
licensed source downloads in `SOURCE/starling` and `SOURCE/crow`:

```
blender -b --python integrations/blender/prepare_birds.py -- SOURCE examples/murmuration/assets/birds
```

The compact derived meshes and textures are committed. The large source scan is
not required at runtime. Full attribution, source URLs, licenses and modifications
are in [CREDITS](CREDITS); keep that file with renders.

## Setting

Poly Haven's CC0 Qwantani Sunset Pure Sky and Grass Bermuda 01 textures are
downloaded by `prepare_launch.py`, verified against `assets.lock.json`, and kept
out of Git. The grass glTF omits its published alpha map, so an ordinary material
binding supplies that separately pinned mask. The authored bloom graph softens highlights.
The water uses OpenPBR's water IOR and low roughness.

## Simulation and limits

A deterministic 32³ grid retains four representatives per cell; each bird chooses
seven neighbours from at most 108 candidates. This approximates seven nearest
neighbours, with less local detail in dense cells. The 20-bit ID packing supports
at most one million birds.

Clear → grid → advance → place remain separate dispatches. Advance reads the old
generation and writes scratch; place commits after neighbour reads finish. The
seed and fixed 60 Hz schedule govern replay, independently of camera, render FPS
and seeking. The roost and predator are forces, not prescribed paths.
