# OCIO display transforms

The sole AgX sample asset is `libraries/std/agx.exr`. `std/agx` reads its
neutral band and nine look bands.
`agx-manifest.json` retains the provenance of the original neutral samples.
Only mathematical transform samples are imported; rendering and GPU integration
remain native vork/wgpu. See `libraries/std/agx-manifest.json` for the tiled
65³ half-float layout and `scripts/color/bake-agx-looks.py` for regeneration.

`std/agx` renders Display P3 and Rec.2100 destinations from Blender's own AgX
views for those displays: their original OCIO cubes, `libraries/std/agx-p3.exr`
and `agx-pq.exr` ("AgX - HDR 1000 nits", HLG), pinned by
`libraries/std/device-views-manifest.json` and baked by
`scripts/color/bake-device-views.py`. `device-reference.json` holds direct
OpenColorIO results for those views and for Standard; the display stage's GPU
tests compare its output for each display against them.
