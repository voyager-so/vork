OUTPUT ENCODINGS
.png RGBA still (keeps alpha; use render.background "transparent"); t=0 unless --at; other channels beside it as NAME.<channel>.png
PNG filename frame patterns (two or more # characters or printf-style %d/%04d) are usage errors; use a directory or %06d.exr for frames. Parent directories and other filename characters are literal.
directory (no extension or trailing /): NNNNNN.png frames plus channel PNGs
PATH/%06d.exr: one multilayer OpenEXR per frame, scene-linear premultiplied, before exposure (a 2D canvas: its picture after its view, below)
.mp4 H.264 (requires opaque final alpha on every composited frame); .mov ProRes 4444 RGBA (keeps final alpha)
--at SECONDS: one still; needs .png, directory or EXR. EXR stills also require %06d
in the filename, even with --at; an unnumbered EXR filename is refused.
--frame-start/--frame-end N: inclusive
absolute frame indices, default the whole clip; not with --at.
Sidecars: OUT/meta.json for directories; OUT.meta.json for files (fps, per-frame camera and depth unit, object_ids pass_index -> node,
exposure, warnings [{code,requested,dropped}] for omitted/substituted effects); camera.json in directories or OUT.camera.json for files when camera is a channel (videos always; vork help camera-json).
EXR sidecar names remove %06d and trailing ._- from its filename prefix, keeping
the extension. A filename of %06d.exr uses directory sidecars in its parent.
The same naming rule applies to camera.json and timings.json.
PNG channels: depth and object_index grayscale16 (depth metres per unit in meta.json
frames[].depth_unit_m); view_normal RGB8 = (v+1)/2; motion RGB16, RG = 0.5+0.5*offset in image
extents, clamped to ±1; shadow_catcher RGB16 linear, clamped 0..1; aov linear 16-bit, clamped 0..1.
EXR channels: R,G,B,A; Z (metres, background 1e10); view_normal.X/Y/Z; object_index.X;
shadow_catcher.R/G/B (unclamped multiplicative light ratio); motion.X/Y (pixels, +Y down, current minus previous); aov <name>.X or <name>.R/G/B; float32.
depth: view-axis metres to the nearest surface, not blended across edges.
--exr-codec none|rle|zips|zip|piz|pxr24|b44|b44a|dwaa|dwab (default zip; pxr24/dwa are lossy).
Timings: OUT/timings.json for directories; OUT.timings.json for files, beside meta.json.

shadow_catcher follows Cycles: catcher material light with CG / without CG, including
shadows, bounce and reflections. A zero baseline component is neutral 1 without
division; light added by CG to an otherwise unlit catcher is not representable.
Combined always treats a catcher as transparent black on transparent film; opaque
film shows background times ratio. Holdouts are transparent black on transparent
film, opaque black on opaque film. Channel selection never changes combined.

meta.json is version 3: frame times are numeric evaluation seconds; fps is a
number or rational string. camera.json remains version 2.
Before writing, a run invalidates previous completion metadata, camera and timing
sidecars. Explicit frame ranges and --at stills preserve numbered PNG/EXR frames
outside the selection; selected frames and their old channels are removed, even
if rendering then fails. Whole-clip renders remove all owned old frames/channels,
including trailing frames. Unrelated files survive. Metadata and sidecars describe
only the current invocation, not accumulated frames. Serialize writers per
destination (concurrent writes fail "output already in use").
Only consume a shot after success with fresh meta.json; partial files can remain
on failure. Cleanup also matches native non-UTF-8 filenames.

--png-depth 8|16: PNG colour bits per component; default 8. Requires PNG colour
output for 16. Both depths use the same exposure and display transform. Standard
display dither scales to the output bit depth; control channels retain their depths.


OUTPUT DEVICES
Repeat --out with a path or JSON {file,display,peak_luminance,reference_white};
display defaults to sRGB. Display P3, Rec.2100-PQ and Rec.2100-HLG are built for PNG
stills and frame directories of a 3D world. The view's result is encoded for the
display on the GPU. The PNG names its display in cICP in place of the sRGB chunk
(Display P3 12,13,0,1; PQ 9,16,0,1; HLG 9,18,0,1). meta.json records output_display
and a PNG's output_png_depth for every viewed output, sRGB included.
Rec.2100 writes 16-bit PNG whatever --png-depth says. peak_luminance (cd/m², default
1000, at most 10000) is the display's peak; reference_white (default 203) is where unit
white lands; meta.json records both, and omits the SDR exposure report.
std/standard and views that return XYZ take any luminances. std/agx renders these
displays with look none and the library lut (Blender's own views); its other looks
and an authored lut are sRGB renderings and refuse them. Its HDR view is for
peak_luminance 1000 with reference_white 203 or 100 (the view as published); other
luminances refuse. SDR displays take no luminance fields at all: any value, 100
included, refuses.
Display P3 video (.mp4, .mov) carries the same code values, tagged with P3 primaries
(H.273 12) and the BT.709 transfer and matrix, as sRGB video is tagged BT.709.
Rec.2100 video is not built yet: video here is 8-bit; write PNG frames.
One render call encodes for one display: destinations asking for different displays
or luminances refuse; render each in its own call. A 2D canvas takes sRGB and
Display P3 (PNG and video): its view encodes the finished picture for the display.
A canvas's frame is 8-bit, so Rec.2100 on a canvas is not built yet.
EXR excludes display and luminances.
EXR takes {file,color_space}: its storage space. A world takes Linear Rec.709,
Linear Rec.2020 or ACEScg; omitted, the working space with the values as they are.
A 2D canvas writes %06d.exr too: its picture after its view, display-referred, premultiplied float RGBA, not clamped or rounded.
Omitted color_space stores the canvas's working space (sRGB by default: its code values as they are); sRGB, Display P3, Linear Rec.709, Linear Rec.2020 and ACEScg are accepted (a linear space stores display-linear light).
A canvas EXR does not make the canvas float: an 8-bit canvas stores the codes its PNG has (premultiplied).
Beauty and colour AOVs are converted; alpha, data channels and shadow_catcher are not.
Every EXR names its colour: chromaticities, vork:color_space, vork:reference_domain (scene for a world, display for a canvas).
