UPGRADE
Breaking changes, newest first. Rewrite scenes and scripts for the current API.
Use matching SDK and executable versions; regenerate SDK-written scenes after upgrading.

0.6.0 (from 0.5.1)
A 0.5.1 scene either loads unchanged or is refused in one line naming the edit below. No aliases.

Scene fields that moved (refused; rewrite as shown, the picture is unchanged)
- render.display is composition.view. A composition states its space; an old scene is a world, "3d".
  before: {"render": {"display": {"program": "std/standard", "inputs": {"gamma": 1.2}}}}
  after:  {"composition": {"space": "3d", "view": {"program": "std/standard", "inputs": {"gamma": 1.2}}}}
- render.compositor is composition.compositor; a rendered image's graph moves beside its render.
  before: {"render": {"compositor": ["bloom", "grade"]},
           "images": {"cap": {"render": {"camera": "c", "compositor": ["blur"]}}}}
  after:  {"composition": {"space": "3d", "compositor": ["bloom", "grade"]},
           "images": {"cap": {"composition": {"compositor": ["blur"]}, "render": {"camera": "c"}}}}
  An image's composition takes no space.
- Compositor graph bindings name a step, not a node.
  before: "inputs": {"source": {"node": "glow"}}
  after:  "inputs": {"source": {"step": "glow"}}
- Animation layer blend add is combine.
  before: {"blend": "add", "tracks": [...]}
  after:  {"blend": "combine", "tracks": [...]}
- blender.nodes MathOperation.FRACTION is MathOperation.FRACT; an enum input "fraction" is "fract".
- import std.particles is import fluid.particles; the functions are the same.

Removed; you choose the replacement
- View input dither_intensity (std/agx, std/standard, blender/filmic). Output encoding dithers a world's
  picture, scaled to the output's bit depth. Delete the input: 1, the default, draws as before.
  0 has no setting: write 16-bit PNG or EXR for less dither. Above 1, as grain, is a compositor program.
  before: {"program": "std/agx", "inputs": {"dither_intensity": 0}}
  after:  {"program": "std/agx"}
- post/white_balance. std.white_balance is a library; write the stage as your own composite program.
  before: "render": {"compositor": [{"program": "post/white_balance", "inputs": {"temperature": 5200}}]}
  after:  "programs": {"balance": {"source": "import vork; import std.white_balance;
            public struct Inputs { float temperature = 6500.; float tint = 10.; };
            public float4 composite(CompositeContext ctx, Inputs inputs) {
              WhiteBalanceInputs w; w.temperature = inputs.temperature; w.tint = inputs.tint;
              float4 c = compositor_sample(ctx.uv); return float4(white_balance(c.rgb, w), c.a); }"}},
          "composition": {"space": "3d",
            "compositor": [{"program": "balance", "inputs": {"temperature": 5200}}]}
- std/text align_x justify and flush. std/text does not wrap: write left, center or right, place
  one std/text per line, or use wrapped canvas text for justification.
  before: {"align_x": "justify"}
  after:  {"align_x": "left"}
- View programs. float3 display(DisplayContext, Inputs) is ViewResult view(ViewContext, Inputs), and
  std.display is std.view. The engine encodes and dithers once; remove a program's own dither.
  before: import vork; import std.display;
          float3 display(DisplayContext ctx, Inputs i) {
            return finish_display(map(ctx.color), ctx, i.gamma, i.dither_intensity); }
  after:  import vork; import std.view;
          ViewResult view(ViewContext ctx, Inputs i) {
            return finish_view(map(view_linear_rec709(ctx)), ctx, i.gamma); }
  Return view_from_xyz(XYZ D65 in cd/m²) or view_from_codes(codes, ctx.display); finish_view builds
  the result from sRGB codes. ViewContext has no bit_depth. blender.display_lut takes a ViewContext.

Images and footage: embedded colour is read (0.5.1 read every file as sRGB, HDR/EXR as Linear Rec.709)
- A file's ICC profile, PNG cICP, gAMA and cHRM chunks, EXR chromaticities and footage colour tags now
  set its colour, so a tagged wide-gamut file draws its true colours. To keep the 0.5.1 picture:
  before: "images": {"plate": {"file": "plate.mov"}}
  after:  "images": {"plate": {"file": "plate.mov", "color_space": "sRGB"}}
  Write "Linear Rec.709" for an EXR or HDR. An explicit color_space always overrides the file.
- Refused until decoded: PQ or HLG files; footage tagged Rec.2020 or DCI-P3; cICP that is not
  full-range RGB or names another pair; CMYK ICC profiles. Choose "sRGB" for the 0.5.1 picture, or
  convert the file. Footage tagged Display P3 asks you to choose "sRGB" (0.5.1 picture) or
  "Display P3" (tagged colours).

Used to render, now refused (the line names the choice)
- Solid glTF transmission coloured by COLOR_0 or a base-colour texture under screen refraction:
  use ray refraction or std/path. Constant colour factors still render.
- glTF KHR_materials_ior below 1, the 0 sentinel included: write a finite ior of at least 1.
- A fog path over its work budget refuses the frame; 0.5.1 warned fog_watchdog and delivered it.
  Simplify the scene or render fog separately.
- An environment program returning negative or non-finite radiance on a camera ray refuses the frame.
- std/hair random outside 0..1 (0.5.1 clamped it).
- Images, fonts and other file inputs that are not regular files.
- Movies whose frame times FFmpeg below 5.1 cannot write exactly: install FFmpeg 5.1+.
- WGPU_BACKEND=vulkan on macOS: macOS renders with Metal only. Unset it or select metal.
- An --out value beginning with { is read as an output descriptor: {file, display, ...}. A path still
  means sRGB.

Look changes (the scene loads; rerender representative frames)
- std/final without denoising (--set denoise=false, or no fast GPU denoiser) returns the filtered
  samples, keeping detail and noise; 0.5.1 smoothed them. Keep denoising on, or raise samples.
- std/draft and std/final with more than 8 analytic lights pick lights per sample: unbiased, noisier.
- std/hair random defaults to a per-strand random value, previously 0. Visible with random_color or
  random_roughness above 0; write "random": 0 for the old look.
- A glTF primitive with no material is single-sided; 0.5.1 drew both faces.
- glTF transmission tints at each crossing by factor × texture × COLOR_0.
- OpenPBR specular_weight on dielectrics scales the optical IOR, for reflection and refraction; 1 is
  unchanged. std.openpbr_core reflection_fresnel follows; reflection_fresnel_at_ior gives the old term.
- blender.nodes Transparent_BSDF(Color) tints by Color; 0.5.1 ignored it. Pass float3(1.) to keep it.
- std/text lays out with the canvas text engine: align_y top, center and bottom use CSS line boxes,
  so the origin shifts for fonts whose ascent plus descent is not 1 em. Baseline alignments hold.
- std extrude d flattens with a different curve subdivider at the same tolerance; vertices can differ.
- Environment programs are evaluated per camera sample, not from a bake: backgrounds sharpen.
- Fixes that move pixels: reflections inside dielectrics, coat over thin transmission, GI behind rough
  glass, fog over glass, index-matched transmission, formerly black shadowed receivers.

Outputs and automation
- Non-finite samples paint opaque magenta and the frame succeeds, with warning nonfinite_samples
  (was nonfinite_path_samples, refusing above 0.1%) and meta.json invalid_pixels.
- Exit 3 means the scene or a program is at fault. An error with no category exits 4 (5 for
  filesystem IO); 0.5.1 exited 3 for it, so a GPU fault could read as an invalid scene.
  Device limits are among them: a render.resolution beyond the GPU's texture side, or a video
  frame larger than it, exits 4; 0.5.1 exited 3.
- A frame refused for an authored program's result (a camera program's invalid ray, a medium
  program's invalid coefficients, an environment program's negative or non-finite radiance)
  exits 3; 0.5.1 exited 4. Unchanged: a frame refused for non-finite radiance exits 4, and so does
  a non-finite colour after a world's view, since a view can pass non-finite radiance through.
- Every written render writes timings.json; --diagnostics adds memory and non-finite counts.
  Diagnostics timings are version 4. meta.json stays version 3 and gains output_display.
- Colour PNGs carry the sRGB chunk; EXRs carry chromaticities. Pixels are unchanged.
- SDK types drop Render.display, Render.compositor, ImageRender.compositor, NodeBinding and the blend
  "add"; regenerate scenes. The Python SDK refuses a renderer of another version and prefers its
  installed wheel over VORK_BIN and PATH.

Not in 0.5.1 (spellings from development builds)
- audio.preservesPitch is time.pitch: follow (false) or keep (true); keep is not built yet.
- Canvas CSS/SVG style tracks interpolate linearly by default; set interpolation STEP to hold.

0.5.1 (from 0.5.0)
- No breaking changes.

0.5.0 (from 0.4.0)
- Depth selects the nearest visible triangle at folded edges, without averaging
  disconnected surfaces sharing an object ID.
- Fluid status reporting uses status bindings, geometry.status and a seven-element
  correction buffer; see vork help fluid.
- Box, sphere, grid, cylinder, cone, torus, capsule and revolve remain std builders.
  std/bloom, std.procedurals and std.vector remain supported.
- Replace post/white_balance with std.white_balance and std.particles with
  fluid.particles. Colour conversion, ramps, mix and float_curve live in std.color;
  curve_segments remains Blender-specific.
- std/path and custom level-2 profiles default to 64 samples, previously 4096.
  Explicit samples overrides retain their budget, independently of denoising.

0.4 (from 0.3)
- Replace std/medium_constant and std/medium_height with authored medium programs;
  attachment, coefficient and transport contracts: vork help fog.

0.3 (from 0.2)
Scene and script changes
- Removed std/fog and Blender volume programs fail lookup even when unused.
  Rewrite media using the current fog contract; only one environment is allowed.
- glTF hierarchies must be acyclic, including unselected nodes.
- Environment preprocessing refuses output/source aliases, including symlinks.
- Previously ignored geometry effects now refuse with an authoring alternative.
- Named geometry_opacity/alpha_cutoff bindings now affect camera and ray hits;
  computed instance attributes follow GPU updates, shutter samples and seeks.
  Custom surfaces retain their program and animation. Remove compensating bindings.
- SDK render at and environmentImage/environment_image fps take numbers.
  Scene clocks, CLI --at and render --set fps still accept rational strings.
- SDK writers preserve explicit $schema but no longer insert it. Completion
  metadata requires warnings; upgrade SDK and executable together.
- Writers retire obsolete generated buffers referenced by the destination's old
  JSON. Copy JSON and buffers for snapshots; serialize saves and reads. New files
  use 0666 subject to umask; replacement JSON retains permissions.
- Transforms, weights, speeds and shutter width stay numeric. Exact clock
  arithmetic covers frames, media, simulation ticks and discrete decisions.
  Recheck cut/STEP and footage boundaries under reversed/looped layers.

Look changes (rerender representative frames)
- Set material.priority for overlapping solids; remove artificial air gaps at
  touching glass/liquid boundaries. Ownership: vork help field material.priority.
- Normal maps follow glTF/MikkTSpace. Remove compensating green-channel flips or
  incorrect tangent.w signs; recheck base/coat normals and anisotropy.
- Path indirect clamping uses mean absolute RGB, limit 10; saturated indirect
  lighting can brighten. Direct light and raster GI are unchanged by this fix.
- Raster ambient/GI uses the dielectric diffuse response consistently. Recheck
  mixed metals, rough/high-IOR materials and subsurface; remove old compensation.
- Compositor images are premultiplied before filtering. Custom composites must
  return premultiplied RGBA. Captures use parent shutter pose/time and retain
  dependencies even when the compositor replaces camera beauty.
- Recheck linked/transmitting shadows and animated attributes; remove stale-data
  workarounds.

Outputs and automation
- MP4 requires opaque final alpha on every frame; use .mov for transparency.
- Overwrites invalidate completion metadata and remove owned old outputs even
  on failure. Snapshot outputs first; serialize writers and require success plus
  fresh meta.json. Selection and cleanup rules: vork help outputs.
- meta.json and diagnostics use version 3 with numeric frame times; fps remains
  number/rational string and camera.json stays version 2.
- Setup builds the path engine and warms the starter at std/final; its shader deadline can defer compilation.
  Required checks still fail. Use --skip-warmup for checks, --all for unbounded
  warming; completed cache entries survive reruns.
- Setup tool failures exit 4, not 5; see vork help errors.

0.2 (from 0.1)
- Before upgrading scatter scripts, save generated instance columns and buffers
  with 0.1. A seed cannot reconstruct old placements. Replace scatter with
  authored, equal-length instance columns, preserving rotations, scales and
  custom material bindings; see vork help field node.instances.
- Replace TS renderArguments / Python render_arguments with SDK render, or
  explicit CLI arguments. Keep the existing render options.
- Replace post/vignette, post/lens_distortion, post/glare and std/highlight with
  authored composite programs. The old lens effect was RGB dispersion; additive
  glare needs a merge with the original. std/bloom remains supported.
- Omitted mesh materials now use specular_roughness 0.3, previously 0.5.
  Set it explicitly to retain the old look; imported/explicit materials are unchanged.
- TS sky/environment options: rename color_space to colorSpace, sun_elevation to
  sunElevation, sun_rotation to sunRotation, air_density to airDensity,
  aerosol_density to aerosolDensity, ozone_density to ozoneDensity, sun_size to
  sunSize and sun_intensity to sunIntensity. Python/JSON remain snake_case.
- Replace @usevoyager/vork with @voyager-so/vork in imports and manifests.
  Installation: vork help sdk.
- Draft/final denoising uses scene-linear units independently of exposure;
  undenoised reconstruction also changed. Recheck dim/high-exposure detail.
- Supplied normals opposing triangle winding are aligned; degenerate normals use
  face normals. Recheck formerly black metals and authored/imported shading.
- Path no longer overcounts finite-source lighting at matched null-solid boundaries.
- Do not parse stdout by line position or treat stderr progress as failure.
  Check exit status and metadata/SDK exposure and warnings.
