LIMITS
Supported GPUs: Apple Silicon Metal, recent NVIDIA Vulkan and AMD Vulkan with
hardware ray tracing and storage buffer arrays (tested on Mesa RADV, RDNA3).
Intel GPUs are unsupported.
CPU rendering requires the explicit opt-in below. Run vork setup to check the
adapter. Windows x64 ships install.ps1;
macOS and Linux provide install.sh as a separate release asset.
Use Apple Silicon with macOS 15 or newer.
See README.md (Install) for binary setup.

Device-limit errors name the failing resource, limit and remedy.

Canvas documents support audio answers only.
A canvas inside a canvas cannot set viewBox or preserveAspectRatio.
A canvas frame draws at most 335,544 ordered items (each path, text or image
is one; a path under a mask or blend may take more). A larger frame is
refused with the count and the limit.

Homogeneous fog uses closed triangle carriers or constant/height world media
through complete-camera path transport. Arbitrary smoke, grids, procedural density
and custom surface callbacks remain unsupported. See vork help fog.

Additional environment lights also fail;
combine their contributions in one environment program or precomposite their maps.

std/final approximates indirect lighting and can omit bounced paths such as
camera → diffuse → glossy → light, with built-in or custom surfaces. A custom
surface material selects the older one-hit GI for the whole scene. More samples
do not restore omitted paths. Choose --profile std/path for full bounce lighting,
interiors, glass and caustics. Shipped path transport still has finite bounce budgets,
indirect clamping, adaptive sampling and denoising; it is not an unclamped oracle.
The indirect clamp limits each contribution's mean absolute RGB to 10, matching
Cycles' brightness measure. It scales all channels uniformly to preserve hue;
direct contributions are unclamped. Bright indirect lighting can still lose energy.
Path transport refuses `render.root` and reachable `image.render.root` subtree
renders on every device, including CPU Vulkan. Use `--profile std/final` with depth of
field and path-only effects disabled for subtree rendering, or render the
selected content in a separate scene without `root`. Computed attributes and their time/seek updates
do not have a separate CPU restriction.
Non-finite camera/light samples mark affected pixels magenta; output metadata lists their coordinates.
Final's rough glass shadows follow straight connections: roughness does not spread
transmitted shadows, light off-axis receivers, or test scattered-direction blockers.

On Apple Silicon, solid glass and heavy interiors can still abort with exit code 4
and "Impacting Interactivity". A retry can succeed; lower resolution is not a reliable
fix. For thin panes use geometry_thin_walled: true. Try std/path or a supported
NVIDIA or AMD GPU for repeated failures; path is slower and is not a universal
workaround.
Dense imported transmitting geometry has not been qualified on Metal. Large scenes
with dense transparency can lose the device in final even at small resolutions;
try path and test the full shot before committing to a batch.

Startup BRDF lookup generation uses bounded row submissions on Metal, retaining
the full lookup resolution and integration sample counts. This bounds the
scene-independent startup workload; lowering scene resolution does not reduce
that workload.

A scene that draws meshes or imported models can bind a limited number of distinct
attribute names across all its materials. Apple Silicon Metal fits 17. On Vulkan the
device's vertex-input and inter-stage limits decide; the refusal states the device's
limit and how many names it fits. A Gaussian capture binds 1, 4, 9 or 16 names at SH
degree 0 to 3, so a degree-3 capture beside meshes leaves one more name on Metal. A
scene of Gaussian captures alone has no such ceiling. Bind fewer distinct names.

Large point scenes and working buffers can exceed a device's per-buffer binding
limit even when total free VRAM is sufficient. Follow the error's resolution/profile
advice or simplify scene buffers. Scene import support does not guarantee rendering
at any size.

vork setup builds the prebuilt path engine (the same few pipelines for every scene) and
warms the vork init starter at the default std/final profile. Without setup, a cold path
render compiles its own scene kernel before the picture, then exits; it does not build
the shared engine afterward. On an NVIDIA L4, path-glass and path-many-lights at std/path
took about 50–60 s to picture and exit, and about 1 s on immediate repeat at authored resolution.
Repeats reuse the saved scene kernel.
With a warm engine, supported scenes draw with it while their own kernel compiles
in a detached background process; delivery and exit do not wait for it. Scenes the engine
cannot draw (authored programs, a participating medium, AOVs, splats, points or hair, or
cameras other than one std/camera, say) compile their own kernel before the first frame. Only
shader warming may stop successfully at its ten-minute limit; required GPU/tool
checks fail on errors or their own stage timeouts.
Use setup --skip-warmup for the GPU/tools report only, or setup --all for the
bundled program/profile catalog and built-in descriptors without the shader time limit. You can interrupt setup
safely and rerun it: completed device-local cache entries are retained.
Other scene kinds, authored programs and uncovered layouts compile once on first use. Cold compilation
can take minutes; setup does not guarantee instant first renders.

Hair displacement and lit stages are ignored with a warning. Curved propagation
selects path transport automatically; motion channels are zero with a warning,
while sampled shutter blur remains available.

Unsupported geometry effects refuse with the node and an authoring alternative.
Approximate effects can warn and render.
Legacy spatial environments evaluate at the origin. Unsupported light masks
are ignored with warnings. Environment programs sampling
render images fall back to the light colour. Inspect warnings and the image. Each dropped effect prints WARNING: REQUESTED EFFECT DROPPED,
with its code, authored request and omitted/substituted behaviour. Every render
meta.json sidecar and SDK render result returns the same structured warnings array;
warning-and-render success does not mean every requested effect was applied.

## Opt-in CPU rendering

On Linux, install recent Mesa Vulkan drivers (`mesa-vulkan-drivers` on
Debian/Ubuntu; `mesa-vulkan-drivers` or `vulkan-swrast` elsewhere), then run
`vork setup --gpu cpu` (or `VORK_GPU=cpu vork setup`) and `VORK_GPU=cpu vork render scene.json --out frame.png`.
The driver must expose ray queries and storage buffer arrays. CPU rendering is
slow and never selected automatically. Setup warms the default starter. CPU shader
warm-up has no time limit. First-run shader compilation can take several minutes. vork's driver
pipeline cache is disabled because it can trigger a lavapipe compiler crash.
Portable shader artifacts and Mesa's own shader disk cache can still speed later runs.

macOS requires Apple Silicon; CPU mode is unavailable through Metal. Windows
requires a Vulkan loader and a recent Mesa lavapipe driver with its ICD manifest;
CPU rendering there has not been qualified.
