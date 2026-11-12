LIMITS
Supported GPUs: Apple Silicon Metal, recent NVIDIA Vulkan and AMD Vulkan with
hardware ray tracing and storage buffer arrays (tested on Mesa RADV, RDNA3).
Intel GPUs are unsupported.
CPU rendering requires the explicit opt-in below. Run vork setup to check the
adapter. Windows x64 ships install.ps1;
macOS and Linux provide install.sh as a separate release asset.
The bundled Slang toolchain targets macOS 26; older macOS is unqualified.
See README.md (Install) for binary setup.

Device-limit errors report the requested size or count, the device limit, and a
resource-specific remedy. For image and camera allocations, lower resolution
with --set resolution=WxH; lower sample-dependent demand with --set samples=N
or --profile std/draft. Reflection allocations can also shrink with a lower
profile budget; std/path avoids the screen reflection record allocation when
suggested by the error. Geometry and compute allocations instead need smaller
meshes, buffer pools or fewer inputs. Denoising allocations can be avoided with
--set denoise=false. Follow the remedy named by the failing resource.

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
A non-finite path contribution (an infinite light sample) is dropped as zero rather
than refusing the frame. Each affected frame warns with its dropped count;
`dropped_light_samples` records the count in frame metadata, even without `--diagnostics`.
Non-finite camera path samples are also zeroed and reported as `dropped_path_samples`.
Drops above 0.1% of the authored camera-sample budget refuse delivery; this
protects against systematic broken transport. Very small frames may have no allowance.
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

Large point scenes and working buffers can exceed a device's per-buffer binding
limit even when total free VRAM is sufficient. Follow the error's resolution/profile
advice or simplify scene buffers. Scene import support does not guarantee rendering
at any size.

vork setup warms only the vork init starter at the default std/final profile. Only
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

Named opacity and alpha-cutoff attributes use the shared camera and ray evaluators.
Computed instance material attributes follow GPU publications, including shutter
samples and seeks. Installation refuses these wholly omitted geometry effects:

- displacement reading computed instance attributes: use static/authored values or
  pre-baked displaced geometry;
- displacement on computed points or implicit geometry: compute displaced positions in the
  producer, evolve the scalar field, or bake triangle geometry;
- normal-dependent point displacement: use an authored direction attribute or triangles;
- conflicting displacement materials or inconsistent attributes at shared vertices:
  use one displacement material and continuous attributes, or split the geometry;
- computed geometry with coverage/transparency, or computed placement of transparent
  instances: use opaque geometry, bake geometry updates, or author placements.

These errors name the node and available alternative. Approximate effects still warn
and render.
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

Fog watchdog termination reports a count and discarded accumulated radiance; the untraced energy remains unknown. Ordinary paths terminate by sampled transport and compensated roulette.
Custom surface shaders with fog are not yet supported; use builtin surfaces.
