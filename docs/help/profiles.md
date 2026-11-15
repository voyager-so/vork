PROFILES
Choose with --profile; profile settings are separate from scene content.

Profile     Speed       Use                         Main omissions
std/previs  fastest     framing and direct light    no indirect bounce lighting
std/draft   quick       lighting previews           approximate GI, 4 samples
std/final   moderate    delivery (default)          approximate GI, 8 samples
std/path    slowest     interiors, glass, caustics  finite depth/budget, 64 samples

Preview means std/previs; use the names above with --profile.
Resolution does not change the per-pixel sample budget.

std/previs: fast iteration on framing, materials and direct lighting. It omits
indirect lighting; raster surfaces use central-emitter hard shadows. Broad-light
highlights can narrow or change brightness; ray-camera effects can remain grainy.
Judge the finished lighting with final or path.
In std/previs, scalar std/openpbr geometry_opacity and alpha_cutoff use each
material's raster coverage, retaining direct lighting and visible emission. Sampled coverage can make colour and outline-driving depth, normals and AOVs noisy at low sample counts. Bindings
or animation of either input, and opacity combined with transmission, retain
whole-view path transport until qualified. Other effects can also require path
transport. The sample budget and denoise setting stay unchanged.
Raster previs uses analytic lights and environment approximations. Emissive
meshes stay visible, but light they cast on other objects needs std/draft,
std/final or std/path; the former alpha-triggered path view included that light.
Hair uses complete-camera path transport automatically, including render-image
captures. Promoted views keep the sample budget and default to denoising;
--set denoise=false turns it off. Glass and water retain final’s raster transport.
std/draft: 4 samples with scene-ray indirect lighting and OIDN denoising.
std/final: 8 samples, the default; scene-ray lighting and OIDN denoising.
Final's approximate GI can omit bounced paths such as camera → diffuse → glossy → light.
A custom surface material selects the older one-hit GI for the whole scene.
More samples cannot recover omitted paths; choose --profile std/path for full bounce lighting.
std/path: 64 samples, independent of denoise; full path transport.
Previs, draft and final use a cheaper implicit-field boundary test; exact zero
faces, edges and tangencies can differ. Use path to judge those shapes.
Views promoted to path transport retain the exact boundary test.
Draft, final and path denoise with bundled OIDN by default. --set denoise=false uses the profile
reconstruction (raw path samples for path); --set seed=42 makes sampling repeatable with
unchanged evaluated scene inputs, frame identity, render settings and backend.
Reauthoring a pose can change its floating-point transform and resulting image.
The same machine, build and inputs produce identical pictures; different GPU models or driver versions can produce slight numerical differences that are not visible, so compare against references made on the same hardware; the operating system does not matter.
--set samples=N overrides the profile budget (1..4096); more samples means cleaner and slower.
Small text, thin geometry and subpixel bevel highlights can need extra camera samples;
see vork help geometry for the spatial sampling limits.
Without a fast GPU denoiser, draft and final announce it and use 16 and 32 undenoised samples;
--set denoise=true uses the slow CPU denoiser anyway.
See vork help limits for lighting approximations and scene-size limits.
<!-- all -->
PROFILE FILE REFERENCE
Profile files contain version (1), level and settings; fields below.
Levels: 0 = std/draft, 1 = std/final, 2 = std/path, 3 = std/previs.
Custom level-2 profiles without settings.samples use 64, independent of denoise.
Adaptive path sampling stops converged pixels early; AOVs and catchers use the full budget.
Camera optics and render.motion_blur belong to the scene.
