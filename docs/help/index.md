vork — headless 3D rendering from scene JSON.
Supported GPUs: Apple Silicon (Metal), recent NVIDIA (Vulkan), and AMD GPUs on a Vulkan driver that exposes ray queries and storage buffer arrays (tested with Mesa RADV on RDNA3).
Linux CPU rendering: opt in with --gpu cpu or VORK_GPU=cpu; see vork help limits.
Unknown fields are errors; omit anything at its default.

FIRST IMAGE
vork init my-shot
cd my-shot
vork render scene.json --out shot.png
vork help
vork init [DIR] [--language json|typescript|python] (default DIR: ., language: json).
Always writes scene.json and README.md; TypeScript adds scene.ts and package.json; Python adds scene.py.
Existing files are never overwritten. Source authoring requires the matching SDK.
Edit the source, regenerate JSON, then render.
Paths are relative to scene.json. RGB background changes the backdrop, not lighting.

Sun angular size: `vork help lights`. Tone curve and grain: `vork help render`.
Shape/revolve examples: `vork help geometry`. Movie rendering: `vork help render`.

Supported features and known limits: vork help limits. Source builds: vork help build.

RENDERING
vork render SCENE --out PATH [--at SECONDS | --frame-start N --frame-end N] [--profile P]
  [--set KEY=VALUE ...] [--channels LIST] [--exr-codec CODEC] [--png-depth 8|16] [--diagnostics]
  repeat --out PATH; the extension picks the output: .png one still (t=0 unless --at); a directory
  gets every frame as NNNNNN.png; PATH/%06d.exr frames; .mp4; .mov (keeps alpha).
  --channels color (default), depth, object_index, view_normal, motion, camera, shadow_catcher, aov:NAME;
  all includes render.aovs; applies to every --out.
  --profile: std/previs (blocking), std/draft (quick scene-ray lighting), std/final (delivery, default), std/path (accuracy).

Quality, sample budgets and denoising: vork help profiles.

<!-- generated-call-keys:start -->
--set KEY=VALUE accepts resolution, fps, seed, samples, denoise.
<!-- generated-call-keys:end -->
Values are JSON; resolution also accepts WxH or W,H.
vork check SCENE|- [--profile P]       CPU-only validation; see vork help check.
vork export SCENE --out ASSET.glb [--root NODE] [--at SECONDS]  static asset; vork help export
vork std FUNCTION JSON                 JSON argument; '-' reads stdin; JSON result
vork setup                             GPU report and init starter warmup at std/final (10 min shader budget)
vork setup --all                       Warm bundled program/profile catalog and built-in descriptors (no shader time limit)
vork setup --skip-warmup               GPU/tools report without shader warmup
vork setup --skip-warmup --verbose     Include raw encoder probe diagnostics
Other scene kinds compile once on first use.
vork help [TOPIC [--all] | field PATH]  topic details or exact field/object lookup

CONVENTIONS
Units: metres, seconds, radians; lens/sensor mm; linear Rec.709.
<!-- generated-standards:start -->
animation: follows glTF 2.0 interpolation
assets: follows glTF 2.0
cameras: follows Blender physical camera (Blender 5.2.1)
colour: follows AgX as Blender 5.2.1
geometry: follows glTF 2.0 topology and attribute semantics
lights: follows Blender 5.2.1 normalized lights
materials: follows OpenPBR Surface 1.1.1
textures: follows glTF 2.0 samplers; KHR_texture_transform lowered on import
<!-- generated-standards:end -->
Coordinates: right-handed Y-up; camera local -Z. Names exclude '/'; paths are scene-relative.
Resources accept a name or inline definition. Named tables use {name: definition}.
Node type fields are flat siblings; type: node has only common fields.
Normal maps: tangent-space +Y; TEXCOORD v=0 is the image top. Topology and program code are fixed during rendering.

INSTANCES
Mesh Node.instances takes equal-length custom attribute columns in instances.attributes;
a material reads a column with {"attribute": name}. Two instances with different colours:
{"type":"mesh","geometry":{"std":"sphere"},
 "instances":{"attributes":{"TRANSLATION":{"type":"vec3","data":[[0,0,0],[2,0,0]]},
                            "tint":{"type":"vec3","data":[[1,0,0],[0,0,1]]}}},
 "materials":[{"inputs":{"base_color":{"attribute":"tint"}}}]}

COMMON FIELDS (advanced fields: vork help TOPIC --all; exact field: vork help field PATH)
<!-- generated-api:start -->

Scene {
  $schema: string; editor schema URL
  animation: layer[] = []
  buffers: {name:buffer} = {}
  geometry: {name:geometry} = {}
  geometry_stages: (deform|compute|displace)[] = [deform,compute,displace]; ordered, each at most once; default [deform,compute,displace]
  images: {name:image} = {}
  materials: {name:material} = {}
  nodes: node[]
  programs: {name:program} = {}
  render: render = {}
  version: 2
}

Node {
  children: node[] = []
  name: string = ""; globally unique, including imported nodes
  rotation: [x,y,z,w] quaternion | {euler:[x,y,z]} (Blender XYZ) = [0,0,0,1]
  scale: [x,y,z] = [1,1,1]; any zero world-scale component hides geometry, including children
  target: node-name|[x,y,z]|null; aims -Z, world Y up; rotation is an offset on top; structural, not animatable: aim at a named node and animate its translation
  translation: [x,y,z] = [0,0,0]; parent frame
  visible: bool = true; false hides the subtree and disables its lights; animatable (step)
  visible_camera: bool = true
  type: node|mesh|asset|camera|light = "node"
}

Mesh (type: mesh) {
  geometry: geometry-name|inline geometry
  materials: (material-name|inline material)[] = []; empty = default std/openpbr with roughness 0.3
}

Asset (type: asset) {
  file: file-path
}

Camera (type: camera) {
  focus_object: node-name|null = null; camera-axis depth supplied to the program at sample time
  material: material-name|inline material|null = null; omitted = std/camera; optics are program inputs
}

Light (type: light) {
  color: RGB
  energy: number|null; sun W/m²; local W; environment multiplier
  material: material-name|inline material|null; environment program; omitted = uniform color × energy radiance
  medium: material-name|inline material|null; environment only; authored medium program for world fog
  shape: sun|point|spot|rectangle|ellipse|environment
  size: number = 1; only rectangle or ellipse lights; rectangle full side, ellipse diameter; size_y sets the second axis
}

Material {
  double_sided: bool = true
  geometry.geometry: mesh-node name; read-only geometry at the program's canonical sample time
  inputs: {name: number|vector|bool|enum-token|{image}|{attribute}|{grid}|{aov}|{node}|{channel}|{geometry}} = {}
  opacity: coverage|transmittance|null = null; null = coverage, transmittance for transmitting surfaces; coverage = occupied area fraction; transmittance = continuous attenuation by 1 - geometry_opacity; renderer chooses integration
  priority: integer = 0; highest owns overlapping solid media; equal priorities: last entered wins, simultaneous entries: later solid order wins; lower-priority boundaries change membership only
  program: program-name|inline program|<library>/<module> = std/openpbr
}
std/openpbr inputs: base_weight base_color base_metalness specular_weight specular_color specular_roughness specular_roughness_anisotropy specular_ior transmission_weight transmission_color transmission_depth subsurface_weight subsurface_color subsurface_radius subsurface_radius_scale subsurface_scatter_anisotropy coat_weight coat_color coat_roughness coat_ior coat_darkening fuzz_weight fuzz_color fuzz_roughness emission_luminance emission_color thin_film_weight thin_film_thickness thin_film_ior geometry_opacity geometry_thin_walled geometry_normal geometry_tangent geometry_coat_normal transmission_dispersion_scale transmission_dispersion_abbe_number alpha_cutoff

Texture binding {
  image: image-name|inline image; the program owns sampling and UV transforms
}

Image {
  color_space: sRGB|Linear Rec.709|Non-Color|null = null; null: hdr/exr/ies linear, others sRGB
  file: file-path; XOR render; png jpg tif hdr exr ies mov mp4 mkv or %04d frame pattern; 8-bit may be block-compressed; 16-bit integer and floating-point inputs use uncompressed half-float precision, not exact source integer codes; ies: LM-63 Type C
}

Render {
  background: environment|transparent|RGB = environment; RGB and transparent apply to camera escapes and opacity passthrough; path glass transmission sees the World
  camera: camera-node-name|null = null; null = the only authored camera, else first imported; a camera cut track may select
  compositor: (material-name|{material,name?,resolution?})[] = []; ordered image graph, last output; material may be inline
  display: material-name|inline material = {"program":"std/agx"}; display program; EXR stays scene-linear
  duration: number|rational string = 125/12; seconds
  fps: number|rational string = 24; positive output frames/second
  motion_blur: MotionBlur|null = null; non-null enables; shutter in frames
  motion_blur.shutter: number = 0.5; frames
  resolution: uvec2 = [1920,1080]
}
<!-- generated-api:end -->
Asset nodes also take materials and nodes overrides (vork help nodes).

LIBRARIES
Std builders return {buffers,geometry}; name sets standalone CLI/SDK fragment prefixes.
Shapes are centred; dimensions are metres.
Text, bevelled extrusion and rounded boxes: vork help std --all.
<!-- generated-shapes:start -->
Scene geometry: {std: NAME, ...options}; omit name: the table key or inline position owns identity.

<!-- generated-shapes:end -->
Compositor: ordered {material,name?,resolution?} image graph; material may be inline.
Image inputs bind earlier results with {node:name}; last entry is output. See vork help compositor.
Environment: light.material evaluates radiance; std/environment_image reads ordinary Image inputs.
vork std environment preprocesses HDR lighting; vork std sky declares atmosphere and an ordinary sun.

Topics (vork help TOPIC [--all]):
check: validate without rendering
upgrade: breaking changes
contract: validation and installation
nodes: assets, skins, morphs, object IDs
cameras: optics and focus
lights: lighting and sky
materials: surface inputs and media ownership
fog: medium programs and admission
geometry: mesh attributes and implicit fields
buffers: binary and computed data
images: textures and render-to-texture
animation: tracks, clips, layers
render: timing and render fields
profiles: quality, samples, denoising
compositor: image graph and sampling
light-paths: rays, sensors, spectral transport
programs: Slang stages and bindings
outputs: channels, metadata, codecs
camera-json: OpenCV sidecars
sdk: TypeScript and Python
std: builders and Slang modules
fluid: liquid simulation
cloth: XPBD fabric, native animated collider references and private geometry state
errors: exit codes
diagnostics: timings and environment variables
limits: hardware and rendering limits
build: source build prerequisites

Scene JSON supports `"$schema": "https://raw.githubusercontent.com/voyager-so/vork/v0.5.1/packages/ts/scene.schema.json"` for editor
completion and validation. `vork init` includes it; SDK writers preserve an explicit
$schema but do not insert one. vork never fetches it or changes the render.

UPDATE NOTICES
Vork checks npm for a newer stable version at most once a day in a detached background worker
after command completion, caches the result in your user cache directory, and shows a one-line
notice on stderr on a later run, including when stderr is redirected for agents. Set
VORK_NO_UPDATE_NOTIFIER (any value, including empty) to disable both checks and notices; CI also
disables them. The check sends only a package-version request to the npm registry, with no
identifiers or telemetry. Network and cache failures are silent and never change rendered files
or exit status.
