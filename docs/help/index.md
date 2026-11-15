vork — 2D and 3D rendering from scene JSON.
Supported GPUs: Apple Silicon (Metal), recent NVIDIA (Vulkan), and AMD GPUs on a Vulkan driver that exposes ray queries and storage buffer arrays (tested with Mesa RADV on RDNA3).
Linux CPU rendering: opt in with --gpu cpu or VORK_GPU=cpu; see vork help limits.
Unknown fields are errors; omit anything at its default.
Colour: composition.color_space is working RGB; composition.view is local (worlds std/agx, canvases std/standard).
Choose each destination with output.display; see vork help outputs for supported pairs.

FIRST IMAGE
vork init my-shot
cd my-shot
vork render scene.json --out shot.png
vork help
Scaffold languages, files and conflict policy: vork help init. SDK setup: vork help sdk.
Edit SDK source, regenerate JSON, then render.
Paths are relative to scene.json. RGB background changes the backdrop, not lighting.

Sun angular size: `vork help lights`. Tone curve and grain: `vork help render`.
Shape builders: `vork help geometry`. Movie rendering: `vork help render`.

Supported features and known limits: vork help limits. Source builds: vork help build.
Authoring skill: vork skills add

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
vork ask SCENE QUESTIONS|-  Read scene values without rendering; see vork help ask.
vork export SCENE --out ASSET.glb [--root NODE] [--at SECONDS]  static asset; vork help export
vork std FUNCTION JSON                 JSON argument; '-' reads stdin; JSON result
vork setup                             GPU report; builds the path engine, warms the init starter at std/final (10 min shader budget)
vork setup --all                       Warm bundled program/profile catalog and built-in descriptors (no shader time limit)
vork setup --skip-warmup               GPU/tools report without shader warmup
vork setup --skip-warmup --verbose     Include raw encoder probe diagnostics
Without setup, a cold path render compiles its scene kernel before the picture, then exits
(about 50–60 s for path-glass and path-many-lights at std/path on an NVIDIA L4); vork help limits.
Other scene kinds also compile once on first use.
vork help [TOPIC [--all] | field PATH]  topic details or exact field/object lookup
Animation recipes: vork help springs, vork help sequences, vork help captions.
2D canvas (text, paths, CSS paint): vork help nodes, then vork help composition, vork help text, vork help path.

CONVENTIONS
Units: metres, seconds, radians; lens/sensor mm; linear Rec.709.
<!-- generated-standards:start -->
animation: follows glTF 2.0 interpolation
assets: follows glTF 2.0
cameras: follows Blender 5.2.1 physical camera, except the names fit, shift, projection
colour: follows AgX as Blender 5.2.1
geometry: follows glTF 2.0 topology and attribute semantics
lights: follows Blender 5.2.1 lights, except radius (Blender: shadow_soft_size)
materials: follows OpenPBR Surface 1.1.1
textures: follows glTF 2.0 samplers; KHR_texture_transform lowered on import
<!-- generated-standards:end -->
Coordinates: right-handed Y-up; camera local -Z. Names exclude '/'; paths are scene-relative.
Resources accept a name or inline definition. Named tables use {name: definition}.
Node type fields are flat siblings; type: node has only common fields.
A field path names one object, not its table: node.font-size, material.inputs, render.fps; the table is scene.nodes.
Normal maps: tangent-space +Y; TEXCOORD v=0 is the image top. Topology and program code are fixed during rendering.

COMMON FIELDS (advanced fields: vork help TOPIC --all; exact field: vork help field PATH)
<!-- generated-api:start -->

Scene {
  $schema: string; editor schema URL
  animation: layer[] = []
  buffers: {name:buffer} = {}
  composition: composition|null; space 2d = canvas, 3d = world; null = 3D world
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
  active: true|false|[start,end); parent seconds; false removes picture and sound, children included; a window starts the clock at start; a boolean is keyable (step)
  animation: layer[]; tracks on this node's clock in content seconds; paths start here; a property has one owner; the clock cannot key itself
  audio: {file,gain}|null; file relative to scene; gain defaults to 1 and scales source plus children; active mutes, visible does not; node time controls playback and pitch
  audio.file: file-path|null; mono/stereo; relative to scene; omitted = children only
  audio.gain: number = 1; linear amplitude over source plus children; default 1; keyable
  children: node[] = []
  clip-path: node-name|null; 2D; that node's shape, in this node's coordinates, clips this node; it is not painted itself
  composition: composition|null; 2d: a canvas, its children painted to one picture; in a 3D world seen through an image's render.root; 3d: in a canvas, a closed world drawn through its one camera to one picture the size of the canvas's box
  filter: string|null; 2D; blur() and drop-shadow(); nonzero lengths require px; blur sigma above 100 target pixels after transforms refuses
  mask: node-name|null; 2D; that node's picture, in this node's coordinates, masks this node by luminance; it is not painted itself
  name: string = ""; globally unique, including imported nodes
  opacity: number = 1; 2D; 0..1; the node's picture, children included, placed once
  rotation: [x,y,z,w] quaternion | {euler:[x,y,z]} (Blender XYZ) = [0,0,0,1]
  scale: [x,y,z] = [1,1,1]; any zero world-scale component hides geometry, including children
  target: node-name|[x,y,z]|null; aims -Z, world Y up; rotation is an offset on top; structural, not animatable: aim at a named node and animate its translation
  time: {start,offset,duration,speed,loop}|null; node-owned animation/media clock; time-reading programs, conflicting movie clocks and scene-time tracks inside retimed compositions refuse; null inherits parent/active-window clock
  time.duration: number|rational string|null; content seconds used from offset; > 0; null = the rest of the one source; only active ends the node
  time.loop: bool = false; wraps inside [offset, offset+duration); needs an end
  time.offset: number|rational string = 0; content seconds where the clip begins; >= 0
  time.pitch: follow|keep = follow; keep holds sound pitch when speed changes; keep is not built yet
  time.speed: number|rational string = 1; keyable >= 0 and integrated; a constant < 0 plays the trimmed section backwards from its end
  time.start: number|rational string|null; parent seconds where local = offset; null = active window start, else 0; shifts loop phase
  translation: [x,y,z] = [0,0,0]; parent frame
  visible: bool = true; false hides the subtree and disables its lights; animatable (step)
  visible_camera: bool = true
  type: node|mesh|asset|camera|light|text|path = "node"
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

Text (type: text, canvas) {
  content: string|run[]; position by the text box's top-left; switch content with STEP keys; runs set font and paint; white-space controls newlines
  fill: "none"|CSS colour|CSS gradient|{image, fit} = #000
  font: file-path|file-path[]; .ttf/.otf; a list offers faces font-weight and font-style choose from
  width: number|null; px; lines wrap and align in it; null = no wrapping
}

Path (type: path, canvas) {
  d: string = none; SVG path data, px; "none" or empty paints nothing
  fill: paint = #000; "none"|CSS colour|CSS linear-/radial-/conic-gradient()|{image, fit}
}

Material {
  double_sided: bool = true
  geometry.geometry: mesh-node name; read-only geometry at the program's canonical sample time
  inputs: {name: number|vector|bool|enum-token|{image}|{attribute}|{grid}|{aov}|{step}|{channel}|{geometry}} = {}
  opacity: coverage|transmittance|null = null; null = coverage, transmittance for transmitting surfaces; coverage = occupied area fraction; transmittance = continuous attenuation by 1 - geometry_opacity; renderer chooses integration
  priority: integer = 0; highest owns overlapping solid media; equal priorities: last entered wins, simultaneous entries: later solid order wins; lower-priority boundaries change membership only
  program: program-name|inline program|<library>/<module> = std/openpbr
}
std/openpbr inputs: base_weight base_color base_metalness specular_weight specular_color specular_roughness specular_roughness_anisotropy specular_ior transmission_weight transmission_color transmission_depth subsurface_weight subsurface_color subsurface_radius subsurface_radius_scale subsurface_scatter_anisotropy coat_weight coat_color coat_roughness coat_roughness_anisotropy coat_ior coat_darkening fuzz_weight fuzz_color fuzz_roughness emission_luminance emission_color thin_film_weight thin_film_thickness thin_film_ior geometry_opacity geometry_thin_walled geometry_normal geometry_tangent geometry_coat_normal geometry_coat_tangent transmission_dispersion_scale transmission_dispersion_abbe_number alpha_cutoff

Texture binding {
  image: image-name|inline image; the program owns sampling and UV transforms
}

Image {
  color_space: sRGB|Display P3|Linear Rec.709|Linear Rec.2020|ACEScg|Non-Color|null = null; omitted/null: embedded colour description, else hdr/exr/ies linear Rec.709 and other files sRGB
  file: file-path; XOR render; png jpg tif hdr exr ies mov mp4 mkv or %04d frame pattern; 8-bit may be block-compressed; 16-bit integer and floating-point inputs use uncompressed half-float precision, not exact source integer codes; ies: LM-63 Type C
}

Render {
  background: environment|transparent|RGB|CSS colour = environment; RGB and transparent apply to camera escapes and opacity passthrough; path glass transmission sees the World; a 2D canvas takes transparent or a CSS colour
  camera: camera-node-name|null = null; null = the only authored camera, else first imported; a camera cut track may select
  duration: number|rational string = 125/12; seconds
  fps: number|rational string = 24; positive output frames/second
  motion_blur: MotionBlur|null = null; non-null enables; shutter in frames
  motion_blur.shutter: number = 0.5; frames
  resolution: uvec2 = [1920,1080]
}
<!-- generated-api:end -->
Asset nodes also take materials and nodes overrides (vork help nodes).
SPZ and Gaussian PLY: import through asset; replace the radiance material through materials.

LIBRARIES
Std builders return {buffers,geometry}; name sets standalone CLI/SDK fragment prefixes.
Shapes are centred; dimensions are metres.
Text, bevelled extrusion and rounded boxes: vork help std --all.
<!-- generated-shapes:start -->
Scene geometry: {std: NAME, ...options}; omit name: the table key or inline position owns identity.

<!-- generated-shapes:end -->
Compositor: composition.compositor, an ordered {material,name?,resolution?} image graph; material may be inline.
Image inputs bind earlier results with {step:name}; last entry is output; see vork help compositor.
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

Scene JSON supports `"$schema": "https://raw.githubusercontent.com/voyager-so/vork/v0.6.0/packages/ts/scene.schema.json"` for editor
completion and validation. `vork init` includes it; SDK writers preserve an explicit
$schema but do not insert one. vork never fetches it or changes the render.

UPDATE NOTICES
Vork checks npm for stable updates at most daily after command completion and caches
results locally. Notices appear on stderr on later runs, including redirected logs.
VORK_NO_UPDATE_NOTIFIER (any value) or CI disables checks and notices.
Only a package-version request is sent; no identifiers or telemetry. Network/cache
failures are silent and never affect outputs or exit status.
