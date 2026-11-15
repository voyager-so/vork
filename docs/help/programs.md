PROGRAMS
For standard geometry use std box, sphere, grid, cylinder, cone, torus, capsule,
revolve and sweep builders; options and units: vork help std --all.
Use std/openpbr data inputs first (vork help materials). Data materials share
the shipped evaluator; custom surfaces can add code to the scene kernel and
increase first-render compilation time. Use programs for custom logic.
Custom Slang programs provide surface shading,
displacement, media, environments, cameras, sensors, compositing and compute.
Medium coefficient units, bounds and admission: vork help fog.
Use vork help programs --all for the advanced reference and stage declarations.
<!-- all -->
PROGRAM REFERENCE
Imported Blender node semantics: vork help blender --all.
Slang modules: `import vork;`, one `struct Inputs {...}` whose initializers are
defaults; surface meshes require surface (lit/displace supplement surface);
medium-only carriers require medium instead (vork help fog):
  Closure|Blend|Hair surface(ShadingContext, Inputs)
    OpenPBR emission, including unlit, is shared by hits and light samples and added to the shaded lit result.
    Light samples use zero differentials; hits retain their shading footprint.
  float3 displace(DisplacementContext, Inputs)       object-local offset after deformation/subdivision; fixed topology
    Use displacement on ordinary meshes with consistent assignments and attributes at shared vertices.
    Hair uses curve geometry and a surface stage only.
  void aov(ShadingContext, Inputs, inout AovOutputs) write_aov(output, inputs.a, value)
  Closure lit(ShadingContext, Inputs, Lighting)      shader-to-RGB: return unlit(rgb)
  MediumCoefficients medium(MediumContext, Inputs)  physical point coefficients; vork help fog
  float3 environment(EnvironmentContext, Inputs)    radiance
  float4 composite(CompositeContext, Inputs)         vork help compositor
  CameraRay camera(CameraContext, Inputs)           camera.material; vork help light-paths
  float3 sensor(SensorContext, Inputs)               per arriving contribution
  ViewResult view(ViewContext, Inputs)               view_from_xyz(XYZ D65 cd/m²) or view_from_codes(codes, ctx.display)
  void compute(ComputeContext, Inputs)               buffers: <binding>_load/_store/_count; dense grids also <binding>_dims (uint3)
Initialized read-only global tables (including static const arrays) are allowed.
Global resources and mutable globals are refused; use Inputs or compute buffers.
Compiler names beginning vork_ or __ and ## token pasting are unavailable to authored
programs. Supplied helper modules vork_sampling and vork_openpbr_constants remain importable.
Inputs bind numbers, bools and enum tokens directly; Image {"image"}, Aov {"aov"},
FloatGrid/VectorGrid {"grid"}, Attribute<float..float4> {"attribute"}.
[InputRange(min, max, min_closed, max_closed)] on numeric Inputs fields validates
bindings componentwise; booleans select inclusive endpoints. It does not clamp values.
Closure.geometry_normal is a world-space normal.
Closure c = {}; is OpenPBR defaults (fields are OpenPBR input names). blend(a,b,w) weights a by
1-w and b by w; add(a,b) keeps both; mix_closure(a,b,w) interpolates parameters (one lobe).
read_attribute(inputs.a, ctx) returns AttributeSample<T> (.value/.dx/.dy) for
ShadingContext and T directly for DisplacementContext; names and types must match
every bound geometry or instance column.
ShadingContext: time (s), instance_index (authored row), primitive_id, front_facing,
tangent_sign; wavelength (nm; 0 in RGB) is available in surface.
ShadingContext.POSITION is world-space metres; NORMAL, true_normal and TANGENT are world-space vectors, and incoming is the unit direction toward the preceding ray vertex. world_to_object converts a world position to
the shaded object's local coordinates.
world_normal_to_object normalizes after transforming to object space.
Use object_direction_to_world for local vectors and object_normal_to_world for local surface normals; both return unit world directions.
DisplacementContext has only undisplaced object-local POSITION, NORMAL and time (s).
Per-instance displacement values use bound Attribute<T> inputs.
For topology:"points", displace evaluates once per undisplaced object-local point centre and translates the whole sphere without changing its width; NORMAL and generated surface UV/tangent inputs are unavailable.

GEOMETRIC QUERIES (surface and compute)
TraceHit trace(float3 origin, float3 direction, float min_distance,
               float max_distance, uint mask=255u, uint object=0xffffffffu)
Closest geometric triangle boundary in world space; direction is normalized.
Finite distances require 0 <= min_distance <= max_distance. Invalid inputs miss.
Use ctx.POSITION and an explicit minimum distance to avoid self-intersection.
The query ignores opacity and materials, including normal maps and backface culling.
Hit fields: hit, distance, position, geometric_normal, shading_normal, object_id,
instance_index, primitive_id, barycentrics (three weights), uv (UV0), front_facing.
Object selection is exact public identity; mask selects the low eight AS visibility bits.
Miss: distance=-1, identities=0xffffffffu, other fields zero.
Reached calls count as work even when hit data is ignored; dead branches do not.
Use fixed bounded loops. Work exhaustion refuses the exposure; it never publishes
partial output. Compute queries require immutable triangle targets: select a static
collider's pass_index with the object argument.
Animated or computed target geometry/placement is refused, including self-dependencies.
Compute queries retain that installed snapshot through ticks and replay. Implicit
geometry queries are not yet qualified. Raster installs ray resources when queries
need them. Real-time queries require
a separately qualified profile and frame budget.

SIMULATION
Use scene.simulation with t0, positive dt, seed, state buffer names, and ordered init/steps
program names. Each named program uses schedule:"simulation" and ordinary bindings/dispatch.
S0 is initialized at t0; before t0 hold S0. Render subframes hold the preceding fixed tick.
Tick inputs sample the left endpoint using ordinary numeric animation tracks. Passes have
GPU barriers; use separate generations for neighbour reads/writes. u32 bindings expose
_atomic_add/sub/min/max/and/or/xor/exchange(index,value) and
_atomic_compare_exchange(index,expected,desired). f32 bindings expose
_atomic_add(index,value). All return the previous value. Out-of-bounds does nothing
and returns zero (UINT_MAX for u32 min).
Seeking replays from initialization or a compatible bounded shared-cache checkpoint.
Floating atomic accumulation replays within float noise on the same backend.
Integer commutative reductions replay exactly with stable operands; exchange/CAS
and use of returned prior values depend on invocation order. Keep other accesses race-free. Use computed
placement for opaque, unskinned geometry. Computed instance material attributes
follow GPU updates; displacement cannot read them. Author transparent placements.


GEOMETRY PROGRAM INPUTS
Geometry programs automatically receive read-only predecessor attributes. Built-in channels
use geometry_POSITION_load(index), geometry_NORMAL_load(index), and the other
present glTF semantics. All authored channels also have typed
geometry_attribute_N_load(index), numbered in attribute-name order, starting at zero;
this includes custom names that are not Slang identifiers. Each accessor has a _count.
geometry_vertex_capacity is the installed capacity; geometry_vertex_count()
returns predecessor membership. Authored point widths and IDs use
geometry_width_load(index) and geometry_id_load(index).
Native triangle morphs and skinning evaluate before these reads: simulation
initialization samples t0, and each tick samples its left endpoint, including replay.
Each object's input preserves authored vertex rows, including unreferenced rows.

Geometry inputs bind {geometry: mesh-node-name}. For an input named body, geometry_POSITION_load(inputs.body,i)
and geometry_NORMAL_load(inputs.body,i) read its local vertices;
geometry_to_world(inputs.body,p) applies its sampled placement.
geometry_vertex_count(inputs.body), geometry_triangle_count(inputs.body), and
geometry_triangle_load(inputs.body,triangle) expose fixed triangle connectivity.
geometry_source_vertex_load(inputs.body,i) retains authored vertex identity;
geometry_object_id(inputs.body) retains the installed object identity.
Colliders may move, morph, skin or displace: inputs sample at initialization and
each canonical tick, including replay. References are read-only and cannot form
cycles. Use one separately named native triangle object per collider instance;
imported, subdivided and compute-produced collider streams report their supported
proxy alternative. Static trace() queries retain their separate snapshot contract.

Every compute binding also has previous_<binding>_load(index) and _count, with its
ordinary element type. These read the state before the whole scheduled tick, including
when several ordered simulation steps write that binding. Initialization reads the initial
buffer bytes; the first simulation step reads initialized S0. Frame schedules follow their
existing nominal frame clock. These inputs expose no stores or atomics and need no new
binding fields. Reads do not publish writes or become checkpoint state.

EnvironmentContext: normalized direction, time (seconds), camera (camera escape versus lighting).
Environment programs depend on direction and may sample file images. Camera rays
evaluate the program at every sample; lighting reads a baked lat-long copy.
Displacement may sample static file images. For points use POSITION, time and bound
attributes. Fixed computed triangles support displacement, including historical
vertex attributes; computed points and implicit fields need producer offsets or
pre-baked triangle geometry.

`scene.propagation={program,bounds}` selects `PropagationStep propagation(PropagationContext,Inputs)`; bounds are world-space `[min,max]` metres.
Context: `position` (metres), unit `direction`, `time` (seconds), `previous_distance` (metres; zero at launch). Step: `direction` (normalized), positive `distance` (metres), `absorbed=false`, `radiance=float3(1)` (multiplied at segment start).
Use std/path for authored propagation; use sampled shutter blur rather than motion channels.
Segments use ordinary ray transport; leaving bounds continues straight and samples the sky along the exit direction.

Programs own image channel selection, factors and UV transforms.
`std.texture` provides footprint filtering and tangent-frame normal mapping; `ShadingContext.sample_spread` retains sample-centre covariance, while UV derivatives remain geometric.
