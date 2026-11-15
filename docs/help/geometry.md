GEOMETRY
Std builders accept scene geometry shorthand; signatures and shape conventions:
vork help std --all. SDK setup: vork help sdk.

Instance columns are node-relative.

COLOR_n buffers accept linear RGB (vec3, alpha 1) or RGBA (vec4).
Use floating point colour components in 0..1, not byte values in 0..255;
glTF normalized unsigned byte/short accessors are decoded on import.
Bind authored colours explicitly with material attribute bindings;
only imported glTF materials multiply base colour by COLOR_0 automatically.
Attribute values use the same linear Rec.709 space as constant material colours;
an sRGB picker value must be converted to linear before storing it.

Std extrude supports inward rounded rims: bevel_depth (metres), bevel_resolution
(quarter-circle steps). Total depth/bounds stay fixed.
Missing NORMAL attributes use face normals; authored NORMAL values control smoothing.
Std builders emit UVs but no TANGENT attribute.
Anisotropy uses TANGENT or material geometry_tangent; without either it uses a
fallback frame, not the UV direction.
Text size is the font em in metres, not visible cap height. Finite pixel/shutter
sampling can leave small strokes and bevel highlights unresolved; extra bevel
segments and denoising do not guarantee recovery. Samples cannot add missing geometry.

Normal maps use glTF/MikkTSpace: B = cross(N,T.xyz)*T.w. Derivative frames
recover handedness from UVs and the original normal, with no image-V sign flip.
Positive UV repeats/offsets preserve authored tangents; rotated, nonpositive-scale
or alternate-UV maps use derivatives. Missing tangents also use derivatives.
Use Non-Color normal textures; do not flip green to compensate for image V.

<!-- all -->
IMPLICIT SURFACE REFERENCE
Implicit geometry uses `topology: "implicit"` and exactly one attribute naming
an f32 dense buffer (`dims: [x,y,z]`, each >= 2). The attribute name is arbitrary.
The surface is the zero of the trilinearly interpolated field; negative is inside,
and the increasing-field gradient is the outward normal. The field need not be a
distance field. Samples are cell centres in local [-1,1]³, x contiguous;
node transforms place the domain. Boundary samples must be strictly positive.
Ordinary mesh materials shade the surface, including OpenPBR transmission.
Use surface materials without subsurface shading or displacement on implicit geometry.

Std builders expand in the author loader before scene decoding. Expanded geometry uses buffers; CPU expansion usage and coverage: `vork help std --all`.

GEOMETRY STAGE ORDER
scene.geometry_stages defaults to ["deform","compute","displace"]. Each named
stage may occur once. deform includes native morphs/skin and the existing fused
subdivision path; compute runs ordinary geometry programs, including cloth;
displace runs material displacement. For displacement before a solver, use
["deform","displace","compute"]. Canonical solver inputs sample their ordered
stateless predecessor prefix at each fixed-clock tick, including replay.
This does not admit compute-produced subdivision cages or computed colliders;
use native fixed triangles or a native collider proxy as diagnostics request.
Every stage required by an authored mesh must be listed; otherwise the error
names the node and the stage to add. Unused stages may be omitted, including an
empty list for scenes without geometry effects.
