MATERIALS
Use std/openpbr (the default) for constant, image and named attribute inputs.
Instance attributes are equal-length columns in node.instances.attributes;
material attribute bindings select a column by name.
Constant, image and attribute bindings share the shipped evaluator. Custom logic
needs a surface program; constant OpenPBR parameters do not.
emission_luminance is cd/m²; emission_color is linear RGB.
base_color is linear RGB; base_metalness=1
makes metal; specular_roughness controls highlight width. Inputs and defaults:
vork help field material.inputs
Image inputs bind {"image": name}; std/openpbr uses UV0, red for scalar inputs
and RGB for colour. Custom factors, channels and UV transforms need a program.
For glass panes use transmission_weight: 1 and geometry_thin_walled: true.
Use transmission_color for glass tint and subsurface_color for scattering colour.
Use std/path when refraction or indirect lighting is important.
Profile coverage, emissive lighting and transport selection: vork help profiles.
<!-- all -->
MATERIAL REFERENCE
Solid glass requires closed mesh boundaries; model hollow vessels with a shell.
For overlapping glass and liquid, assign material.priority (see vork help field
material.priority). The owning medium supplies both IOR and absorption; hidden
lower-priority boundaries pass straight through. Equal priorities preserve nested
replacement: the last entered solid wins. Touching faces cross directly between
the two media, without an air interface. Thin sheets have no interior.
With transmission_depth=0, a closed solid shows the authored transmission_color;
each thin sheet applies the full tint. Use thin-walled panes for windows.
Thin-walled subsurface suits leaves, paper and curtains: geometry_thin_walled=true,
subsurface_weight>0. Set subsurface_color and subsurface_scatter_anisotropy;
positive anisotropy favours transmission. Radius controls apply to thick objects.
On thick meshes, transmission_weight mixes glass with the subsurface substrate.
Thick subsurface transport freezes homogeneous coefficients at entry and uses an
index-matched diffuse boundary; zero mean-free-path channels retain local response.
geometry_normal and geometry_coat_normal accept tangent-space XYZ vectors or
linear RGB textures. An omitted coat normal stays unperturbed by the base normal map.
Normal-map frames follow glTF/MikkTSpace (vork help geometry).
For transformed normal maps, custom programs use std.texture.
Raster ambient/GI reserves the same dielectric diffuse response as direct/path
shading, including mixed metals and thin subsurface. Raster transport remains an
approximation; use std/path to assess indirect lighting.
Use constant, texture or named attribute bindings for geometry_opacity and
alpha_cutoff; camera and ray hits share evaluation, preserving custom surface
programs and their material animation. Computed instance material
attributes follow GPU publications, including shutter samples and seeks.
Numeric inputs animate with material tracks. Weighted/additive program tracks
start from the authored input, or zero when omitted. Untracked inputs retain
program initializers. Custom shading: vork help programs --all.
