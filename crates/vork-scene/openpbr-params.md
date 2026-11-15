# OpenPBR Surface parameters (exact)

Source: OpenPBR Surface specification **v1.1.1** (2026-04-17), Academy Software Foundation.
- Table: https://github.com/AcademySoftwareFoundation/OpenPBR/blob/v1.1.1/parametrization.md.html ("Parameter reference")
- Meanings: https://github.com/AcademySoftwareFoundation/OpenPBR/blob/v1.1.1/index.html (per-section parameter tables)
- Fetched 2026-10-21 from tag `v1.1.1` (latest release). Copied, not recalled.

This table holds the 38 of the specification's 41 parameters that vork shades, with the specification's
identifiers, types, ranges and defaults. All are std/openpbr material inputs. The other three (base_diffuse_roughness,
transmission_scatter, transmission_scatter_anisotropy) are neither inputs nor Closure fields.
No aliases. The std-only evaluation input below is excluded from Closure. Range: `[]` inclusive, `()` exclusive, per channel for colours and vectors. Norm is the
typically useful range where it differs from the allowed range (a soft range, not a clamp). Lengths other than
`thin_film_thickness` are in scene world units (metres in vork). `color3` is linear in the model colour space (OpenPBR
default ACEScg; vork uses linear Rec.709, see [colour](../../docs/help/materials.md)).

The binding-restriction column records the supported table-input subset; an empty cell uses the type's ordinary bindings.

| Identifier | Type | Range | Norm | Default | Unit | Meaning (spec wording) | Binding restriction |
|---|---|---|---|---|---|---|---|
| **Base** | | | | | | |
| `base_weight` | float | [0, 1] | | 1 | | Scalar multiplier to `base_color` | |
| `base_color` | color3 | [0, 1]³ | | (0.8, 0.8, 0.8) | | Base reflection albedo (dielectric); F0 colour at normal incidence (metal) | |
| `base_metalness` | float | [0, 1] | | 0 | | Fraction of the microsurface that is metallic | |
| **Specular** | | | | | | |
| `specular_weight` | float | [0, ∞) | [0, 1] | 1 | | Modulates the dielectric reflectivity at normal incidence (via IOR); overall multiplier of the metallic Fresnel | |
| `specular_color` | color3 | [0, 1]³ | | (1, 1, 1) | | Tints the dielectric Fresnel factor; metal colour at near-grazing incidence | |
| `specular_roughness` | float | [0, 1] | | 0.3 | | Roughness of the NDF of the dielectric BSDF and conductor BRDF | |
| `specular_roughness_anisotropy` | float | [0, 1] | | 0 | | Anisotropy of that NDF (stretch along `geometry_tangent`) | |
| `specular_ior` | float | (0, ∞) | [1, 3] | 1.5 | | Refractive index of the base dielectric | |
| **Transmission** | | | | | | |
| `transmission_weight` | float | [0, 1] | | 0 | | Mix weight between translucent base and opaque base slabs | |
| `transmission_color` | color3 | [0, 1]³ | | (1, 1, 1) | | Transmission colour; sets the extinction of the interior medium | |
| `transmission_depth` | float | [0, ∞) | [0, 1] | 0 | length | Depth at which `transmission_color` is realised; 0 = constant on-surface tint | |
| **Subsurface** | | | | | | |
| `subsurface_weight` | float | [0, 1] | | 0 | | Mix weight between subsurface and diffuse slabs | |
| `subsurface_color` | color3 | [0, 1] | | (0.8, 0.8, 0.8) | | Observed reflection colour of the subsurface medium | |
| `subsurface_radius` | float | [0, ∞) | [0, 1] | 1 | length | Length scale of the mean free path | |
| `subsurface_radius_scale` | color3 | [0, 1]³ | | (1.0, 0.5, 0.25) | | RGB multiplier to `subsurface_radius` (per-channel MFP) | |
| `subsurface_scatter_anisotropy` | float | [-1, 1] | | 0 | | Henyey-Greenstein anisotropy of the subsurface medium | |
| **Coat** | | | | | | |
| `coat_weight` | float | [0, 1] | | 0 | | Coverage weight of the coat slab | |
| `coat_color` | color3 | [0, 1]³ | | (1, 1, 1) | | Square of the coat's normal-incidence transmittance | |
| `coat_roughness` | float | [0, 1] | | 0 | | Roughness of the coat NDF | |
| `coat_roughness_anisotropy` | float | [0, 1] | | 0 | | Anisotropy of NDF of coat BSDF | |
| `coat_ior` | float | (0, ∞) | [1, 3] | 1.6 | | Refractive index of the coat | |
| `coat_darkening` | float | [0, 1] | | 1 | | Modulates the physical coat darkening effect | |
| **Fuzz** | | | | | | |
| `fuzz_weight` | float | [0, 1] | | 0 | | Coverage weight of the fuzz slab | |
| `fuzz_color` | color3 | [0, 1]³ | | (1, 1, 1) | | Reflection albedo colour of the fuzz | |
| `fuzz_roughness` | float | [0, 1] | | 0.5 | | Reflection roughness of the fuzz | |
| **Emission** | | | | | | |
| `emission_luminance` | float | [0, ∞) | [0, 1000] | 0 | cd/m² (nits) | Emission luminance with `emission_color` = (1,1,1), before coat and fuzz | |
| `emission_color` | color3 | [0, ∞)³ | | (1, 1, 1) | | Emission colour multiplier (emitted = `emission_color` × `emission_luminance`) | |
| **Thin-film** | | | | | | |
| `thin_film_weight` | float | [0, 1] | | 0 | | Coverage weight of the thin film | |
| `thin_film_thickness` | float | [0, ∞) | [0, 1] | 0.5 | µm | Thickness of the film in micrometres | |
| `thin_film_ior` | float | (0, ∞) | [1, 3] | 1.4 | | Refractive index of the film | |
| **Geometry** | | | | | | |
| `geometry_opacity` | float | [0, 1] | | 1 | | Opacity (presence) of the surface | |
| `geometry_thin_walled` | boolean | {false, true} | | false | | Surface is a thin shell with no interior volume | |
| `geometry_normal` | vector3 | N/A | | unperturbed normal | | Shading normal of the base | |
| `geometry_tangent` | vector3 | N/A | | unperturbed tangent | | Reference tangent for specular anisotropy | |
| `geometry_coat_normal` | vector3 | N/A | | unperturbed normal | | Shading normal of the coat; linear tangent-space RGB; factor scales XY; independent of base normal | |
| `geometry_coat_tangent` | vector3 | N/A | | unperturbed tangent | | Reference tangent for coat anisotropy | |

Spec-text notes (copied as the reference table gives them):
- `subsurface_color` range is written `[0, 1]` (no ³) although its type is color3; read it per channel.
- Not in OpenPBR, so not vork material inputs: `emission_strength`, `sheen_*`, `clearcoat_*`, `specular_anisotropy`,
  `specular_rotation`, `subsurface_scale`, `coat_normal`, `coat_affect_*`, `geometry_thickness`,
  `double_sided`, displacement.

| **Dispersion** | | | | | | | |
| `transmission_dispersion_scale` | float | [0, 1] | | 0 | | Linearly scales the amount of dispersion | |
| `transmission_dispersion_abbe_number` | float | (0, ∞) | [9, 91] | 20 | | Physical Abbe number of the base dielectric medium | |

Std evaluation input (not an OpenPBR Closure field):

| `alpha_cutoff` | float | [-1, 1] | | -1 | | Negative disables threshold; otherwise geometry_opacity >= alpha_cutoff produces 1, else 0 | Input only |
