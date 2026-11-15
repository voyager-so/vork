//! The forward variant key and its generated entry module. build.rs includes
//! this file too, so the precompiled generic and runtime variants come from one
//! definition (docs/design/shaders.md: a variant is one `IVariant` type).

/// CPU-only installed capabilities. Shaders receive folded bools, not this mask.
/// Keep independently authored response and transport flags beyond the first word.
pub type FeatureMask = u128;
/// Coverage samples of multisampled raster passes (constants::RASTER_MSAA_SAMPLES).
pub const RASTER_MSAA_SAMPLES: u32 = 4;
pub const SHADING_LAYERS: u8 = 1;
pub const SHADING_METAL: u8 = 2;
// Install-time capability key: layer/metal bits, one texture-presence bit per input,
// then independent coat/fuzz, F82 tint, coverage and scene light-mix bits.
pub const FEATURE_COAT: FeatureMask = 1 << 44;
pub const FEATURE_FUZZ: FeatureMask = 1 << 45;
pub const FEATURE_TINT: FeatureMask = 1 << 46;
pub const FEATURE_MASK: FeatureMask = 1 << 47;
pub const FEATURE_PUNCTUAL: FeatureMask = 1 << 49;
pub const FEATURE_FINITE: FeatureMask = 1 << 50;
/// Installed finite-emitter geometry. Radius/angle animation may enable an ellipse;
/// the authored area shape itself never changes after install.
pub const FEATURE_RECTANGLE: FeatureMask = 1 << 57;
pub const FEATURE_ELLIPSE: FeatureMask = 1 << 58;
/// Installed shadow-ray consumers, independent of energy and other TLAS uses
/// such as refraction or thickness. Generic variants conservatively retain rays.
pub const FEATURE_SHADOW_RAYS: FeatureMask = 1 << 60;
/// A Blend program: the draw shades each lobe of a mixed surface.
pub const FEATURE_LOBES: FeatureMask = 1 << 55;
/// A program with a lit stage: the draw shades the closure it returns.
pub const FEATURE_LIT: FeatureMask = 1 << 56;
/// Installed transmission and possible solid boundaries, conservative for programs/tracks.
pub const FEATURE_TRANSMISSION: FeatureMask = 1 << 61;
pub const FEATURE_SOLID_TRANSMISSION: FeatureMask = 1 << 62;
pub const FEATURE_ANISOTROPY: FeatureMask = 1 << 64;
/// Coat quadrature is independent of the base lobe's anisotropy.
pub const FEATURE_COAT_ANISOTROPY: FeatureMask = 1 << 72;
pub const FEATURE_FILM: FeatureMask = 1 << 65;
/// A closure may modulate dielectric IOR away from its authored value.
pub const FEATURE_WEIGHTED_IOR: FeatureMask = 1 << 73;
/// Imported glTF boundary tint and vertex base colour.
pub const FEATURE_GLTF_TINT: FeatureMask = 1 << 74;
/// Camera results consumed by installed screen effects; shared by all material bins.
pub const FEATURE_CAMERA_RECEIVERS: FeatureMask = 1 << 66;
pub const FEATURE_REFLECTIONS: FeatureMask = 1 << 67;
pub const FEATURE_INDIRECT_CONTROLS: FeatureMask = 1 << 68;
/// Only the combined SSS/screen-refraction composition kernel captures a second receiver.
pub const FEATURE_RECEIVER_DIFFUSION: FeatureMask = 1 << 69;
pub const FEATURE_SPECIALIZED: FeatureMask = 1 << 63;
/// Stored evaluation records' optional words. `camera_config.words` sizes each record
/// from the scene's mask, so a kernel that writes evaluations must agree on these exactly:
/// one that wrote more would overwrite the next record's words.
pub const FEATURE_RECORD_WORDS: FeatureMask = FEATURE_CAMERA_RECEIVERS | FEATURE_REFLECTIONS | FEATURE_INDIRECT_CONTROLS;
/// An unspecialized key takes `FEATURE_RECORD_WORDS` from its mask instead of enabling them.
pub const FEATURE_EXACT_RECORD: FeatureMask = 1 << 70;
/// Installed implicit geometry; path-only scene union, never a per-material key.
pub const FEATURE_IMPLICIT_GEOMETRY: FeatureMask = 1 << 71;
/// Some installed material can pass a straight path connection through a
/// covered point (materials.rs `null_connection`); path-only scene union,
/// never a per-material key.
pub const FEATURE_NULL_CONNECTION: FeatureMask = 1 << 75;
/// Installed proof that no camera sample starts inside a solid
/// (passes::path::camera_exterior). Unlike the capability bits, absence is the
/// safe value: only the path key of an install that holds the proof sets it,
/// and a union of keys must not carry it unless every member does.
pub const FEATURE_CAMERA_EXTERIOR: FeatureMask = 1 << 78;
/// Some installed closure can have a subsurface lobe (a nonzero, bound or tracked
/// `subsurface_weight`, or a program whose proofs leave it possible), and some
/// installed material is a fibre program. Path-only scene unions like the field
/// above, never per-material keys: raster keys carry neither.
pub const FEATURE_SUBSURFACE: FeatureMask = 1 << 76;
pub const FEATURE_HAIR: FeatureMask = 1 << 77;
/// The key is the union of every installed material (materials.rs
/// `installed_features`), so a capability it lacks is absent from the whole scene.
/// Only such a key may state the scene-wide proofs (surface.slang
/// `installed_surface`): a raster or hit key without this bit never carries the
/// unions above, and reading them as proofs would drop a lobe.
pub const FEATURE_INSTALLED: FeatureMask = 1 << 79;
/// Hashed (dithered) camera coverage: the entry declares SV_PrimitiveID.
/// Declaring that input costs every draw of the pipeline, measured +8% of a
/// motion-blurred car's controls pass on L4, so table entries declare it only
/// for dithered materials (shipped generics come with and without it).
/// Pipeline roles add it (never Material::features); it selects only the entry
/// signature, not the `Key : IVariant` capabilities.
pub const FEATURE_DITHERED: FeatureMask = 1 << 69;
/// Four emitter kinds, matching Light.position.w (sun, point, spot, area).
pub const FEATURE_LIGHT_SHIFT: u32 = 51;
pub const FEATURE_LIGHT_TYPES: FeatureMask = 15 << FEATURE_LIGHT_SHIFT;
/// The light kinds a scene-union key carries. Transport folds on the kind only
/// when a scene installs exactly one (vork_prelude.slang `specialized_light`):
/// no lights, or several kinds, compile the same kernel as every kind. Those
/// masks are therefore written as every kind, so equal kernels have one key
/// and one cache identity.
pub const fn folded_light_kinds(installed: FeatureMask) -> FeatureMask {
    let kinds = installed & FEATURE_LIGHT_TYPES;
    if kinds.count_ones() == 1 { kinds } else { FEATURE_LIGHT_TYPES }
}
/// Texture-presence bits from bit 2: every OpenPBR input plus glTF occlusion.
pub const TEXTURE_FEATURES: usize = 40;
const _: () = assert!(TEXTURE_FEATURES + 2 <= 44);

/// The control-pass fragment stage a variant compiles: the opaque or the
/// nearest transparent control targets (colour is the decoupled shade's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Controls,
    OpaqueControls,
}
impl Stage {
    /// The fragment stages, each with a precompiled generic, indexed by discriminant
    /// (SlangForward::generic).
    pub const ALL: [Self; 2] = [Self::Controls, Self::OpaqueControls];
}
const _: () = {
    let mut i = 0;
    while i < Stage::ALL.len() {
        assert!(Stage::ALL[i] as usize == i);
        i += 1;
    }
};

/// The compile-time capabilities of one control-pass pipeline: a materials.rs
/// feature word. `true` is always correct: every capability's code is gated on
/// its material weight as well, so a key with more capabilities renders the
/// same pixels. Controls light nothing, so no shadow technique specializes them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForwardKey {
    pub stage: Stage,
    pub features: FeatureMask,
}

impl ForwardKey {
    pub const fn new(stage: Stage, features: FeatureMask) -> Self {Self {stage, features}}
    /// The precompiled generic (the superset fallback) for a stage, with or
    /// without hashed coverage.
    pub const fn generic(stage: Stage, hashed: bool) -> Self {
        Self::new(stage, (SHADING_LAYERS | SHADING_METAL) as FeatureMask | if hashed {FEATURE_DITHERED} else {0})
    }
    pub fn is_generic(self) -> bool {self == Self::generic(self.stage, self.hashed())}
    /// The entry declares SV_PrimitiveID for hashed coverage (raster_accepts).
    pub const fn hashed(self) -> bool {self.features & FEATURE_DITHERED != 0}
    /// The entry point, (entry, stage). Its vertex stage is the precompiled
    /// `depth.main_vs`, shared with the depth prepass.
    pub fn entry(self) -> (&'static str, &'static str) {
        let entry = match self.stage {
            Stage::Controls => "transparent_controls_fs",
            Stage::OpaqueControls => "controls_fs",
        };
        (entry, "fragment")
    }
    /// The module name, also its file stem.
    pub fn name(self) -> String {
        let stage = match self.stage {
            Stage::Controls => "controls",
            Stage::OpaqueControls => "opaque_controls",
        };
        if self.is_generic() {
            return format!("{stage}_generic{}", if self.hashed() {"_hashed"} else {""});
        }
        format!("{stage}_{:016x}", self.features)
    }
    /// The generated composition: one conforming type and the stage entry. An
    /// unspecialized word keeps every capability, as the WGSL overrides' defaults do.
    /// `programs` is a scene's `struct Programs : IProgram` and its imports (Slang),
    /// applied in place of NoPrograms.
    pub fn module_with(self, programs: Option<&str>) -> String {
        let program = if programs.is_some() { "Programs" } else { "NoPrograms" };
        let (entry, _) = self.entry();
        let function = match self.stage {Stage::Controls => "transparent_controls", Stage::OpaqueControls => "controls"};
        // Primitive identity preserves averaging on one surface; pixel depth slopes join continuous creases.
        let (primitive, assign) = (", uint primitive : SV_PrimitiveID", "v.primitive=primitive;");
        let body = format!(
            "import forward;\n#ifdef VORK_SOFTWARE_RASTER\nimport software_surface;\nimport frame;\nimport vork_constants;\n#define SOFTWARE_SAMPLE , float2 sample_position:SV_VulkanSamplePosition\n#else\n#define SOFTWARE_SAMPLE\n#endif\n[shader(\"fragment\")]\n\
             Output {entry}(Varying v, bool front : SV_IsFrontFace{primitive} SOFTWARE_SAMPLE) {{{assign}\n#ifdef VORK_SOFTWARE_RASTER\nif ((objects[v.object].metadata.y&REG_OBJECT_IMPLICIT)==0u) v=software_surface(v.object,primitive,v.position,floor(v.position.xy)+sample_position,float2(0.));\n#endif\nreturn {function}<Key,{program}>(v,front);}}\n"
        );
        variant_module(self.features, false, false, false, programs, &body)
    }
}

/// Values shared by literal Slang keys and late SPIR-V specialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability { Bool(bool), Uint(u32) }
impl Capability {
    fn ty(self) -> &'static str { match self { Self::Bool(_)=>"bool", Self::Uint(_)=>"uint" } }
    fn literal(self) -> String { match self { Self::Bool(value)=>value.to_string(), Self::Uint(value)=>format!("{value}u") } }
}

// Field order is the internal SPIR-V SpecId ABI. Both the shipped declarations
// and per-scene values come from this list, including texture masks and light kinds.
pub fn variant_capabilities(f: FeatureMask, ray_shadows: bool, subsurface: bool, emitter_profiles: bool) -> Vec<(&'static str, Capability)> {
    let specialized = f & FEATURE_SPECIALIZED != 0;
    let bit = |b: FeatureMask| !specialized || f & b != 0;
    let record = |b: FeatureMask| if f & FEATURE_EXACT_RECORD != 0 {f & b != 0} else {bit(b)};
    // A scene union is a proof only in a key that is one. Every other key, the
    // specialized raster and hit keys included, keeps the capability.
    let scene = |b: FeatureMask| f & FEATURE_INSTALLED == 0 || f & b != 0;
    let textures = (f >> 2) & ((1 << TEXTURE_FEATURES) - 1);
    let (lo, hi, light_types) = if specialized {
        (
            textures as u32,
            (textures >> 32) as u32,
            ((f & FEATURE_LIGHT_TYPES) >> FEATURE_LIGHT_SHIFT) as u32,
        )
    } else {
        (u32::MAX, u32::MAX, 15)
    };
    vec![
        ("layered", Capability::Bool(f & SHADING_LAYERS as FeatureMask != 0)),
        ("metal", Capability::Bool(f & SHADING_METAL as FeatureMask != 0)),
        ("ray_shadows", Capability::Bool(ray_shadows)),
        ("subsurface", Capability::Bool(subsurface)),
        ("specialized", Capability::Bool(specialized)),
        ("textures_lo", Capability::Uint(lo)),
        ("textures_hi", Capability::Uint(hi)),
        ("has_coat", Capability::Bool(bit(FEATURE_COAT))),
        ("has_fuzz", Capability::Bool(bit(FEATURE_FUZZ))),
        ("has_anisotropy", Capability::Bool(bit(FEATURE_ANISOTROPY))),
        ("has_film", Capability::Bool(bit(FEATURE_FILM))),
        ("has_tint", Capability::Bool(bit(FEATURE_TINT))),
        ("has_mask", Capability::Bool(bit(FEATURE_MASK))),
        ("has_punctual", Capability::Bool(bit(FEATURE_PUNCTUAL))),
        ("has_finite", Capability::Bool(bit(FEATURE_FINITE))),
        ("has_rectangle", Capability::Bool(bit(FEATURE_RECTANGLE))),
        ("has_ellipse", Capability::Bool(bit(FEATURE_ELLIPSE))),
        ("has_shadow_rays", Capability::Bool(bit(FEATURE_SHADOW_RAYS))),
        ("has_transmission", Capability::Bool(bit(FEATURE_TRANSMISSION))),
        ("has_solid_transmission", Capability::Bool(bit(FEATURE_SOLID_TRANSMISSION))),
        ("has_emitter_profiles", Capability::Bool(emitter_profiles)),
        ("has_camera_receivers", Capability::Bool(record(FEATURE_CAMERA_RECEIVERS))),
        ("has_reflections", Capability::Bool(record(FEATURE_REFLECTIONS))),
        ("has_indirect_controls", Capability::Bool(record(FEATURE_INDIRECT_CONTROLS))),
        ("has_lobes", Capability::Bool(bit(FEATURE_LOBES))),
        ("has_lit", Capability::Bool(bit(FEATURE_LIT))),
        ("light_types", Capability::Uint(light_types)),
        ("has_implicit_geometry", Capability::Bool(bit(FEATURE_IMPLICIT_GEOMETRY))),
        ("has_coat_anisotropy", Capability::Bool(bit(FEATURE_COAT_ANISOTROPY))),
        ("has_weighted_ior", Capability::Bool(bit(FEATURE_WEIGHTED_IOR))),
        ("has_gltf_tint", Capability::Bool(bit(FEATURE_GLTF_TINT))),
        ("has_null_connection", Capability::Bool(scene(FEATURE_NULL_CONNECTION))),
        // Every key keeps camera containment except one carrying the proof.
        ("has_camera_interior", Capability::Bool(!specialized || f & FEATURE_CAMERA_EXTERIOR == 0)),
        ("has_subsurface", Capability::Bool(scene(FEATURE_SUBSURFACE))),
        ("has_hair", Capability::Bool(scene(FEATURE_HAIR))),
        ("installed", Capability::Bool(f & FEATURE_INSTALLED != 0)),
    ]
}

// Static-only policies need an explicit empty initializer: pinned Slang
// crashes synthesizing one under -zero-initialize with the combined capabilities.
pub fn variant_module(f: FeatureMask, ray_shadows: bool, subsurface: bool, emitter_profiles: bool, programs: Option<&str>, body: &str) -> String {
    use std::fmt::Write;
    let mut text=format!("// Generated by forward_variant.rs: features {f:#x}.\nimport variant;\nimport stream;\nimport surface;\n{}\nstruct Key : IVariant {{\n    __init() {{}}\n",programs.unwrap_or_default());
    for (name,value) in variant_capabilities(f,ray_shadows,subsurface,emitter_profiles) {
        writeln!(text,"    static const {} {name} = {};",value.ty(),value.literal()).unwrap();
    }
    write!(text,"}};\n{body}").unwrap();
    text
}

/// Compile the capability graph once; the Vulkan driver folds scene values at
/// pipeline creation. No material feature combination requires another Slang artifact.
#[allow(dead_code)] // Used by build_slang.rs and interface tests.
pub fn late_variant_module(body: &str, ray_shadows: bool) -> String {
    use std::fmt::Write;
    let fields=variant_capabilities(0,false,false,false);
    let mut text="import variant;\nimport stream;\nimport surface;\n".to_owned();
    for (id,(name,value)) in fields.iter().enumerate() {
        if *name=="ray_shadows" || *name=="subsurface" && !ray_shadows {continue;}
        let zero=match value {Capability::Bool(_)=>"false",Capability::Uint(_)=>"0u"};
        writeln!(text,"[vk::constant_id({id})] const {} vork_cap_{name} = {zero};",value.ty()).unwrap();
    }
    text.push_str("struct Key : IVariant {\n    __init() {}\n");
    for (name,value) in fields {
        // This capability selects descriptor types, so it belongs to the
        // resource interface rather than the driver's late feature values.
        if name=="ray_shadows" {
            writeln!(text,"    static const bool ray_shadows = {ray_shadows};").unwrap();
            continue;
        }
        if name=="subsurface" && !ray_shadows {
            text.push_str("    static const bool subsurface = false;\n");
            continue;
        }
        writeln!(text,"    static const {} {name} = vork_cap_{name};",value.ty()).unwrap();
    }
    write!(text,"}};\n{body}").unwrap();
    text
}

/// slangc arguments shared by the build and runtime variant compiles.
pub const SLANG_FLAGS: &[&str] = &["-matrix-layout-row-major"];
/// Native speed profile, shared by build-time kernels and runtime path compilation.
pub const NATIVE_OPTIMIZATION: &str = "-O2";
/// Per backend: (file extension, slangc target, target flags).
pub const SPIRV_TARGET: (&str, &str, &[&str]) = (
    "spv",
    "spirv",
    &["-emit-spirv-directly", "-fvk-use-entrypoint-name"],
);
pub fn material_registers(material: u32) -> Vec<String> {
    crate::binding_abi::material_registers(material)
}

/// The irradiance volume's three consecutive Metal buffer registers.
pub fn probe_registers(first: u32) -> [String; 3] {
    std::array::from_fn(|offset| format!("-DVORK_PROBE_{}_REGISTER={}{}", ["IRRADIANCE", "VISIBILITY", "GRID"][offset], if offset==2 {"u"}else{"t"}, first + offset as u32))
}

/// Compute hit-kernel tables start after that kernel's group-0 buffers.
/// Shared by embedded compilation and scene specializations; native layout tests
/// verify every index against the renderer's actual bind-group entries.
pub fn hit_kernel_registers(module: &str, adaptive: bool, probes: bool) -> Vec<String> {
    let material = match module {
        "gi_ray" => 11,
        "reflections_hits" => 14,
        "probe_update" => 14,
        "path" => 18 + u32::from(adaptive),
        _ => return Vec::new(),
    };
    let mut defines = material_registers(material);
    defines.push("-DVORK_HIT_KERNEL".to_owned());
    defines.push(format!("-DVORK_HIT_REGISTER=b{}",material+5+if probes {3} else {0}));
    if module == "gi_ray" {
        defines.push(format!("-DVORK_GI_QUERY_REGISTER=u{}",material+6+if probes {3} else {0}));
        defines.push(format!("-DVORK_GI_GUIDE_REGISTER=u{}",material+7+if probes {3} else {0}));
        defines.push(format!("-DVORK_GI_LIGHT_REGISTER=t{}",material+8+if probes {3} else {0}));
    }
    if module == "reflections_hits" {
        defines.push(format!("-DVORK_REFLECTION_FIELDS_REGISTER=b{}",material+6+if probes {3} else {0}));
        defines.push(format!("-DVORK_REFLECTION_MEDIUM_REGISTER=b{}",material+7+if probes {3} else {0}));
    }
    if module == "path" {
        defines.push("-DVORK_PATH_ENVIRONMENT_REGISTER=t16".to_owned());
        defines.push("-DVORK_CATCHER_REGISTER=u14".to_owned());
        defines.push("-DVORK_CATCHER_PIXELS_REGISTER=u15".to_owned());
    }
    if module == "path" && adaptive {defines.push("-DVORK_PATH_ADAPTIVE_NATIVE".to_owned());}
    if probes {
        defines.extend(probe_registers(material + 5));
    }
    defines
}

// Native callbacks need link-time visibility, not externally callable library
// exports. Slang keeps `export` roots alive with unspecialized mutable parameters,
// which Metal cannot emit (shader-slang/slang#11981). C linkage lets unused
// callbacks disappear and reachable callbacks specialize from the shader entry.
// The token macros leave strings/comments and authored program analysis alone.
// MSL carries no `#line` paths: Metal's system cache keys libraries by source
// text, so build or workspace paths would make identical shaders miss it.
pub const METAL_TARGET: (&str, &str, &[&str]) = ("metal", "metal", &["-line-directive-mode", "none", "-DVORK_METAL", "-Dexport=[__unsafeForceInlineEarly] __extern_cpp", "-Dextern=__extern_cpp extern"]);

#[cfg(test)]
mod capability_tests {
    use super::*;
    #[test]
    fn implicit_geometry_is_conservative_for_generic_keys() {
        let has_fields = |features| variant_capabilities(features, false, false, false)
            .into_iter().find(|(name, _)| *name == "has_implicit_geometry").unwrap().1;
        assert_eq!(has_fields(0), Capability::Bool(true));
        assert_eq!(has_fields(FEATURE_SPECIALIZED), Capability::Bool(false));
        assert_eq!(has_fields(FEATURE_SPECIALIZED | FEATURE_IMPLICIT_GEOMETRY), Capability::Bool(true));
        // Existing late-specialization IDs stay stable when capabilities are appended.
        assert_eq!(variant_capabilities(0, false, false, false)[27].0, "has_implicit_geometry");
    }

    #[test]
    fn coat_and_dielectric_capabilities_are_independent() {
        let fields = [("has_coat_anisotropy", FEATURE_COAT_ANISOTROPY),
            ("has_weighted_ior", FEATURE_WEIGHTED_IOR), ("has_gltf_tint", FEATURE_GLTF_TINT)];
        for mask in 0..8 {
            let features = fields.iter().enumerate().fold(FEATURE_SPECIALIZED,
                |f, (i, (_, bit))| if mask & (1 << i) != 0 {f | bit} else {f});
            let capabilities = variant_capabilities(features, false, false, false);
            for (i, (name, _)) in fields.iter().enumerate() {
                assert_eq!(capabilities.iter().find(|(field, _)| field == name).unwrap().1,
                    Capability::Bool(mask & (1 << i) != 0), "{name}, mask {mask}");
            }
        }
    }

    #[test]
    fn dielectric_capabilities_fold_only_with_installed_proofs() {
        for (field,bit) in [("has_weighted_ior",FEATURE_WEIGHTED_IOR),("has_gltf_tint",FEATURE_GLTF_TINT)] {
            let value=|features| variant_capabilities(features,false,false,false).into_iter().find(|(name,_)|*name==field).unwrap().1;
            assert_eq!(value(0),Capability::Bool(true));
            assert_eq!(value(FEATURE_SPECIALIZED),Capability::Bool(false));
            assert_eq!(value(FEATURE_SPECIALIZED|bit),Capability::Bool(true));
        }
    }

    /// Raster refraction, shipped surfaces and every other key that never
    /// receives the installed proof must keep camera containment.
    #[test]
    fn camera_containment_stays_unless_a_key_carries_the_proof() {
        let interior=|features| variant_capabilities(features,false,false,false).into_iter().find(|(name,_)|*name=="has_camera_interior").unwrap().1;
        assert_eq!(interior(0),Capability::Bool(true));
        assert_eq!(interior(FEATURE_CAMERA_EXTERIOR),Capability::Bool(true));
        assert_eq!(interior(FEATURE_SPECIALIZED),Capability::Bool(true));
        assert_eq!(interior(FEATURE_SPECIALIZED|FEATURE_TRANSMISSION|FEATURE_SOLID_TRANSMISSION),Capability::Bool(true));
        assert_eq!(interior(FEATURE_SPECIALIZED|FEATURE_CAMERA_EXTERIOR),Capability::Bool(false));
        assert!(variant_module(FEATURE_SPECIALIZED,false,false,false,None,"").contains("static const bool has_camera_interior = true;"));
    }

    /// The generated key module is compiler input, so the compiled-shader cache
    /// identity follows the feature set: one key, one text, and any capability
    /// added or removed is a different kernel.
    #[test]
    fn a_kernel_key_names_its_scene_unions() {
        let module=|features| variant_module(FEATURE_SPECIALIZED|FEATURE_INSTALLED|features,false,false,false,None,"");
        for (name,bit) in [("has_null_connection",FEATURE_NULL_CONNECTION),("has_subsurface",FEATURE_SUBSURFACE),("has_hair",FEATURE_HAIR)] {
            assert!(module(0).contains(&format!("static const bool {name} = false;")),"{name}");
            assert!(module(bit).contains(&format!("static const bool {name} = true;")),"{name}");
            assert_ne!(module(0),module(bit),"{name}");
        }
        assert!(module(0).contains("static const bool installed = true;"));
    }

    /// A scene union is absent from a raster or hit key because that key is not the
    /// scene's, not because the scene lacks it. Only a key that says it is the
    /// union turns the absence into a proof; every other key keeps the capability.
    #[test]
    fn scene_unions_are_proofs_only_in_a_scene_key() {
        let value=|features,field:&str| variant_capabilities(features,false,false,false).into_iter().find(|(name,_)|*name==field).unwrap().1;
        for (field,bit) in [("has_null_connection",FEATURE_NULL_CONNECTION),("has_subsurface",FEATURE_SUBSURFACE),("has_hair",FEATURE_HAIR)] {
            for other in [0,FEATURE_SPECIALIZED,FEATURE_SPECIALIZED|FEATURE_COAT|FEATURE_TRANSMISSION,(SHADING_LAYERS|SHADING_METAL) as FeatureMask] {
                assert_eq!(value(other,field),Capability::Bool(true),"{field}: {other:#x} is not a scene key");
            }
            assert_eq!(value(FEATURE_SPECIALIZED|FEATURE_INSTALLED,field),Capability::Bool(false),"{field}");
            assert_eq!(value(FEATURE_SPECIALIZED|FEATURE_INSTALLED|bit,field),Capability::Bool(true),"{field}");
        }
        for (features,installed) in [(0,false),(FEATURE_SPECIALIZED,false),(u128::MAX & !FEATURE_INSTALLED,false),(FEATURE_SPECIALIZED|FEATURE_INSTALLED,true)] {
            assert_eq!(value(features,"installed"),Capability::Bool(installed),"{features:#x}");
        }
    }

    /// One installed light kind is the only light fact a transport kernel folds
    /// on. Every other mix is the all-kinds key, as the unspecialized key has it.
    #[test]
    fn a_key_names_a_light_kind_only_when_it_is_the_only_one() {
        let kind=|index:u32| (1 as FeatureMask) << (FEATURE_LIGHT_SHIFT+index);
        let light_types=|features| variant_capabilities(FEATURE_SPECIALIZED|FEATURE_INSTALLED|features,false,false,false)
            .into_iter().find(|(name,_)|*name=="light_types").unwrap().1;
        for index in 0..4 {
            assert_eq!(folded_light_kinds(kind(index)),kind(index));
            assert_eq!(light_types(folded_light_kinds(kind(index))),Capability::Uint(1<<index));
            // Other installed light facts never change which kinds the key names.
            assert_eq!(folded_light_kinds(kind(index)|FEATURE_SHADOW_RAYS|FEATURE_FINITE|FEATURE_ELLIPSE),kind(index));
        }
        // No analytic light and every mix of kinds are one key.
        let mut mixes=vec![0];
        for mask in 0..16u32 {
            if mask.count_ones()>1 {mixes.push((mask as FeatureMask)<<FEATURE_LIGHT_SHIFT);}
        }
        assert_eq!(mixes.len(),12);
        for mix in mixes {
            assert_eq!(folded_light_kinds(mix),FEATURE_LIGHT_TYPES,"{mix:#x}");
            assert_eq!(light_types(folded_light_kinds(mix)),Capability::Uint(15),"{mix:#x}");
        }
        let module=|features| variant_module(FEATURE_SPECIALIZED|FEATURE_INSTALLED|folded_light_kinds(features),false,false,false,None,"");
        assert_eq!(module(0),module(kind(0)|kind(3)),"equal kernels have one key text");
        assert_eq!(module(kind(1)|kind(2)),module(FEATURE_LIGHT_TYPES));
        assert_ne!(module(kind(0)),module(kind(3)),"a single kind stays its own kernel");
        assert_ne!(module(kind(0)),module(0));
    }

    #[test]
    fn late_capabilities_cover_the_shader_interface() {
        let source=include_str!("../shaders/variant.slang");
        let interface=source.split_once("interface IVariant {").unwrap().1.split_once("};").unwrap().0;
        let names:std::collections::BTreeSet<_>=interface.lines().filter_map(|line| {
            let line=line.trim().strip_prefix("static const ")?;
            Some(line.split_whitespace().nth(1).unwrap().trim_end_matches(';'))
        }).collect();
        let fields=variant_capabilities(FEATURE_SPECIALIZED,false,false,false);
        assert_eq!(names,fields.iter().map(|(name,_)|*name).collect());
        assert_eq!(names.len(),fields.len(),"every capability has one SpecId");
        let module=late_variant_module("",false);
        for (id,(name,_)) in fields.iter().enumerate() {
            if *name=="ray_shadows" {assert!(module.contains("ray_shadows = false;"));continue;}
            if *name=="subsurface" {assert!(module.contains("subsurface = false;"));continue;}
            assert!(module.contains(&format!("[vk::constant_id({id})]")));
            assert!(module.contains(&format!("{name} = vork_cap_{name};")));
        }
    }
    /// Every control stage retains primitive identity for depth resolve.
    #[test]
    fn control_stages_declare_the_primitive() {
        let declares=|key:ForwardKey,programs:Option<&str>| {
            let module=key.module_with(programs);
            let entry=module.lines().find(|l|l.contains(key.entry().0)).expect("entry line").to_owned();
            assert_eq!(entry.contains("SV_PrimitiveID"),entry.contains("v.primitive=primitive;"),"{entry}");
            entry.contains("SV_PrimitiveID")
        };
        for stage in Stage::ALL {
            for hashed in [false,true] {
                let key=ForwardKey::generic(stage,hashed);
                assert!(declares(key,None),"{stage:?}");
            }
            assert!(declares(ForwardKey::new(stage,FEATURE_SPECIALIZED|FEATURE_MASK),Some("")),"programs may read ctx.primitive_id");
            assert!(ForwardKey::generic(stage,false).is_generic() && ForwardKey::generic(stage,true).is_generic());
            assert_ne!(ForwardKey::generic(stage,false).name(),ForwardKey::generic(stage,true).name());
        }
    }

    #[test]
    fn response_capabilities_do_not_alias_transparent_transport() {
        let response=FEATURE_ANISOTROPY|FEATURE_FILM;
        let transport=FEATURE_TRANSMISSION|FEATURE_SOLID_TRANSMISSION;
        assert_eq!(response&transport,0);
        for (flags,aniso,film,trans,solid) in [
            (FEATURE_ANISOTROPY,true,false,false,false),
            (FEATURE_FILM,false,true,false,false),
            (FEATURE_TRANSMISSION,false,false,true,false),
            (transport,false,false,true,true),
            (response|transport,true,true,true,true),
        ] {
            let module=variant_module(FEATURE_SPECIALIZED|flags,false,false,false,None,"");
            for (name,enabled) in [("has_anisotropy",aniso),("has_film",film),("has_transmission",trans),("has_solid_transmission",solid)] {
                assert!(module.contains(&format!("static const bool {name} = {enabled};")),"{name}, flags {flags:x}");
            }
        }
    }
}
