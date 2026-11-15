// The compile-time capabilities one forward pipeline is specialised for. Each
// generated variant module names one conforming type; every query below folds to a
// constant, so absent capabilities cost neither instructions nor registers. An
// unspecialized key (the superset fallback) has every capability: `true` is always
// correct because each gate also tests its material weight.
interface IVariant {
    // OpenPBR coat and fuzz layers.
    static const bool layered;
    // F82-tint metal Fresnel.
    static const bool metal;
    // Shadow rays for profiles with settings.shadow_method: ray (the scene binds its TLAS).
    static const bool ray_shadows;
    static const bool subsurface;
    // Per-draw-group capabilities (materials.rs FEATURE_*); all true when unspecialized.
    static const bool specialized;
    // Texture capability per OpenPBR input plus glTF occlusion, inputs 0-31 and 32-63.
    static const uint textures_lo;
    static const uint textures_hi;
    static const bool has_transmission;
    static const bool has_solid_transmission;
    static const bool has_coat;
    static const bool has_fuzz;
    static const bool has_anisotropy;
    static const bool has_coat_anisotropy;
    static const bool has_film;
    static const bool has_tint;
    static const bool has_mask;
    static const bool has_punctual;
    static const bool has_finite;
    static const bool has_rectangle;
    static const bool has_ellipse;
    // Installed light-shadow support, separate from the ray-query technique.
    static const bool has_shadow_rays;
    static const bool has_emitter_profiles;
    static const bool has_camera_receivers;
    static const bool has_reflections;
    static const bool has_indirect_controls;
    // Blend programs: shade each lobe of a mixed surface.
    static const bool has_lobes;
    // Programs with a lit stage (Shader to RGB).
    static const bool has_lit;
    // The scene's light kinds as bits (sun, point, spot, rect); 15 is any.
    static const uint light_types;
    static const bool has_implicit_geometry;
    static const bool has_weighted_ior;
    static const bool has_gltf_tint;
    // Some installed material can pass a straight path connection through a
    // covered point; false proves every covered candidate blocks it.
    static const bool has_null_connection;
    // Some camera sample may start inside a solid. Only a path key carrying
    // the installed proof (FEATURE_CAMERA_EXTERIOR) is false.
    static const bool has_camera_interior;
    // Scene unions of a transport kernel's key (surface.slang installed_surface):
    // some installed closure can have a subsurface lobe; some installed material
    // is a fibre program. Distinct from `subsurface`, the raster diffusion technique.
    static const bool has_subsurface;
    static const bool has_hair;
    // The key is the union of every installed material. Only then are the
    // capabilities it lacks absent from the whole scene, and only then does
    // installed_surface state them.
    static const bool installed;
}
// Whether `input` may be textured in this variant.
bool input_textured<V:IVariant>(uint input) {
    if (input<32u) {return (V.textures_lo&(1u<<input))!=0u;}
    return (V.textures_hi&(1u<<(input-32u)))!=0u;
}
// The depth prepass and every unspecialized pipeline: every capability.
struct Unspecialized : IVariant {
    // Static-only policy: avoid Slang 2026.21 synthesizing a zero initializer.
    __init() {}
    static const bool layered = true;
    static const bool metal = true;
    static const bool ray_shadows = false;
    static const bool subsurface = false;
    static const bool specialized = false;
    static const uint textures_lo = 0xffffffffu;
    static const uint textures_hi = 0xffffffffu;
    static const bool has_transmission = true;
    static const bool has_solid_transmission = true;
    static const bool has_coat = true;
    static const bool has_fuzz = true;
    static const bool has_anisotropy = true;
    static const bool has_coat_anisotropy = true;
    static const bool has_film = true;
    static const bool has_tint = true;
    static const bool has_mask = true;
    static const bool has_punctual = true;
    static const bool has_finite = true;
    static const bool has_rectangle = true;
    static const bool has_ellipse = true;
    static const bool has_shadow_rays = true;
    static const bool has_emitter_profiles = true;
    static const bool has_camera_receivers = true;
    static const bool has_reflections = true;
    static const bool has_indirect_controls = true;
    static const bool has_lobes = true;
    static const bool has_lit = true;
    static const uint light_types = 15u;
    static const bool has_implicit_geometry = true;
    static const bool has_weighted_ior = true;
    static const bool has_gltf_tint = true;
    static const bool has_null_connection = true;
    static const bool has_camera_interior = true;
    static const bool has_subsurface = true;
    static const bool has_hair = true;
    static const bool installed = false;
};
