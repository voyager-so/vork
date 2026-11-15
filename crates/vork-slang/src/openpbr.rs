//! OpenPBR types and composition follow the single checked-in parameter table.
use super::*;
use vork_scene::v2::{OPENPBR, OpenPbrKind};

pub fn openpbr_fields() -> Vec<Field> {
    OPENPBR
        .iter()
        .filter(|p| p.closure)
        .map(|p| {
            let (ty, default, frame) = match p.kind {
                OpenPbrKind::Float => (Type::Float, Some(format!("{:?}", p.default[0])), false),
                OpenPbrKind::Bool => (Type::Bool, Some((p.default[0] != 0.).to_string()), false),
                OpenPbrKind::Color3 => (
                    Type::Vector(3),
                    Some(format!(
                        "float3({})",
                        p.default
                            .iter()
                            .map(|v| format!("{v:?}"))
                            .collect::<Vec<_>>()
                            .join(",")
                    )),
                    false,
                ),
                OpenPbrKind::Vector3 => (Type::Vector(3), None, true),
            };
            Field {
                frame,
                ..Field::new(p.name, ty, default.as_deref())
            }
        })
        .collect()
}

pub(crate) fn module(closure: &[Field]) -> String {
    let fields = openpbr_fields();
    let mut s = String::from("// OpenPBR inputs.\npublic struct OpenPbr {\n");
    for f in &fields {
        writeln!(
            s,
            "public {} = {};",
            f.slang(),
            f.default.as_deref().unwrap_or("0")
        )
        .unwrap();
    }
    s.push_str("};\npublic OpenPbr openpbr(Closure c) { OpenPbr p = {};\n");
    for f in closure
        .iter()
        .filter(|f| fields.iter().any(|p| p.name == f.name))
    {
        writeln!(s, "p.{0}=c.{0};", f.name).unwrap();
    }
    s.push_str(
        "return p; }\npublic Closure mix_closure(Closure a, Closure b, float w) { Closure c = {};\n",
    );
    // Shared values are kept exactly, one scalar at a time (see effects.rs, Mix).
    for (_, f, at) in super::slots(closure) {
        let x = format!("{}{at}", f.name);
        let value = if matches!(f.ty, Type::Bool | Type::Int | Type::Uint) {
            format!("w < 0.5 ? a.{x} : b.{x}")
        } else {
            format!("a.{x} == b.{x} ? a.{x} : lerp(a.{x}, b.{x}, w)")
        };
        writeln!(s, "c.{x}={value};").unwrap();
    }
    s.push_str("return c; }\n");
    if [
        "base_weight",
        "specular_weight",
        "emission_color",
        "emission_luminance",
    ]
    .iter()
    .all(|name| closure.iter().any(|f| f.name == *name))
    {
        s.push_str("public Closure unlit(float3 color) { Closure c = {};\n");
        for name in [
            "base_weight",
            "specular_weight",
            "coat_weight",
            "fuzz_weight",
            "transmission_weight",
            "subsurface_weight",
        ] {
            if closure.iter().any(|f| f.name == name) {
                writeln!(s, "c.{name}=0;").unwrap();
            }
        }
        s.push_str("c.emission_color=color; c.emission_luminance=1; return c; }\n");
        if closure.iter().any(|f| f.name == "geometry_opacity") {
            s.push_str("public Closure unlit(float3 color, float alpha) { Closure c=unlit(color); c.geometry_opacity=alpha; return c; }\n");
        }
    }
    s.push_str(&blend_module(closure));
    s
}

pub(super) fn emission_effect(closure: &[Field]) -> String {
    if ["emission_color", "emission_luminance"].iter().all(|name| closure.iter().any(|f| f.name == *name)) {
        format!("(c.emission_color.x*c.emission_color.x+c.emission_color.y*c.emission_color.y+c.emission_color.z*c.emission_color.z)*c.emission_luminance*c.emission_luminance{}",
            if closure.iter().any(|f|f.name=="geometry_opacity") {"*c.geometry_opacity*c.geometry_opacity"} else {""})
    } else { "0.".into() }
}

/// Solid-interior effect, separate from the thin sheet scattering response.
pub(super) fn spatial_subsurface(closure: &[Field]) -> String {
    if !closure.iter().any(|f| f.name == "subsurface_weight") { return "0.".into(); }
    if closure.iter().any(|f| f.name == "geometry_thin_walled") {
        "c.geometry_thin_walled ? 0. : c.subsurface_weight*c.subsurface_weight".into()
    } else { "c.subsurface_weight*c.subsurface_weight".into() }
}

/// `Blend` sums independently shaded closures, rather than interpolating them.
fn blend_module(closure: &[Field]) -> String {
    blend_module_with_capacity(closure, BLEND_LOBES, false)
}

fn blend_module_with_capacity(closure: &[Field], capacity: u32, layer_effects: bool) -> String {
    let square = |field: &str, value: &str| if closure.iter().any(|f| f.name == field) { format!("({value})*({value})") } else { "0.".into() };
    let dispersion=if ["transmission_weight","transmission_dispersion_scale"].into_iter().all(|name|closure.iter().any(|f|f.name==name)) {
        square("transmission_dispersion_scale","c.transmission_weight*c.transmission_dispersion_scale")
    } else {"0.".into()};
    let opacity = if closure.iter().any(|f| f.name == "geometry_opacity") {
        "s.lobes[i].geometry_opacity"
    } else {
        "1."
    };
    let set_opacity = if closure.iter().any(|f| f.name == "geometry_opacity") {
        "c.geometry_opacity=opacity;"
    } else {
        ""
    };
    // The WGSL/effect representation stores full lobes. Native replay captures
    // the Closure directly, avoiding a second nominal type and field adapters.
    let copy = |from: &str, to: &str| closure.iter().map(|f| format!("{to}.{0}={from}.{0}; ", f.name)).collect::<String>();
    let (to_lobe, to_closure) = (copy("c", "l"), copy("l", "c"));
    let lobe_fields = closure.iter().map(|f| format!("{}; ", f.slang())).collect::<String>();
    let lobe_storage=format!("struct BlendLobe {{ {lobe_fields}}};\nBlendLobe blend_pack(__constref Closure c) {{ BlendLobe l; {to_lobe}return l; }}\nClosure blend_closure(__constref BlendLobe l) {{ Closure c={{}}; {to_closure}return c; }}\nBlendLobe blend_lobe(__constref Closure c) {{ return blend_pack(c); }}\n");
    let (lobe_return,selected,tail)=("Closure","var c=blend_closure(s.lobes[pick]);","Closure d={}; s.lobes[i]=blend_lobe(d);");
    // The opaque probe calls prevent Slang from cancelling dependencies between
    // different closure components or complementary mixture weights.
    let wavelength=slots(closure).into_iter().map(|(_,f,at)|format!("{WAVELENGTH_EFFECT}({})",to_f32_slang(f.ty,&format!("c.{}{at}",f.name)))).collect::<Vec<_>>().join("+");
    let wavelength=if wavelength.is_empty() {"0."} else {&wavelength};
    // Count and effect summaries do not depend on the storage append loop. Keep
    // them explicit so the effect probe can prove bounds and absent features
    // without expanding capacity-many aggregate copies.
    format!(
        "{lobe_storage}\n\
         public struct Blend {{ internal BlendLobe lobes[{capacity}]; internal float weights[{capacity}]; internal uint count; {effect_fields}{layer_field} }};\n\
         public Blend blend(__constref Closure c) {{ Blend s; for (uint i=0u;i<{capacity}u;i++) {{ {tail} s.weights[i]=0.; }} s.lobes[0]=blend_lobe(c); s.weights[0]=1.; s.count=1u; {effect_init}{layer_init} return s; }}\n\
         void blend_append(inout Blend a, __constref Blend b, float w) {{\n    \
         for (uint i=0u;i<{capacity}u;i++) {{\n        \
         if (i>=b.count || a.count+i>={capacity}u) {{ break; }}\n        \
         a.lobes[a.count+i]=b.lobes[i]; a.weights[a.count+i]=b.weights[i]*w;\n    }}\n    \
         a.count=min(a.count+b.count,{overflow}u); {effect_sum} {layer_sum} }}\n\
         Blend blend_sum(__constref Blend a, __constref Blend b, float wa=1., float wb=1.) {{ Blend r={{}}; blend_append(r,a,wa); blend_append(r,b,wb); return r; }}\n\
         // The compiler's effect slice (subsurface, transparency, transmission) of every\n\
         // closure in the Blend, as sums of squares so a constant zero proves absence.\n\
         public uint vork_blend_count(__constref Blend s) {{ return s.count; }}\n\
         {layer_reader}\n\
         {effect_readers}\n\
         // Both closures at full weight.\n\
         public Blend add(__constref Blend a, __constref Blend b) {{ return blend_sum(a,b); }}\n\
         public Blend add(__constref Closure a, __constref Blend b) {{ return blend_sum(blend(a),b); }}\n\
         public Blend add(__constref Blend a, __constref Closure b) {{ return blend_sum(a,blend(b)); }}\n\
         public Blend add(__constref Closure a, __constref Closure b) {{ return blend_sum(blend(a),blend(b)); }}\n\
         // `a` weighted 1-weight and `b` weighted weight, weight clamped to [0,1].\n\
         // Each closure is shaded on its own; `mix_closure` interpolates parameters instead.\n\
         public Blend blend(__constref Blend a, __constref Blend b, float weight) {{ let f=saturate(weight); return blend_sum(a,b,1.-f,f); }}\n\
         public Blend blend(__constref Closure a, __constref Blend b, float weight) {{ return blend(blend(a),b,weight); }}\n\
         public Blend blend(__constref Blend a, __constref Closure b, float weight) {{ return blend(a,blend(b),weight); }}\n\
         public Blend blend(__constref Closure a, __constref Closure b, float weight) {{ return blend(blend(a),blend(b),weight); }}\n\
         // The renderer's view of a Blend. As for Blender's Transparent BSDF, the\n\
         // transparency weights add: T = sum weight_i * (1 - opacity_i), the surface\n\
         // covers O = 1 - min(T, 1) and every lobe reports O. Lobe i reflects\n\
         // e_i = weight_i * opacity_i. A lobe index returns that lobe with\n\
         // radiance weight e_i / O; UINT_MAX picks by `u` in [0,1) with probability\n\
         // e_i / sum e and weight sum e / O. `count` is the number of lobes.\n\
         public {lobe_return} vork_lobe(__constref Blend s, uint lobe, float u, out float weight, out uint count) {{\n    \
         count=s.count; float e[{capacity}]; float total=0.; float transparency=0.;\n    \
         for (uint i=0u;i<{capacity}u;i++) {{\n        \
         let w=i<s.count ? max(s.weights[i],0.) : 0.; let o=saturate({opacity});\n        \
         e[i]=w*o; total+=e[i]; transparency+=w*(1.-o);\n    }}\n    \
         let opacity=1.-min(transparency,1.);\n    \
         var pick=min(lobe,{last}u);\n    \
         if (lobe==0xffffffffu) {{\n        \
         pick=0u; var sum=0.; let target=u*total;\n        \
         for (uint i=0u;i<{capacity}u;i++) {{ if (e[i]>0.) {{ pick=i; sum+=e[i]; if (sum>target) {{ break; }} }} }}\n        \
         weight=opacity>0. ? total/opacity : 0.;\n    \
         }} else {{ weight=opacity>0. ? e[pick]/opacity : 0.; }}\n    \
         {selected} {set_opacity} return c;\n}}\n\
         public {lobe_return} vork_checked_lobe(__constref Blend s, uint capacity, uint lobe, float u, out float weight, out uint count) {{\n\
         if (s.count>capacity) {{ vork_blend_overflow(s.count,capacity); weight=0.; count=1u; {lobe_return} c={{}}; return c; }}\n\
         return vork_lobe(s,lobe,u,weight,count); }}\n",
        overflow = u32::MAX,
        last = capacity - 1,
        layer_field = if layer_effects {" internal float3 layers; internal float weighted_ior; internal float wavelength; internal float emission;"} else {""},
        layer_init = if layer_effects {format!(" s.layers={}; s.weighted_ior=vork_closure_weighted_ior(c); s.wavelength=vork_closure_wavelength(c); s.emission=vork_closure_emission(c);",format!("float3({},{},{})",square("coat_weight", "c.coat_weight"),square("fuzz_weight", "c.fuzz_weight"), if closure.iter().any(|f| f.name == "specular_color") {"(c.specular_color.x-1.)*(c.specular_color.x-1.)+(c.specular_color.y-1.)*(c.specular_color.y-1.)+(c.specular_color.z-1.)*(c.specular_color.z-1.)"} else {"0."}))}else{String::new()},
        layer_sum = if layer_effects {"a.layers=float3(a.layers.x+b.layers.x,a.layers.y+b.layers.y,a.layers.z+b.layers.z); a.weighted_ior+=b.weighted_ior; a.wavelength+=b.wavelength+vork_effect_wavelength(w); a.emission+=b.emission*w*w;"} else {""},
        layer_reader = if layer_effects {format!("public float vork_closure_weighted_ior(__constref Closure c) {{return {weighted_ior};}}\npublic float vork_blend_weighted_ior(__constref Blend s) {{return s.weighted_ior;}}\npublic float vork_closure_emission(__constref Closure c) {{return {emission};}}\npublic float vork_blend_emission(__constref Blend s) {{return s.emission;}}\npublic float3 vork_blend_layers(__constref Blend s) {{ return s.layers; }}\npublic float {WAVELENGTH_EFFECT}(float v) {{ __target_switch {{case wgsl: __intrinsic_asm \"{WAVELENGTH_EFFECT}($0)\"; }} }}\npublic float vork_closure_wavelength(__constref Closure c) {{return {wavelength};}}\npublic float vork_blend_wavelength(__constref Blend s) {{return s.wavelength;}}", emission=emission_effect(closure), weighted_ior=square("specular_weight","c.specular_weight-1.")) } else {String::new()},
        effect_fields = "internal float4 effects; internal float4 response;",
        effect_init = format!("s.effects=float4({},{},{},{}); s.response=float4({},{},{},{});", square("subsurface_weight", "c.subsurface_weight"), square("geometry_opacity", "1.-c.geometry_opacity"), square("transmission_weight", "c.transmission_weight"), spatial_subsurface(closure), format!("{}+{}",square("specular_roughness_anisotropy", "c.specular_roughness_anisotropy"),square("coat_roughness_anisotropy", "c.coat_weight*c.coat_roughness_anisotropy")), square("thin_film_weight", "c.thin_film_weight"), dispersion, square("coat_roughness_anisotropy", "c.coat_weight*c.coat_roughness_anisotropy")),
        effect_sum = "a.effects=float4(a.effects.x+b.effects.x,a.effects.y+b.effects.y,a.effects.z+b.effects.z,a.effects.w+b.effects.w); a.response=float4(a.response.x+b.response.x,a.response.y+b.response.y,a.response.z+b.response.z,a.response.w+b.response.w);",
        effect_readers = "public float4 vork_blend_response(__constref Blend s) { return s.response; }\npublic float4 vork_blend_effects(__constref Blend s) { return s.effects; }",
    )
}

// A native Blend is a pure expression DAG. Re-evaluating it assigns the same
// postorder node IDs, even when locals are reused. Walk from the root to one
// requested leaf by replaying that expression; retain only scalar summaries and
// the chosen Closure. No lobe-sized private arrays or aggregate copies reach a
// driver, and neither lobe count nor expression depth needs a storage ceiling.
fn replay_module(closure: &[Field]) -> String {
    let zero=closure.iter().map(|f|format!("{}(0)",f.ty.slang())).collect::<Vec<_>>().join(",");
    let opacity=if closure.iter().any(|f|f.name=="geometry_opacity") {"saturate(c.geometry_opacity)"} else {"1."};
    let set_opacity=if closure.iter().any(|f|f.name=="geometry_opacity") {"c.geometry_opacity=opacity;"} else {""};
    format!(r#"
public struct BlendLobe {{ public uint leaf=0xffffffffu; public float geometry_opacity; }};
public struct Blend {{ internal uint node; internal uint count; internal float energy; internal float transparency; }};
static uint vork_node_cursor;
static uint vork_target_node;
static uint vork_target_index;
static float vork_target_energy;
static float vork_leaf_scale;
static float vork_leaf_opacity;
static float vork_root_opacity;
static float vork_root_energy;
static bool vork_random_leaf;
static bool vork_found_leaf;
static Closure vork_selected_closure;
public void vork_begin_blend(uint node) {{
    vork_node_cursor=0u;vork_target_node=node;vork_found_leaf=node==0xffffffffu;
    vork_selected_closure={{{zero}}};
}}
public bool vork_blend_done() {{return vork_found_leaf;}}
public void vork_continue_blend() {{vork_node_cursor=0u;}}
public Blend blend(__constref Closure c) {{
    Blend s;s.node=vork_node_cursor++;s.count=1u;s.energy={opacity};s.transparency=1.-s.energy;
    if(s.node==vork_target_node) {{vork_selected_closure=c;vork_leaf_opacity=s.energy;vork_found_leaf=true;}}
    return s;
}}
Blend blend_sum(Blend a,Blend b,float wa=1.,float wb=1.) {{
    Blend s;s.node=vork_node_cursor++;s.count=a.count+b.count;
    s.energy=a.energy*wa+b.energy*wb;s.transparency=a.transparency*wa+b.transparency*wb;
    if(s.node==vork_target_node) {{
        bool left=vork_random_leaf ? (b.energy*wb<=0. || (a.energy*wa>0. && vork_target_energy<a.energy*wa)) : vork_target_index<a.count;
        if(left) {{
            vork_target_node=a.node;vork_target_energy=wa>0. ? vork_target_energy/wa : 0.;vork_leaf_scale*=wa;
        }} else {{
            vork_target_node=b.node;vork_target_index-=a.count;
            vork_target_energy=wb>0. ? (vork_target_energy-a.energy*wa)/wb : 0.;vork_leaf_scale*=wb;
        }}
    }}
    return s;
}}
public Blend add(Blend a,Blend b) {{return blend_sum(a,b);}}
public Blend add(Closure a,Blend b) {{return add(blend(a),b);}}
public Blend add(Blend a,Closure b) {{return add(a,blend(b));}}
public Blend add(Closure a,Closure b) {{return add(blend(a),blend(b));}}
public Blend blend(Blend a,Blend b,float weight) {{let f=saturate(weight);return blend_sum(a,b,1.-f,f);}}
public Blend blend(Closure a,Blend b,float weight) {{return blend(blend(a),b,weight);}}
public Blend blend(Blend a,Closure b,float weight) {{return blend(a,blend(b),weight);}}
public Blend blend(Closure a,Closure b,float weight) {{return blend(blend(a),blend(b),weight);}}
public uint vork_blend_count(Blend s) {{return s.count;}}
public BlendLobe vork_lobe(Blend s,uint lobe,float u,out float weight,out uint count) {{
    count=s.count;vork_random_leaf=lobe==0xffffffffu;vork_target_index=lobe;
    vork_root_opacity=1.-min(s.transparency,1.);vork_root_energy=s.energy;
    vork_target_energy=u*s.energy;vork_leaf_scale=!vork_random_leaf && lobe>=s.count ? 0. : 1.;
    if(!vork_random_leaf && lobe>=s.count) vork_target_index=0u;
    vork_leaf_opacity=0.;weight=0.;
    BlendLobe selected;selected.leaf=s.node;selected.geometry_opacity=vork_root_opacity;return selected;
}}
public float vork_blend_weight() {{return vork_root_opacity>0. ? (vork_random_leaf ? vork_root_energy : vork_leaf_scale*vork_leaf_opacity)/vork_root_opacity : 0.;}}
public Closure vork_replay_closure(float opacity) {{var c=vork_selected_closure;{set_opacity}return c;}}
"#)
}

/// Default storage when the compiler cannot prove a Blend's lobe bound.
pub const BLEND_LOBES: u32 = 4;
/// Specialize the generated Blend storage without rewriting authored source.
pub(crate) fn with_capacity(module: &str, closure: &[Field], capacity: u32) -> String {
    module.replace(&blend_module(closure), &blend_module_with_capacity(closure, capacity, false))
}

/// Native Blend values carry scalar expression summaries. Replaying the pure
/// expression walks to one selected leaf without lobe-sized storage. Node IDs
/// preserve reuse of a Blend local (including adding a local to itself).
pub(crate) fn with_replay(module: &str, closure: &[Field]) -> String {
    module.replace(&blend_module(closure), &replay_module(closure))
}

/// Extra abstract-effect state belongs to the probe, never runtime Blend values.
pub(crate) fn with_layer_effects(module: &str, closure: &[Field]) -> String {
    module.replace(&blend_module(closure), &blend_module_with_capacity(closure, BLEND_LOBES, true))
}

#[cfg(test)]
mod replay_tests {
    use super::*;

    #[test]
    fn native_blends_have_no_lobe_sized_storage() {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let module=abi.native_module();
        let fields=module.split("public struct Blend {").nth(1).unwrap().split("};").next().unwrap();
        assert!(!fields.contains('['));
        assert!(fields.contains("uint node") && fields.contains("uint count"));
        assert!(!fields.contains("Closure"));
    }

    #[test]
    fn native_replay_has_no_effect_accumulators_or_closure_adapters() {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        {
            let module = abi.native_module();
            assert!(module.contains("static Closure vork_selected_closure;"));
            assert!(module.contains("vork_selected_closure={float(0),float3(0),"));
            assert!(!module.contains("vork_selected_closure={};"));
            for redundant in ["SelectedLobe", "blend_pack(", "blend_closure(",
                "internal float4 effects;", "internal float4 response;",
                "vork_blend_effects(", "vork_blend_response(", "vork_blend_wavelength("] {
                assert!(!module.contains(redundant), "runtime retains {redundant}");
            }
        }
        let probe = with_layer_effects(&abi.module(), &abi.closure);
        for required in ["internal float4 effects;", "internal float4 response;",
            "vork_blend_effects(", "vork_blend_response(", "vork_blend_layers("] {
            assert!(probe.contains(required), "effect probe lost {required}");
        }
    }

    #[test]
    fn native_replay_initializes_each_phase_before_reading_scratch() {
        // native_blend::separate relies on this generator contract: no replay
        // state survives a material call. Keep the phase boundaries explicit.
        let module = replay_module(&openpbr_fields());
        let begin = module.split("public void vork_begin_blend(uint node) {").nth(1).unwrap()
            .split("\n}").next().unwrap();
        let zero = openpbr_fields().iter().map(|f| format!("{}(0)", f.ty.slang()))
            .collect::<Vec<_>>().join(",");
        assert_eq!(begin.trim(), format!(
            "vork_node_cursor=0u;vork_target_node=node;vork_found_leaf=node==0xffffffffu;\n    vork_selected_closure={{{zero}}};"));
        // During the metadata pass the sentinel target cannot match a node.
        // Reads of selection state must stay inside that target guard.
        let sum = module.split("Blend blend_sum(").nth(1).unwrap();
        assert!(sum.contains("if(s.node==vork_target_node) {\n        bool left=vork_random_leaf"));
        let lobe = module.split("public BlendLobe vork_lobe(").nth(1).unwrap()
            .split("\n}").next().unwrap().split_once(" {\n").unwrap().1;
        assert!(lobe.starts_with(concat!(
            "    count=s.count;vork_random_leaf=lobe==0xffffffffu;vork_target_index=lobe;\n",
            "    vork_root_opacity=1.-min(s.transparency,1.);vork_root_energy=s.energy;\n",
            "    vork_target_energy=u*s.energy;vork_leaf_scale=!vork_random_leaf && lobe>=s.count ? 0. : 1.;\n",
            "    if(!vork_random_leaf && lobe>=s.count) vork_target_index=0u;\n",
            "    vork_leaf_opacity=0.;weight=0.;\n")));
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn native_blend_selection_and_replay_share_one_outlined_wrapper() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let program = Compiler::discover().compile_surface(&abi,
            "struct Inputs {}; Blend surface(ShadingContext ctx, Inputs inputs) { return add(unlit(float3(.2)), unlit(float3(.4))); }",
            Path::new("outlined-blend.slang"), "outlinedblend")?;
        let (outlined, ordinary) = program.slang.split_once("#else\npublic void vork_surface_outlinedblend(").expect("outlined and ordinary native boundaries");
        assert!(outlined.contains("[noinline]\nvork_surface_outlinedblend_result vork_surface_outlinedblend("), "{}",program.slang);
        assert_eq!(outlined.matches("surface(ctx, inputs)").count(), 2);
        assert_eq!(ordinary.matches("surface(ctx, inputs)").count(), 2);
        // Pin the ordering, including the first surface evaluation: moving the
        // reset after it would invalidate per-wrapper scratch localization.
        assert!(program.slang.contains(concat!(
            "vork_begin_blend(0xffffffffu);\n",
            "    let selected=vork_lobe(surface(ctx, inputs), lobe, u, weight, count);\n",
            "    vork_begin_blend(selected.leaf);\n",
            "    [loop] while(!vork_blend_done()) {surface(ctx, inputs);vork_continue_blend();}\n",
            "    weight=vork_blend_weight();\n",
            "    Closure c=vork_replay_closure(selected.geometry_opacity);")));
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn native_blend_replay_has_no_closure_arrays() -> Result<()> {
        let fields = openpbr_fields();
        let compiler = Compiler::discover();
        for capacity in [4, 5, 6, 8, 65] {
            let mut source = String::from("public struct Closure {\n");
            for f in &fields {
                writeln!(source, "public {} = {};", f.slang(), f.default.as_deref().unwrap_or("0"))?;
            }
            source.push_str("};\nRWStructuredBuffer<float4> result;\nvoid vork_blend_overflow(uint n,uint cap) { result[1]=float4(n,cap,0,0); }\n");
            source.push_str(&with_replay(&blend_module(&fields), &fields));
            source.push_str(&format!("Blend surface(float x) {{ Closure c={{}};c.emission_color=float3(x);var s=blend(c);for(uint i=1u;i<{capacity}u;i++) {{c.emission_color+=1.;s=add(s,c);}}return s;}}\n"));
            source.push_str("[shader(\"compute\")][numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID) { vork_begin_blend(0xffffffffu); float w;uint n;let pick=vork_lobe(surface(float(id.x)),0xffffffffu,.7,w,n);vork_begin_blend(pick.leaf);while(!vork_blend_done()) {surface(float(id.x));vork_continue_blend();}w=vork_blend_weight();let c=vork_replay_closure(pick.geometry_opacity); result[0]=float4(c.emission_color*w,float(n)); }");
            let spirv=compiler.compile_modules(&[("probe",&source)],"probe",&[("main","compute")],&["-target","spirv","-O2"])?;
            assert!(!spirv.is_empty());
            let msl = compiler.compile_modules(&[("probe", &source)], "probe", &[("main", "compute")], &["-target", "metal", "-O2"])?;
            let msl = String::from_utf8(msl)?;
            assert!(!msl.contains("SelectedLobe"), "redundant replay closure type survived lowering");
            assert!(!msl.contains("array<Closure"), "closure array survived replay lowering");
            assert!(!msl.contains("array<Blend"), "native Blend must not retain lobe-sized private arrays");
            if let Ok(directory) = std::env::var("VORK_BLEND_EVIDENCE_DIR") {
                std::fs::create_dir_all(&directory)?;
                std::fs::write(Path::new(&directory).join(format!("replay-{capacity}.metal")), &msl)?;
            }
        }
        Ok(())
    }
}
