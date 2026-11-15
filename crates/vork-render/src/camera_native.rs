//! Fixed camera kernel ABI shared by build-time Slang and runtime dispatch.
// One field list owns the upload struct and its Slang declaration.
// Explicit padding aligns the trailing uint4 in both constant-buffer layouts.
macro_rules! camera_config_fields {
    ($consumer:ident) => { $consumer! { materials:u32=>uint, material:u32=>uint, samples:u32=>uint, direct:u32=>uint, diffusion:u32=>uint, aovs:u32=>uint, capacity:u32=>uint, words:u32=>uint, ray_shadows:u32=>uint, lit:u32=>uint, screen:u32=>uint, flags:u32=>uint, transparent:u32=>uint, transmittance_offset:u32=>uint, fibre_light_count:u32=>uint, band:u32=>uint, receiver_record_offset:u32=>uint, seed:u32=>uint, transport_parts:u32=>uint, transport_part:u32=>uint, packet_pixels:u32=>uint, packet_first_pixel:u32=>uint, packet_pad1:u32=>uint, packet_pad2:u32=>uint, boundaries:[u32;4]=>uint4 } };
}
#[allow(unused_imports)] // Build-time generator expands the macro locally.
pub(crate) use camera_config_fields;

use std::fmt::Write;
/// Prebuilt visibility variants are an install-time cache, never an admission limit.
// Camera samples and retained-map intervals use ordered arithmetic in shipped
// and runtime variants. Metal carries this contract into its native compiler.
pub const COMPILER_FLAGS: &[&str] = &["-fp-mode", "precise"];
pub const CONFIG_BINDING:u32=11;
/// Shared camera cache covers triangle-only scenes; fields compile at install.
#[allow(dead_code)] // Used by build_slang.rs.
pub fn defines(rays:bool)->Vec<String> {
    let mut defines=vec!["VORK_NO_IMPLICIT".to_owned(),"VORK_CAMERA_SPLIT".to_owned(),"VORK_ATTRIBUTES=8".into(),
        "VORK_PROBES".into(),"VORK_PROBE_BINDING=9".into(),"VORK_PROBE_IRRADIANCE_REGISTER=t16".into(),
        "VORK_PROBE_VISIBILITY_REGISTER=t17".into(),"VORK_PROBE_GRID_REGISTER=u18".into(),
        "VORK_FIBRE".into(),"VORK_FIBRE_BINDING=4".into()];
    if rays {defines.extend(["VORK_RAY_BINDINGS".into(),"VORK_RAY_REFRACTION".into()]);}
    defines
}
/// Metal evaluation kernels compile with every function inlined. Left to its
/// own heuristics, the Metal compiler keeps calls once a helper has more than
/// one call site, and on M2 Ultra such a call in a transmissive evaluation
/// aborted the bin with Impacting Interactivity or never finished, at any
/// resolution, whether or not the code on that call path ran. The Metal
/// mechanism is unproven: forcing inlining flips it, and API validation reports
/// nothing. Slang emits MSL source, so the contract travels as source directives
/// around its functions, leaving the kernel entry itself out.
pub fn inline_metal_functions(code:&[u8])->Vec<u8> {
    const OPEN:&[u8]=b"#pragma clang attribute push(__attribute__((always_inline)),apply_to=function)\n";
    const CLOSE:&[u8]=b"#pragma clang attribute pop\n";
    const NAMESPACE:&[u8]=b"using namespace metal;";
    const KERNEL:&[u8]=b"[[kernel]]";
    let find=|needle:&[u8]|code.windows(needle.len()).position(|window|window==needle);
    let start=find(NAMESPACE).expect("Slang MSL declares the metal namespace")+NAMESPACE.len();
    // Slang writes native text files: Windows uses CRLF, Unix uses LF.
    // Insert after the complete line without changing any emitted bytes.
    let start=start+code[start..].iter().position(|&byte|byte==b'\n').expect("Slang MSL namespace ends with a newline")+1;
    let end=find(KERNEL).expect("Slang MSL declares a kernel");
    assert!(start<=end,"Slang MSL declares its kernel before the metal namespace");
    [&code[..start],OPEN,&code[start..end],CLOSE,&code[end..]].concat()
}

#[cfg(test)]
mod tests {
    #[test]
    fn metal_inlining_preserves_lf_and_crlf_source() {
        for newline in ["\n", "\r\n"] {
            let prefix=format!("#include <metal_stdlib>{newline}using namespace metal;{newline}");
            let helper=format!("float helper() {{ return 1.; }}{newline}");
            let kernel=format!("[[kernel]] void evaluate() {{ helper(); }}{newline}");
            let source=format!("{prefix}{helper}{kernel}");
            let expected=format!("{prefix}#pragma clang attribute push(__attribute__((always_inline)),apply_to=function)\n{helper}#pragma clang attribute pop\n{kernel}");
            assert_eq!(super::inline_metal_functions(source.as_bytes()),expected.as_bytes());
        }
    }
}
#[allow(dead_code)] // Used by build_slang.rs.
pub fn config_source()->String {
    let binding=CONFIG_BINDING;
    macro_rules! declare {
        ($($field:ident:$rust:ty=>$slang:ident),*) => {concat!("public struct CameraConfig {", $("public ",stringify!($slang)," ",stringify!($field),";",)* "};\n")};
    }
    format!("{}[[vk::binding({binding},4)]] public ConstantBuffer<CameraConfig> camera_config : register(b27);\n",camera_config_fields!(declare))
}
pub struct BufferTable {
    pub name:&'static str,
    pub element:&'static str,
    pub read_only:bool,
    pub binding:u32,
    pub resources:&'static [&'static str],
}
pub const TABLES:[BufferTable;3]=[
    BufferTable {name:"camera_read_words",element:"uint",read_only:true,binding:2,resources:&["slots","draw_table","object_draws","environment_images"]},
    BufferTable {name:"camera_write_words",element:"uint",read_only:false,binding:3,resources:&["reflection_guides","subsurface_records","evaluation_metadata","evaluation_bins","evaluation_allocator"]},
    BufferTable {name:"camera_write_vectors",element:"float4",read_only:false,binding:4,resources:&["aov_values","refraction_receivers"]},
];
#[allow(dead_code)] // Used by build_slang.rs.
pub fn evaluation_source(source:&str)->String {
    let mut source=source.to_owned();
    let mut declarations=String::new();
    for table in &TABLES {
        let (name,binding,names)=(table.name,table.binding,table.resources);
        let ty=format!("{}StructuredBuffer<{}>",if table.read_only {""} else {"RW"},table.element);
        let count=names.len();
        let register=19+binding;
        writeln!(declarations,"#ifdef VORK_METAL\nstruct {name}Table {{{ty} buffers[{count}];}};\n[[vk::binding({binding},3)]] ParameterBlock<{name}Table> {name} : register(b{register});\n#else\n[[vk::binding({binding},3)]] {ty} {name}[{count}];\n#endif").unwrap();
        for (index,resource) in names.iter().enumerate() {
            writeln!(declarations,"#ifdef VORK_METAL\n#define {resource} {name}.buffers[{index}]\n#else\n#define {resource} {name}[{index}]\n#endif").unwrap();
        }
    }
    declarations.push_str(r#"
#ifdef VORK_METAL
struct CameraReflectionFields {RWStructuredBuffer<uint> buffers[REG_VISIBILITY_REFLECTION_WORDS];};
[[vk::binding(5,3)]] ParameterBlock<CameraReflectionFields> camera_reflections : register(b24);
#define reflection_records camera_reflections.buffers
#else
[[vk::binding(5,3)]] RWStructuredBuffer<uint> camera_reflections[REG_VISIBILITY_REFLECTION_WORDS];
#define reflection_records camera_reflections
#endif
"#);
    source=format!("{declarations}\n{source}");
    let mut code=String::new();
    writeln!(code,"#ifdef VORK_METAL\nstruct CameraEvaluationPages {{RWStructuredBuffer<float4> buffers[];}};\n[[vk::binding(1,3)]] ParameterBlock<CameraEvaluationPages> evaluation_pages : register(b20);\n#else\n[[vk::binding(1,3)]] RWStructuredBuffer<float4> evaluation_pages[] : register(u12);\n#endif").unwrap();
    code.push_str(r#"
float4 evaluation_load(uint page,uint at) {
#ifdef VORK_METAL
    return evaluation_pages.buffers[page][at];
#else
    return evaluation_pages[NonUniformResourceIndex(page)][at];
#endif
}
void evaluation_store(uint page,uint at,float4 value) {
#ifdef VORK_METAL
    evaluation_pages.buffers[page][at]=value;
#else
    evaluation_pages[NonUniformResourceIndex(page)][at]=value;
#endif
}
"#);
    for write in [false,true] {
        let (result,name,arg)=if write {("void","write",",EvaluatedSlot e")} else {("EvaluatedSlot","read","")};
        writeln!(code,"{result} evaluation_{name}<V:IVariant>(uint index{arg}) {{uint page=index/camera_config.capacity;uint at=(index%camera_config.capacity)*camera_config.words;\n").unwrap();
        if !write {code.push_str("EvaluatedSlot e=(EvaluatedSlot)0;\n");}

            let fields=[("response.color","e.response.roughness"),("response.baseline.diffuse","e.ray_distance"),("response.reflected","e.response.reflection.data[11].x"),("response.diffuse","0."),("response.normal","V.has_reflections ? e.response.reflection.data[11].y : 0."),("world","e.barycentrics.x"),("geometric","e.barycentrics.y"),("alpha.xyz","e.alpha.w"),("diffusion.light","0."),("diffusion.ambient","0."),("diffusion.radius","0."),("diffusion.albedo","0."),("diffusion.light_albedo","0."),("diffusion.ambient_albedo","0.")];
            for (i,(field,w)) in fields.iter().enumerate() {
                if i==8 {code.push_str("if(VORK_CAMERA_DIFFUSION!=0u) {\n");}
                let gate=match i {2=>Some("V.has_reflections"),3=>Some("V.has_indirect_controls"),4=>Some("V.has_camera_receivers || V.has_reflections"),5|6=>Some("V.has_camera_receivers || V.subsurface || camera_config.diffusion!=0u"),7=>Some("VORK_CAMERA_TRANSPARENT!=0"),_=>None};
                if let Some(gate)=gate {writeln!(code,"if({gate}) {{").unwrap();}
                if write {writeln!(code,"evaluation_store(page,at+{i}u,float4(e.{field},{w}));").unwrap();}
                else {
                    writeln!(code,"let v{i}=evaluation_load(page,at+{i}u);e.{field}=v{i}.xyz;").unwrap();
                    match i {0=>writeln!(code,"e.response.roughness=v{i}.w;"),1=>writeln!(code,"e.ray_distance=v{i}.w;"),2=>writeln!(code,"e.response.reflection.data[11].x=v{i}.w;"),4=>writeln!(code,"if(V.has_reflections) {{e.response.reflection.data[11].y=v{i}.w;}}"),5=>writeln!(code,"e.barycentrics.x=v{i}.w;"),6=>writeln!(code,"e.barycentrics.y=v{i}.w;"),7=>writeln!(code,"e.alpha.w=v{i}.w;"),_=>Ok(())}.unwrap();
                }
                if gate.is_some() {code.push_str("}\n");}
            }
            code.push_str("}\n");
            code.push_str("if(V.has_reflections) {for(uint i=0u;i<REG_REFLECTION_EVALUATION_VECTORS;i++) {let offset=8u+VORK_CAMERA_DIFFUSION*6u+VORK_CAMERA_AOVS+i;\n");
            if write {code.push_str("var value=e.response.reflection.data[i];if(i==9u) {value.w=e.response.reflection.data[11].z;}if(i==10u) {value.w=e.response.reflection.data[11].w;}evaluation_store(page,at+offset,value);\n");} else {code.push_str("e.response.reflection.data[i]=evaluation_load(page,at+offset);\n");}
            code.push_str("}\n");
            if !write {code.push_str("e.response.reflection.data[11].zw=float2(e.response.reflection.data[9].w,e.response.reflection.data[10].w);e.response.reflection.data[9].w=0.;e.response.reflection.data[10].w=0.;\n");}
            code.push_str("}\n");
            writeln!(code,"if(VORK_AOV_CHANNELS!=0u) for(uint a=0u;a<VORK_CAMERA_AOVS;a++) {{let offset=8u+VORK_CAMERA_DIFFUSION*6u+a;").unwrap();
            if write {writeln!(code,"evaluation_store(page,at+offset,e.aovs[a]);").unwrap();} else {writeln!(code,"e.aovs[a]=evaluation_load(page,at+offset);").unwrap();}
            code.push_str("}\n");
        code.push_str(if write {"}\n"} else {"return e;}\n"});
    }
    source.replace("// EVALUATION_STORAGE",&code)
}

/// Shipped camera entries: classification, composition and generic table evaluators.
#[allow(dead_code)] // Build-time entry generation; runtime uses the embedded entries.
pub fn entry_source(stage:&str)->String {
    let body=match stage {
        "evaluate"|"evaluate_opaque"=>"evaluate_material<CameraPolicy,NoPrograms>(gid);",
        "index"=>"index_evaluation_slots(gid,lane);",
        "classify"=>"classify_evaluations(gid,lane,false);",
        "prefix"=>"if(all(gid==uint3(0u))) {prefix_evaluations();}",
        "scatter"=>"classify_evaluations(gid,lane,true);",
        "combine"=>"if(camera_config.screen!=0u && camera_config.lit!=0u) {composite_pixel<CameraPolicy,NoPrograms,CameraTransmittance,3>(gid);} else {shade_pixel<CameraPolicy,NoPrograms,CameraTransmittance,CameraDiffusion,3>(gid);}",
        _=>panic!("unknown camera stage"),
    };
    let threads=if matches!(stage,"evaluate"|"evaluate_opaque") {"64,1,1"} else {"8,8,1"};
    let lane=if matches!(stage,"index"|"classify"|"scatter") {",uint lane:SV_GroupIndex"} else {""};
    let stage=if stage=="evaluate_opaque" {"evaluate"} else {stage};
    format!("import camera_config;\nimport surface;\nimport decoupled;\n[shader(\"compute\")][numthreads({threads})] void camera_{stage}(uint3 gid:SV_DispatchThreadID{lane}) {{{body}}}\n")
}
