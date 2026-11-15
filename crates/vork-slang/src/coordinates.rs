//! Coordinate transforms supplied by the native scene, rather than duplicate
//! Blender socket values in every shading context. Object transforms take the
//! shading context; its renderer object index stays internal to `vork`.
pub(crate) fn module(native: bool) -> String {
    let transforms = [
        ("object_to_generated", "uint object_index, float3 position", "let bounds=objects[object_index].reserved; let minimum=float3(asfloat(material_words[bounds]),asfloat(material_words[bounds+1u]),asfloat(material_words[bounds+2u])); let extent=float3(asfloat(material_words[bounds+3u]),asfloat(material_words[bounds+4u]),asfloat(material_words[bounds+5u])); return (position-minimum)/select(extent!=float3(0.),extent,float3(1.));"),
        ("world_to_object", "uint object_index, float3 position", "return mul(float4(position,1.),transpose(object_normal(object_index))).xyz;"),
        ("object_direction_to_world", "uint object_index, float3 direction", "return normalize(mul(float4(direction,0.),object_model(object_index)).xyz);"),
        ("object_normal_to_world", "uint object_index, float3 normal", "return normalize(mul(float4(normal,0.),object_normal(object_index)).xyz);"),
        ("world_normal_to_object", "uint object_index, float3 normal", "return normalize(mul(float4(normal,0.),transpose(object_model(object_index))).xyz);"),
        ("world_to_camera", "float3 position", "return mul(float4(position,1.),frame.view).xyz;"),
        ("world_to_window", "float3 position", "let clip=mul(float4(position,1.),frame.vp); return float3(clip.xy/clip.w*.5+.5,0.);"),
    ];
    let mut output = String::new();
    for (name, arguments, body) in transforms {
        let object = arguments.starts_with("uint");
        let body = if native { body.to_owned() } else {
            let args = if object { "$0, $1" } else { "$0" };
            format!("__target_switch {{ case wgsl: __intrinsic_asm \"vork_{name}({args})\"; }}")
        };
        if object {
            let (_, value) = arguments.split_once(", ").unwrap();
            let value_name = value.rsplit(' ').next().unwrap();
            output.push_str(&format!("float3 {name}_of({arguments}) {{ {body} }}\n"));
            output.push_str(&format!("public float3 {name}(ShadingContext ctx, {value}) {{ return {name}_of(ctx.object_index, {value_name}); }}\n"));
        } else {
            output.push_str(&format!("public float3 {name}({arguments}) {{ {body} }}\n"));
        }
    }
    output
}

pub(crate) fn validation_stubs() -> String {
    let mut stubs = super::trace::validation_stubs().to_owned();
    stubs.push_str(
    "fn vork_object_to_generated(i:u32,p:vec3<f32>)->vec3<f32>{return p+vec3(f32(i));}\n\
     fn vork_world_to_object(i:u32,p:vec3<f32>)->vec3<f32>{return p+vec3(f32(i));}\n\
     fn vork_object_direction_to_world(i:u32,p:vec3<f32>)->vec3<f32>{return p+vec3(f32(i));}\n\
     fn vork_object_normal_to_world(i:u32,p:vec3<f32>)->vec3<f32>{return p+vec3(f32(i));}\n\
     fn vork_world_normal_to_object(i:u32,p:vec3<f32>)->vec3<f32>{return p+vec3(f32(i));}\n\
     fn vork_world_to_camera(p:vec3<f32>)->vec3<f32>{return p;}\n\
     fn vork_world_to_window(p:vec3<f32>)->vec3<f32>{return p;}\n");
    stubs
}
