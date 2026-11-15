use std::{collections::BTreeMap, fs, path::Path};
use vork_slang::{Abi, Compiler, Field, Type};

fn abi() -> Abi {
    Abi { aov_count: 3,
        context: Abi::spec_context(),
        closure: vec![
            Field::new("base_color", Type::Vector(3), Some("float3(0.8)")),
            Field::new("specular_roughness", Type::Float, Some("0.3")),
            Field::new("coat_ior", Type::Float, Some("1.6")),
            Field::new("geometry_thin_walled", Type::Bool, None),
        ],
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain; VORK_SLANGC=/path/to/slangc cargo test -p vork-slang -- --ignored"]
fn compile_reflect_pack_and_diagnose() {
    let compiler = Compiler::discover();
    let abi = abi();
    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/stripes.slang"
    ));
    let source = fs::read_to_string(path).unwrap();
    let a = compiler
        .compile_surface(&abi, &source, path, "a1")
        .unwrap();
    let b = compiler
        .compile_surface(&abi, &source, path, "b2")
        .unwrap();
    assert!(
        a.dependencies.iter().any(|d| d.ends_with("helpers.slang")),
        "{:?}",
        a.dependencies
    );
    let names: Vec<_> = a
        .inputs
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.ty, f.offset))
        .collect();
    assert_eq!(
        names,
        [
            ("tint", Type::Vector(3), 1),
            ("roughness", Type::Float, 4),
            ("scale", Type::Float, 5),
            ("glossy", Type::Bool, 6),
            ("bands", Type::Int, 7),
            ("albedo", Type::Image, 8)
        ]
    );
    let words = a
        .inputs
        .pack(
            &BTreeMap::from([
                ("tint".into(), serde_json::json!([0.2, 0.4, 0.8])),
                ("glossy".into(), serde_json::json!(false)),
                ("albedo".into(), serde_json::json!("brick")),
            ]),
            |_, v| Ok(if v == "brick" { 7 } else { 0 }),
        )
        .unwrap();
    assert_eq!(words.len(), 9);
    assert_eq!(words[0], 0b101001);
    assert_eq!(words[2], 0.4f32.to_bits());
    assert_eq!(words[8], 7);
    assert!(
        a.inputs
            .pack(
                &BTreeMap::from([("nope".into(), serde_json::json!(1))]),
                |_, _| Ok(0)
            )
            .is_err()
    );
    assert!(
        a.inputs
            .pack(
                &BTreeMap::from([("tint".into(), serde_json::json!([1, 2]))]),
                |_, _| Ok(0)
            )
            .is_err()
    );

    // Two programs plus kernel glue must form one valid WGSL module.
    let kernel = format!(
        "{}@group(0) @binding(0) var<storage, read> words: array<u32>;\n\
         fn {}(i:u32)->u32 {{ return words[i]; }}\n\
         fn {}(slot:u32, uv:vec2<f32>, ddx:vec2<f32>, ddy:vec2<f32>)->vec4<f32> {{ return vec4(uv, f32(slot), 1.); }}\n{}\
         @fragment fn main()->@location(0) vec4<f32> {{\n\
             var c:ShadingContext; c.TEXCOORD_0=vec2(.5);\n\
             let ctx=vork_pack_context(c);\n\
             var r:array<vec4<f32>,2>; {}(ctx,0u,&r); let x=vork_unpack_closure(r,Closure());\n\
             var q:array<vec4<f32>,2>; {}(ctx,9u,&q); let y=vork_unpack_closure(q,Closure());\n\
             return vec4(x.base_color+y.base_color, x.coat_ior);\n}}\n{}\n{}",
        abi.structs_wgsl(),
        vork_slang::MATERIAL_WORD,
        vork_slang::SAMPLE_IMAGE,
        abi.kernel_wgsl(),
        a.surface,
        b.surface,
        a.wgsl,
        b.wgsl
    );
    let module = naga::front::wgsl::parse_str(&kernel)
        .unwrap_or_else(|e| panic!("{}\n{kernel}", e.emit_to_string(&kernel)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{e:?}\n{kernel}"));

    let path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/broken.slang"
    ));
    let error = compiler
        .compile_surface(&abi, &fs::read_to_string(path).unwrap(), path, "c3")
        .unwrap_err()
        .to_string();
    assert!(error.contains("broken.slang:3:"), "{error}");
}

#[test]
#[ignore = "requires pinned Slang toolchain; VORK_SLANGC=/path/to/slangc cargo test -p vork-slang -- --ignored"]
fn medium_mixed_stages_position_dependence_and_errors() {
    let compiler = Compiler::discover();
    let abi = abi();
    let path = Path::new("mixed.slang");
    let source = "struct Inputs {float density=0.1;}; Closure surface(ShadingContext ctx,Inputs inputs){Closure c={};return c;} MediumCoefficients medium(MediumContext ctx, Inputs inputs){ MediumCoefficients m;m.absorption=float3(inputs.density);return m;}";
    compiler
        .compile_surface(&abi, source, path, "surface")
        .unwrap();
    let point = compiler
        .compile_medium(&abi, source, path, "point")
        .unwrap();
    assert!(point.homogeneous,"{}",point.wgsl);
    assert_eq!(point.inputs.fields[0].name, "density");
    for (body, homogeneous) in [
        ("m.absorption=float3(inputs.density);", true),
        ("m.absorption=float3(inputs.density+ctx.time);", true),
        ("m.absorption=float3(inputs.density*exp(-ctx.position.y));", false),
        ("m.emission=float3(ctx.position.x);", false),
        ("m.scattering=float3(lerp(0.,1.,ctx.position.x));", false),
        ("float v=ctx.position.y/0.;m.absorption=float3(v*0.);", false),
        ("m.emission=abs(ctx.position);", false),
        ("m.absorption=float3(inputs.density>0 ? 1. : 2.);", true),
        ("m.absorption=float3(ctx.position.y>0 ? 1. : 2.);", false),
        ("m.scattering=abs(ctx.position);", false),
        ("m.anisotropy=ctx.position.z;", false),
        ("if(ctx.position.y>0) m.absorption=float3(2);", false),
        ("float unused=ctx.position.y; m.absorption=float3(inputs.density);", true),
    ] {
        let source=format!("struct Inputs {{float density=.1;}}; MediumCoefficients medium(MediumContext ctx,Inputs inputs) {{MediumCoefficients m;{body}return m;}}");
        let program=compiler.compile_medium(&abi,&source,path,"dependency").unwrap();
        assert_eq!(program.homogeneous, homogeneous, "{body}\n{}",program.wgsl);
    }
    let medium=include_str!("../../vork-scene/tests/fixtures/medium.slang");
    for source in [medium.to_owned(), format!("{medium} Closure surface(ShadingContext ctx, Inputs inputs) {{Closure c={{}};return c;}}")] {
    let fog=compiler.compile_medium(&abi,&source,Path::new("fog.slang"),"boundfog").unwrap();
    for (values,expected) in [(serde_json::json!({}),true),(serde_json::json!({"density":0.4,"height_falloff":0}),true),(serde_json::json!({"density":0.4,"height_falloff":0.7}),false)] {
        let words=fog.inputs.pack(&serde_json::from_value(values.clone()).unwrap(),|_,_|unreachable!()).unwrap();
        assert_eq!(fog.homogeneous_for_inputs(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>()).unwrap(),expected,"{values}");
    }
    }
    let overflow=compiler.compile_medium(&abi,"struct Inputs {uint flag=0;}; MediumCoefficients medium(MediumContext ctx,Inputs inputs) {MediumCoefficients m;m.absorption=float3((inputs.flag+1u==0u)?ctx.position.y:1.);return m;}",path,"overflow").unwrap();
    let words=overflow.inputs.pack(&serde_json::from_value(serde_json::json!({"flag":4294967295u32})).unwrap(),|_,_|unreachable!()).unwrap();
    assert!(!overflow.homogeneous_for_inputs(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>()).unwrap());
    let image = "struct Inputs {Image density_tex;}; MediumCoefficients medium(MediumContext ctx, Inputs inputs){ MediumCoefficients m;return m;}";
    let error = compiler
        .compile_medium(&abi, image, path, "image")
        .unwrap_err()
        .to_string();
    assert!(error.contains("medium image inputs"), "{error}");
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_blender_procedurals() {
    let source = r#"
import vork;
import blender.procedurals;
struct Inputs { float mode = 0.; };
Closure surface(ShadingContext ctx, Inputs inputs) {
    float3 p = ctx.POSITION;
    float n = noise(p);
    for(int d=1;d<=4;d++) for(int t=0;t<5;t++)
        n += noise(p,5.,2.5,0.5,2.,1.,true,NoiseType(t),d,0.7,0.5,1.);
    n += noise_color(p).g;
    n += white_noise(p,4,0.7).value + magic(p).factor + gabor(p).value;
    for(int d=1;d<=4;d++)
        n += voronoi(p,5.,1.,VoronoiFeature.Smooth_F1,DistanceMetric.Euclidean,1.,0.5,d,0.7,2.5).distance;
    n += gabor(p,5.,2.,0.5,3).intensity;
    n += wave(p) + voronoi(p).distance + brick(p).factor;
    n += voronoi(p,5.,1.,VoronoiFeature.Smooth_F1).distance;
    n += voronoi(p,5.,1.,VoronoiFeature.Distance_To_Edge).distance;
    n += voronoi(p,5.,1.,VoronoiFeature.N_Sphere_Radius).radius;
    Closure c = {}; c.base_color = float3(n); return c;
}"#;
    for home in ["blender", "std"] {
        let source = source.replace("import blender.procedurals;", &format!("import {home}.procedurals;"));
        Compiler::discover().compile_surface(&abi(), &source, Path::new("procedurals.slang"), "proc").unwrap();
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn environment_numeric_stage_and_std_sky() {
    let compiler = Compiler::discover();
    let abi = abi();
    let source = "struct Inputs {float3 color=float3(0.2);}; float3 environment(EnvironmentContext ctx, Inputs inputs){ float3 direction=ctx.direction; return inputs.color;}";
    let result = compiler
        .compile_environment(&abi, source, Path::new("constant.slang"), "world")
        .unwrap();
    assert_eq!(result.inputs.fields[0].name, "color");
    assert!(!result.animated);
    assert!(!result.spatial);
    for (body, spatial) in [
        ("return ctx.origin;",true),
        ("float3 p=ctx.origin; return float3(length(p));",true),
        ("if(ctx.origin.x>0.) return float3(1.); return float3(0.);",true),
        ("float3 p=ctx.origin; return ctx.direction;",false),
    ] {
        let text=format!("struct Inputs {{}}; float3 environment(EnvironmentContext ctx,Inputs inputs) {{{body}}}");
        assert_eq!(compiler.compile_environment(&abi,&text,Path::new("origin.slang"),"originprobe").unwrap().spatial,spatial,"{body}");
    }
    let animated = source.replace("return inputs.color;", "return inputs.color * (1. + ctx.time);");
    assert!(compiler.compile_environment(&abi, &animated, Path::new("animated.slang"), "animatedworld").unwrap().animated);
    for body in ["return float3(asuint(ctx.time) & 255u);", "return float3(sin(ctx.time));", "if(ctx.time>0.) return float3(1); return float3(0);", "float v=ctx.time; for(int n=0;n<2;n++) v+=1.; return float3(v);", "float v=0.; for(int n=0;n<int(ctx.time);n++) v+=1.; return float3(v);", "float v=0.; do {v+=1.;} while(v<ctx.time); return float3(v);"] {
        let source=format!("struct Inputs {{}}; float3 environment(EnvironmentContext ctx, Inputs inputs) {{{body}}}");
        assert!(compiler.compile_environment(&abi, &source, Path::new("time.slang"), "timeprobe").unwrap().animated, "{body}");
    }
    assert!(result.wgsl.contains(&result.entry));
    let source = vork_slang::module_source("std/environment_image").unwrap();
    let image=compiler.compile_environment(&abi,source,Path::new("image.slang"),"image").unwrap();
    assert_eq!(image.inputs.fields[0].ty,vork_slang::Type::Image);
    assert!(!image.animated && !image.spatial);
    let animated=source.replace("atan2(d.z,d.x)","atan2(d.z,d.x)+ctx.time");
    assert!(compiler.compile_environment(&abi,&animated,Path::new("image.slang"),"image").unwrap().animated);
    let source = vork_slang::module_source("std/sky").unwrap();
    let result = compiler
        .compile_environment(&abi, source, Path::new("std/sky.slang"), "sky")
        .unwrap();
    assert_eq!(result.inputs.fields.len(), 6);
    assert!(!result.animated, "std/sky must retain install-only evaluation");
    for (name, maximum) in [("altitude",100000.),
        ("air_density",1000.),("aerosol_density",1000.),("ozone_density",1000.)] {
        for value in [-1., maximum + 1.] {
            let values = [(name.to_owned(),serde_json::json!(value))].into_iter().collect();
            assert!(result.inputs.pack(&values,|_,_| unreachable!()).is_err(), "{name}: {value}");
        }
    }
    let words = result
        .inputs
        .pack(&BTreeMap::new(), |_, _| unreachable!())
        .unwrap();
    assert!(!words.is_empty());
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_std_and_blender_color() {
    let source = r#"
import vork;
import std.color;
import blender.color;
struct Inputs {};
Closure surface(ShadingContext ctx,Inputs inputs) {
    float3 c=ctx.POSITION;
    for(int mode=0;mode<19;mode++) c+=mix(c,float3(0.2),0.5,BlendMode(mode),true,true);
    RampStop stops[2]; stops[0].position=0.;stops[0].color=float4(0.);
    stops[1].position=1.;stops[1].color=float4(1.);
    for(int mode=0;mode<5;mode++) c+=color_ramp(ctx.TEXCOORD_0.x,stops,RampInterpolation(mode)).xyz;
    c+=color_ramp(ctx.TEXCOORD_0.y,stops,RampInterpolation.Linear,RampColorMode.HSV).xyz;
    CurvePoint points[2];
    points[0].position=float2(0.);points[0].left=float2(-1.);points[0].right=float2(0.3);
    points[1].position=float2(1.);points[1].left=float2(0.7);points[1].right=float2(2.);
    c+=float_curve(c.x,points);
    c+=blackbody()+wavelength();
    Closure result={};result.base_color=c;return result;
}"#;
    for imports in ["import std.color;", "import blender.color;", "import std.color; import blender.color;"] {
        let source = source.replace("import std.color;\nimport blender.color;", imports);
        Compiler::discover().compile_surface(&abi(),&source,Path::new("color.slang"),"color").unwrap();
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_blender_white_balance_from_authored_module() {
    let source = r#"
import vork;
import blender.white_balance;
struct Inputs {};
Closure surface(ShadingContext ctx, Inputs inputs) {
    WhiteBalanceInputs white; white.temperature=4300.; white.tint=-12.;
    Closure result={}; result.base_color=white_balance(float3(.4),white); return result;
}"#;
    for home in ["blender", "std"] {
        let source = source.replace("import blender.white_balance;", &format!("import {home}.white_balance;"));
        Compiler::discover().compile_surface(&abi(),&source,Path::new("balance.slang"),"balance").unwrap();
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_blender_bump() {
    let source = r#"
import vork;
import blender.vector;
struct Inputs {};
struct Height:BumpHeight {
    float evaluate(float2 pixel_offset) { return sin(pixel_offset.x)+cos(pixel_offset.y); }
}
Closure surface(ShadingContext ctx,Inputs inputs) {
    float3 p=ctx.POSITION;
    p+=bump(ctx.NORMAL,float3(1.,0.,0.),float3(0.,1.,0.),0.1,0.2);
    Height height={};
    p+=bump(height,ctx.NORMAL,float3(1.,0.,0.),float3(0.,1.,0.));
    Closure result={};result.base_color=p;return result;
}"#;
    for home in ["blender", "std"] {
        let source = source.replace("import blender.vector;", &format!("import {home}.vector;"));
        Compiler::discover().compile_surface(&abi(),&source,Path::new("vector.slang"),"vector").unwrap();
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn shading_context_named_fields_compile() {
    let source = r#"
import vork;
struct Inputs {};
Closure surface(ShadingContext ctx, Inputs inputs) {
    float3 direction = ctx.NORMAL + ctx.true_normal + ctx.incoming + ctx.TANGENT;
    float2 coordinates = ctx.TEXCOORD_0 + ctx.duv_dx[0] + ctx.duv_dy[0];
    Closure c = {};
    c.base_color = ctx.POSITION + direction + float3(coordinates, ctx.time);
    c.specular_roughness = float(ctx.object_id);
    return c;
}
"#;
    Compiler::discover()
        .compile_surface(&abi(), source, Path::new("context.slang"), "context")
        .unwrap();
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_texture_coordinate_bump() {
    let source = r#"
import vork;
import blender.texture_coordinate;
import blender.vector;
struct Inputs {};
struct Height : BumpHeight {
    ShadingContext context;
    float evaluate(float2 pixel_offset) {
        let c=texture_coordinate(context,pixel_offset);
        return sin(c.Generated.x*20.)+cos(c.Object.y*7.)+c.UV.x;
    }
}
Closure surface(ShadingContext ctx, Inputs inputs) {
    let c=texture_coordinate(ctx);
    Height height; height.context=ctx;
    Closure result={};
    result.base_color=c.Generated+c.Normal+c.UV+c.Object+c.Camera+c.Window+c.Reflection;
    result.geometry_normal=bump(height,ctx.NORMAL,ctx.position_dx,ctx.position_dy);
    return result;
}
"#;
    let mut abi=abi();
    abi.closure.push(Field::new("geometry_normal",Type::Vector(3),None));
    Compiler::discover().compile_surface(&abi,source,Path::new("texture-coordinate.slang"),"coordinates").unwrap();
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_medium_grid_reads() {
    let source=r#"
import vork;
import std.grid;
struct Inputs { FloatGrid density; VectorGrid color; };
MediumCoefficients medium(MediumContext ctx, Inputs inputs) { float3 position=ctx.position;
    MediumCoefficients result={};
    result.absorption=float3(sample_grid(inputs.density,position));
    result.scattering=sample_grid(inputs.color,position);
    return result;
}
"#;
    let program=Compiler::discover().compile_medium(&abi(),source,Path::new("grid.slang"),"grid").unwrap();
    assert_eq!(program.inputs.fields[0].ty,Type::FloatGrid);
    assert_eq!(program.inputs.fields[1].ty,Type::VectorGrid);
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_named_aov_writes() {
    let source=r#"
import vork;
struct Inputs { Aov heat; Aov tint; };
Closure surface(ShadingContext ctx,Inputs inputs) {Closure c={};return c;}
void aov(ShadingContext ctx,Inputs inputs,inout AovOutputs output) {
    write_aov(output,inputs.heat,ctx.POSITION.x);
    write_aov(output,inputs.tint,float3(ctx.TEXCOORD_0,.25));
}
"#;
    let program=Compiler::discover().compile_surface(&abi(),source,Path::new("aov.slang"),"aov").unwrap();
    assert_eq!(program.inputs.fields[0].ty,Type::Aov);
    assert!(program.aov.is_some());
    assert!(program.slang.contains("public void vork_aov_aov"));
    let hair_source = source.replace(
        "Closure surface(ShadingContext ctx,Inputs inputs) {Closure c={};return c;}",
        "Hair surface(ShadingContext ctx,Inputs inputs) {Hair h={};return h;}",
    );
    let hair = Compiler::discover().compile_surface(&abi(), &hair_source, Path::new("hair_aov.slang"), "hairaov").unwrap();
    assert!(hair.capabilities.hair);
    assert!(hair.wgsl.contains(&format!("fn {}(", hair.aov.as_ref().unwrap())));
    assert!(hair.slang.contains("public void vork_aov_hairaov"));
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn bundled_hair_accepts_every_documented_socket() {
    let source = vork_slang::module_source("std/hair").unwrap();
    let program = Compiler::discover()
        .compile_surface(&abi(), source, Path::new("std/hair.slang"), "hairsockets")
        .unwrap();
    assert!(program.capabilities.hair);
    let sockets = ["parametrization", "color", "melanin", "melanin_redness", "tint",
        "absorption_coefficient", "roughness", "radial_roughness", "coat", "ior", "offset",
        "random_roughness", "random_color", "random"];
    let reflected: std::collections::BTreeSet<_> = program
        .inputs
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(reflected, sockets.into_iter().collect());
    let inputs = BTreeMap::new();
    program.inputs.pack(&inputs, |_, _| unreachable!()).unwrap();
    for mode in ["color", "melanin", "absorption"] {
        let mut selected = inputs.clone();
        selected.insert("parametrization".into(), serde_json::json!(mode));
        program
            .inputs
            .pack(&selected, |_, _| unreachable!())
            .unwrap();
    }
    let mut invalid = inputs;
    invalid.insert("parametrization".into(), serde_json::json!("pigment"));
    assert!(
        program
            .inputs
            .pack(&invalid, |_, _| unreachable!())
            .is_err()
    );
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn scene_coordinate_queries_retain_runtime_material_effects() {
    let mut abi = abi();
    abi.closure.push(Field::new("transmission_weight", Type::Float, Some("0.")));
    let source = r#"
import vork;
struct Inputs {};
Closure surface(ShadingContext ctx, Inputs inputs) {
    Closure result = {};
    result.transmission_weight = world_to_camera(float3(0.)).x;
    return result;
}
"#;
    let program = Compiler::discover().compile_surface(&abi,source,Path::new("scene-coordinate.slang"),"coordinateEffects").unwrap();
    assert!(program.capabilities.transmission, "a constant position still depends on the installed camera");
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn typed_attribute_inputs_reflect_and_read_footprints() {
    let source = "struct Inputs {Attribute<float> a;Attribute<float2> b;Attribute<float3> c;Attribute<float4> d;}; Closure surface(ShadingContext ctx, Inputs i) {Closure c={};c.base_color=read_attribute(i.c,ctx).value+float3(read_attribute(i.b,ctx).dx,read_attribute(i.a,ctx).dy)+read_attribute(i.d,ctx).value.xyz;return c;}";
    let program=Compiler::discover().compile_surface(&abi(), source, Path::new("attributes.slang"), "attrs").unwrap();
    assert_eq!(program.inputs.fields.iter().map(|f|f.ty).collect::<Vec<_>>(), (1..=4).map(Type::Attribute).collect::<Vec<_>>());
    assert!(program.wgsl.contains("vork_attribute"));
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn material_displacement_is_optional_vector_and_shares_inputs() -> anyhow::Result<()> {
    let compiler = Compiler::discover();
    let abi = abi();
    let base = "import vork; struct Inputs { float scale=0.2; Image height; }; Closure surface(ShadingContext ctx, Inputs i) { return {}; }";
    let origin = Path::new("displacement.slang");
    let plain = compiler.compile_surface(&abi, base, origin, "dp0")?;
    assert!(!plain.capabilities.displacement);
    let source = format!("{base}\nfloat3 displace(DisplacementContext ctx, Inputs i) {{return ctx.NORMAL * i.scale + sample(i.height,ctx.POSITION.xy,ctx).xyz;}}");
    let program = compiler.compile_surface(&abi, &source, origin, "dp1")?;
    assert!(program.capabilities.displacement);
    assert_eq!(program.inputs.words, plain.inputs.words);
    assert!(program.slang.contains("public float3 vork_displace_dp1("));
    for bad in [
        "float displace(DisplacementContext ctx, Inputs i) {return i.scale;}",
        "int3 displace(DisplacementContext ctx, Inputs i) {return int3(1);}",
        "float2 displace(DisplacementContext ctx, Inputs i) {return float2(i.scale);}",
        "float4 displace(DisplacementContext ctx, Inputs i) {return float4(i.scale);}",
        "float3 displace(ShadingContext ctx, Inputs i) {return ctx.NORMAL;}",
    ] {
        assert!(compiler.compile_surface(&abi, &format!("{base}\n{bad}"), origin, "bad").is_err(), "{bad}");
    }
    Ok(())
}

#[test]
#[ignore="requires pinned Slang toolchain"]
fn displacement_reads_exclude_surface_only_attributes()->anyhow::Result<()> {
    let source="import vork; struct Inputs {Attribute<float> height; Attribute<float3> paint;}; float3 displace(DisplacementContext ctx, Inputs i){return ctx.NORMAL*read_attribute(i.height,ctx);} Closure surface(ShadingContext ctx,Inputs i){Closure c={};c.base_color=read_attribute(i.paint,ctx).value;return c;}";
    let p=Compiler::discover().compile_surface(&abi(),source,Path::new("reads.slang"),"reads")?;
    assert_eq!(p.displacement_attributes,["height"]);
    assert!(p.displacement_uses_normal);
    let p=Compiler::discover().compile_surface(&abi(),&source.replace("ctx.NORMAL*read_attribute(i.height,ctx)","float3(0.)"),Path::new("none.slang"),"none")?;
    assert!(p.displacement_attributes.is_empty());assert!(!p.displacement_uses_normal);Ok(())
}

#[test]
#[ignore="requires pinned Slang toolchain"]
fn displacement_resource_probe_identifies_images()->anyhow::Result<()> {
    let source="import vork; struct Inputs {Image height;Image paint;}; float3 displace(DisplacementContext ctx,Inputs i){return float3(0.,sample(i.height,ctx.POSITION.xy,ctx).r,0.);} Closure surface(ShadingContext ctx,Inputs i){Closure c={};c.base_color=sample(i.paint,ctx.TEXCOORD_0,ctx).rgb;return c;}";
    let p=Compiler::discover().compile_surface(&abi(),source,Path::new("images.slang"),"images")?;
    assert_eq!(p.displacement_images,["height"]);assert!(!p.displacement_uses_normal);Ok(())
}

#[test]
#[ignore="requires pinned Slang toolchain"]
fn displacement_rejects_scene_queries_and_surface_evaluation()->anyhow::Result<()> {
    let base="import vork; struct Inputs {}; Closure surface(ShadingContext ctx,Inputs i){return {}; }";
    for body in ["return world_to_camera(ctx.POSITION);", "ShadingContext s={};return object_direction_to_world(s,float3(1.,0.,0.));", "ShadingContext s={};return object_normal_to_world(s,float3(0.,1.,0.));","ShadingContext s={};return surface(s,i).base_color;"] {
        let error=Compiler::discover().compile_surface(&abi(),&format!("{base} float3 displace(DisplacementContext ctx,Inputs i){{{body}}}"),Path::new("restricted.slang"),"restricted").err().expect("invalid displacement stage installed");
        assert!(format!("{error:#}").contains("displace cannot call"),"{error:#}");
    }
    Ok(())
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn implicit_closure_defaults_match_effect_analysis() {
    let abi=Abi { aov_count: 3,context:Abi::spec_context(),closure:vork_slang::openpbr_fields()};
    let program=Compiler::discover().compile_surface(
        &abi,
        "struct Inputs {}; Closure surface(ShadingContext ctx, Inputs inputs) { Closure result; return result; }",
        Path::new("implicit-closure.slang"),
        "implicitclosure",
    ).unwrap();
    // Declared geometry_opacity=1 and transmission/subsurface weights=0.
    // WGSL's implicit zero alone would incorrectly classify this as transparent.
    assert!(!program.capabilities.opacity);
    assert!(!program.capabilities.transmission);
    assert!(!program.capabilities.subsurface);
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn constant_blender_math_drops_unused_native_operations() {
    // Compile the real scalar helper independently of surface ABI plumbing.
    let library=vork_slang::module_source("blender/nodes").unwrap();
    let start=library.find("public enum MathOperation").unwrap();
    let end=library.find("// Tangent-space Normal Map").unwrap();
    let math=&library[start..end];
    let compiler=Compiler::discover();
    for operation in ["ADD","MULTIPLY"] {
        let main=format!("import math;RWStructuredBuffer<float> values;[shader(\"compute\")][numthreads(1,1,1)] void main() {{values[0]=Math(MathOperation.{operation},values[0],values[1],0.,false);}}");
        let code=compiler.compile_modules(&[("math",math),("main",&main)],"main",&[("main","compute")],
            &["-target","spirv","-O1"]).unwrap();
        let words:Vec<_>=code.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut cursor=5;
        while cursor<words.len() {
            let instruction=words[cursor];let opcode=instruction&0xffff;
            // OpFunctionCall, OpBranchConditional, OpSwitch.
            assert!(!matches!(opcode,57|250|251),"constant {operation} retained a call or conditional dispatcher (opcode {opcode})");
            cursor+=(instruction>>16) as usize;
        }
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn compile_blender_socket_helpers() {
    let source = r#"
import vork;
import blender.nodes;
struct Inputs { float value; };
Closure surface(ShadingContext ctx,Inputs inputs) {
    float value=0.;
    for(int op=0;op<41;op++) value+=Math(MathOperation(op),inputs.value,.5,.25,false);
    value+=Fresnel(1.45,ctx.NORMAL,ctx)+Layer_Weight_Fresnel(.5,ctx.NORMAL,ctx);
    value+=Normal_Map(1.,float3(.5,.5,1.),ctx).x;
    let glass=Glass_BSDF(float3(1.),.15,1.45,ctx.NORMAL,0.,1.33);
    value+=glass.transmission_weight;
    let empty=Transparent_BSDF(float3(1.));
    return Diffuse_BSDF(float3(value+empty.geometry_opacity),0.,ctx.NORMAL);
}"#;
    let abi=Abi { aov_count: 3,context:Abi::spec_context(),closure:vork_slang::openpbr_fields()};
    Compiler::discover().compile_surface(&abi,source,Path::new("blender-nodes.slang"),"blendernodes").unwrap();
    let glass=Compiler::discover().compile_surface(&abi,
        "import vork;import blender.nodes;struct Inputs {};Closure surface(ShadingContext ctx,Inputs i) {return Glass_BSDF(float3(.8),.2,1.45,ctx.NORMAL,500.,1.33);}",
        Path::new("blender-glass.slang"),"blenderglass").unwrap();
    assert!(glass.capabilities.transmission);
    assert!(glass.capabilities.film);
    let no_film=Compiler::discover().compile_surface(&abi,
        "import vork;import blender.nodes;struct Inputs {};Closure surface(ShadingContext ctx,Inputs i) {Closure c=Glass_BSDF(float3(.8),.2,1.45,ctx.NORMAL,0.,1.33);c.thin_film_weight=0.;return c;}",
        Path::new("blender-glass-no-film.slang"),"blenderglassnofilm").unwrap();
    assert!(no_film.capabilities.transmission);
    assert!(!no_film.capabilities.film);

}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn outlined_surface_preserves_material_effects() {
    let abi=Abi { aov_count: 3,context:Abi::spec_context(),closure:vork_slang::openpbr_fields()};
    let compiler=Compiler::discover();
    for body in [
        "Closure surface(ShadingContext ctx,Inputs i) {Closure c={};c.base_color=float3(i.value);return c;}",
        "Closure surface(ShadingContext ctx,Inputs i) {Closure c={};c.geometry_opacity=i.value;c.transmission_weight=1.;c.geometry_thin_walled=i.value>0.;return c;}",
        "Blend surface(ShadingContext ctx,Inputs i) {Closure a={};Closure b={};b.geometry_opacity=.3;b.transmission_weight=1.;return blend(a,b,i.value);}",
    ] {
        let source=format!("import vork;struct Inputs {{float value;}};{body}");
        let before=compiler.compile_surface(&abi,&source,Path::new("surface-call.slang"),"before").unwrap();
        let source=source.replace(body,&format!("[noinline]\n{body}"));
        let after=compiler.compile_surface(&abi,&source,Path::new("surface-call.slang"),"after").unwrap();
        assert_eq!(serde_json::to_value(before.capabilities).unwrap(),serde_json::to_value(after.capabilities).unwrap(),"{body}");
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn proven_blends_above_64_lobes_compile_with_scene_sized_storage() {
    let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:vork_slang::openpbr_fields()};
    let mut source=String::from("import vork;struct Inputs {};Blend surface(ShadingContext ctx,Inputs i){Closure c={};Blend result=blend(c);\n");
    for _ in 1..65 {source.push_str("result=add(result,c);\n");}
    source.push_str("return result;}");
    let program=Compiler::discover().compile_surface(&abi,&source,Path::new("many-lobes.slang"),"manylobes").unwrap();
    assert_eq!(program.capabilities.blend_lobes,Some(65));
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn std_display_programs_own_grading_and_pixel_context() {
    let compiler = Compiler::discover();
    for (name, has_look) in [("std/agx", true), ("std/standard", false), ("blender/filmic", true)] {
        let program = compiler.compile_display(&abi(), vork_slang::module_source(name).unwrap(),
            Path::new(&format!("{name}.slang")), "displaytest").unwrap();
        for input in ["exposure", "gamma"] {
            assert!(program.inputs.fields.iter().any(|f| f.name == input));
        }
        assert_eq!(program.inputs.fields.iter().any(|f| f.name == "look"), has_look);
        assert!(!program.inputs.fields.iter().any(|f| f.name == "dither_intensity"));
        for gamma in [0., -1.] {
            let values = BTreeMap::from([("gamma".to_owned(), serde_json::json!(gamma))]);
            assert!(program.inputs.pack(&values, |_,_| Ok(0)).is_err());
        }
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn hair_dropped_stages_survive_compiler_cache_serialization() {
    let source = format!("{}\nClosure lit(ShadingContext ctx, Inputs inputs, Lighting lighting) {{Closure c={{}};return c;}}\nfloat3 displace(DisplacementContext ctx, Inputs inputs) {{return float3(1);}}",
        vork_slang::module_source("std/hair").unwrap());
    let program = Compiler::discover().compile_surface(&abi(), &source, Path::new("hair-stages.slang"), "hairstages").unwrap();
    assert!(program.capabilities.hair);
    assert!(!program.capabilities.lit && !program.capabilities.displacement);
    assert_eq!(program.dropped_stages, ["lit", "displace"]);
    let cached: vork_slang::Program = serde_json::from_slice(&serde_json::to_vec(&program).unwrap()).unwrap();
    assert_eq!(cached.dropped_stages, program.dropped_stages);
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn geometric_queries_remain_unknown_to_effect_analysis() {
    let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: vork_slang::openpbr_fields() };
    let compiler = Compiler::discover();
    for (index, field) in ["h.hit", "h.distance > 0.", "h.position.x > 0.", "h.geometric_normal.x > 0.", "h.shading_normal.x > 0.", "h.object_id == 0u", "h.instance_index == 0u", "h.primitive_id == 0u", "h.barycentrics.x > 0.", "h.uv.x > 0.", "h.front_facing"].iter().enumerate() {
        let source = format!("import vork; struct Inputs {{}}; Closure surface(ShadingContext ctx, Inputs inputs) {{ let h=trace(float3(0.),float3(0.,1.,0.),0.,1.); Closure c={{}}; c.transmission_weight=({field}) ? 1. : 0.;c.geometry_opacity=({field}) ? .5 : 1.; return c; }}");
        let program = compiler.compile_surface(&abi, &source, Path::new("trace-effects.slang"), &format!("trace{index}")).unwrap();
        assert!(program.capabilities.transmission, "{field} must remain scene-dependent");
        assert!(program.capabilities.dependencies.trace, "{field} must retain the AS dependency");
        assert!(program.effect_probe().unwrap().coverage_varies(&[],false), "{field} can vary with the scene snapshot");
    }
    let source = "import vork; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs inputs) { Closure c={}; if(false) { c.transmission_weight=trace(float3(0.),float3(0.,1.,0.),0.,1.).distance; } return c; }";
    let program = compiler.compile_surface(&abi, source, Path::new("trace-unused.slang"), "traceUnused").unwrap();
    assert!(!program.capabilities.dependencies.trace);
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn geometric_queries_require_bounded_caller_loops() {
    let compiler = Compiler::discover();
    for (index, condition) in ["k<8u", "k<uint(ctx.time)", "trace(ctx.POSITION,ctx.NORMAL,0.,1.).hit", "inputs.repeat", "k<inputs.limit", "sample(inputs.picture,float2(0.),fixed).x>0."].iter().enumerate() {
        let source = format!("import vork; struct Inputs {{bool repeat=false;uint limit=8u;Image picture;}}; Closure surface(ShadingContext ctx, Inputs inputs) {{ ShadingContext fixed={{}};float hits=0.; for(uint k=0u;{condition};k++) {{hits+=trace(ctx.POSITION,ctx.NORMAL,0.,1.).hit ? 1. : 0.;}} Closure c={{}};c.base_color=float3(hits);return c;}}");
        let result = compiler.compile_surface(&abi(), &source, Path::new("trace-loops.slang"), &format!("traceLoops{index}"));
        if index == 0 {result.unwrap();} else {assert!(format!("{:#}",result.unwrap_err()).contains("provably finite"));}
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn geometric_queries_admit_finite_procedural_helpers() {
    let compiler=Compiler::discover();
    for (index,expr) in [
        "noise(ctx.POSITION)",
        "noise(ctx.POSITION,5.,2.5)",
        "noise(ctx.POSITION,5.,2.,0.5,2.,1.,true,NoiseType.FBM,1)",
        "noise(ctx.POSITION,5.,2.,0.5,2.,1.,true,NoiseType.FBM,2)",
        "noise(ctx.POSITION,5.,2.,0.5,2.,1.,true,NoiseType.FBM,4)",
        "noise_color(ctx.POSITION).x", "voronoi(ctx.POSITION).distance",
        "white_noise(ctx.POSITION).value", "gabor(ctx.POSITION).value",
        "magic(ctx.POSITION).factor", "wave(ctx.POSITION)",
    ].iter().enumerate() {
        let source=format!("import vork; import blender.procedurals; struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs inputs) {{let h=trace(ctx.POSITION,ctx.NORMAL,0.,1.);Closure c={{}};c.base_color=float3({expr})+float3(h.hit?0.:.1);return c;}}");
        eprintln!("admitting {expr}");
        compiler.compile_surface(&abi(),&source,Path::new("finite-procedurals.slang"),&format!("finiteProcedural{index}")).unwrap_or_else(|e|panic!("{expr}: {e:#}"));
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn geometric_queries_prove_returns_and_name_unbounded_helpers() {
    let compiler=Compiler::discover();
    for (index,body) in [
        "for(int i=0;i<4;i++){if(i==3)return .5;}return 0.;",
        "for(int i=0;i<4;i++){if(x>float(i))return .5;}return 0.;",
        "for(int i=0;i<4;i++){for(int j=0;j<3;j++){if(x>float(j))return .5;}}return 0.;",
        "while(x>0.){x+=1.;}return x;",
        "for(int i=0;i<4;i++){if(x>0.)return .5;while(x<0.){x-=1.;}}return 0.;",
    ].iter().enumerate() {
        let source=format!("import vork;float failingHelper(float x) {{{body}}} struct Inputs {{}};Closure surface(ShadingContext ctx,Inputs inputs) {{Closure c={{}};c.base_color=float3(failingHelper(ctx.time))+float3(trace(ctx.POSITION,ctx.NORMAL,0.,1.).hit?0.:.1);return c;}}");
        let result=compiler.compile_surface(&abi(),&source,Path::new("helper-bounds.slang"),&format!("helperBounds{index}"));
        if index<3 {result.unwrap();} else {
            let error=format!("{:#}",result.unwrap_err());
            assert!(error.contains("failingHelper") && error.contains("loop count is not provably finite"),"{error}");
            assert_eq!(error.lines().count(),1,"{error}");
        }
    }
    // A finite graph is bounded by its function count, not a fixed depth cap.
    let mut source=String::from("import vork;float chain0(float x){return x;}");
    for i in 1..48 {source+=&format!("float chain{i}(float x){{return chain{}(x);}}",i-1);}
    source+="struct Inputs {};Closure surface(ShadingContext ctx,Inputs inputs){Closure c={};c.base_color=float3(chain47(ctx.time))+float3(trace(ctx.POSITION,ctx.NORMAL,0.,1.).hit?0.:.1);return c;}";
    compiler.compile_surface(&abi(),&source,Path::new("finite-chain.slang"),"finiteChain").unwrap();
    let source="import vork;float recursiveHelper(float x){return recursiveHelper(x);}
        struct Inputs {};Closure surface(ShadingContext ctx,Inputs inputs){Closure c={};c.base_color=float3(recursiveHelper(ctx.time))+float3(trace(ctx.POSITION,ctx.NORMAL,0.,1.).hit?0.:.1);return c;}";
    let error=format!("{:#}",compiler.compile_surface(&abi(),source,Path::new("recursive-helper.slang"),"recursiveChain").unwrap_err());
    assert!(error.contains("recursiveHelper") && error.to_lowercase().contains("recurs"),"{error}");
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn geometric_queries_refuse_geometry_and_output_stages() {
    let compiler=Compiler::discover();
    for (index, stage) in [
        "float3 displace(DisplacementContext ctx,Inputs i) {return float3(trace(float3(0.),float3(0.,1.,0.),0.,1.).distance);}",
        "Closure lit(ShadingContext ctx,Inputs i,Lighting l) {Closure c={};c.base_color=trace(ctx.POSITION,ctx.NORMAL,0.,1.).position;return c;}",
        "void aov(ShadingContext ctx,Inputs i,inout AovOutputs o) {write_aov(o,i.a,trace(ctx.POSITION,ctx.NORMAL,0.,1.).distance);}",
    ].iter().enumerate() {
        let source=format!("import vork;struct Inputs {{Aov a;}};Closure surface(ShadingContext c,Inputs i) {{return Closure();}} {stage}");
        let result=compiler.compile_surface(&abi(),&source,Path::new("trace-stage.slang"),&format!("traceStage{index}"));
        assert!(format!("{:#}",result.unwrap_err()).contains("trace is supported only in surface"));
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn reachable_queries_retain_native_work_effects() {
    let abi=abi();let compiler=Compiler::discover();
    let source="import vork;struct Inputs {};Closure surface(ShadingContext ctx,Inputs i) {let discarded=trace(ctx.POSITION,ctx.NORMAL,0.,1.);Closure c={};return c;}";
    let program=compiler.compile_surface(&abi,source,Path::new("trace-discarded.slang"),"discardedTrace").unwrap();
    assert!(program.capabilities.dependencies.trace);
    let native=abi.native_module();
    let start=native.find("__exported import program_trace;").unwrap();
    let query=&native[start..native[start..].find("public struct Lighting").unwrap()+start];
    let types=include_str!("../src/trace_types.slang");
    let compile=|body:&str,provider:Option<&str>| {
        let main=format!("import query;{} [[vk::binding(0,0)]] RWStructuredBuffer<uint> output;[shader(\"compute\")][numthreads(1,1,1)] void main() {{{body}}}",if provider.is_some(){"import provider;"}else{""});
        let mut modules=vec![("query",query),("program_trace",types),("main",main.as_str())];
        if let Some(provider)=provider {modules.push(("provider",provider));}
        compiler.compile_modules(&modules,"main",&[("main","compute")],&["-target","spirv","-O2","-emit-spirv-directly"])
    };
    let plain=compile("output[0]=7u;",None).unwrap();
    let unreachable=compile("if(false) {let ignored=trace(float3(0.),float3(0.,1.,0.),0.,1.);}output[0]=7u;",None).unwrap();
    assert_eq!(plain,unreachable,"unreachable queries must leave identical native code");
    let ignored="let ignored=trace(float3(0.),float3(0.,1.,0.),0.,1.);output[0]=7u;";
    assert!(format!("{:#}",compile(ignored,None).unwrap_err()).contains("unresolved external"));
    let live="output[0]=trace(float3(0.),float3(0.,1.,0.),0.,1.).hit ? 1u : 0u;";
    assert!(format!("{:#}",compile(live,None).unwrap_err()).contains("unresolved external"));
    let provider="import program_trace;[[vk::binding(1,0)]] RWStructuredBuffer<uint> counter;export TraceHit native_geometric_trace(float3 o,float3 d,float lo,float hi,uint mask,uint object) {uint index;InterlockedAdd(counter[0],1u,index);TraceHit h={};h.hit=index<16u;return h;}";
    for body in [live,ignored] {
    let counted=compile(body,Some(provider)).unwrap();
    let words:Vec<_>=counted.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
    let mut cursor=5;let mut atomics=0;
    while cursor<words.len() {let instruction=words[cursor];if instruction&0xffff==234 {atomics+=1;}cursor+=(instruction>>16) as usize;}
    assert!(atomics>0,"reachable queries must retain work accounting");

    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn authored_fog_candidates_use_the_physical_point_contract() {
    let compiler = Compiler::discover();
    let abi = abi();
    for (name, homogeneous) in [("fog-constant", true), ("fog-height", false)] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(format!("{name}.slang"));
        let source = fs::read_to_string(&path).unwrap();
        let program = compiler.compile_medium(&abi, &source, &path, &name.replace("-", "")).unwrap();
        assert_eq!(program.homogeneous, homogeneous);
        assert_eq!(program.inputs.fields.len(), if homogeneous {4} else {6});
        assert!(program.inputs.fields.iter().all(|field| !matches!(field.ty, Type::Image | Type::FloatGrid | Type::VectorGrid)));
        let words = program.inputs.pack(&BTreeMap::new(), |_, _| unreachable!()).unwrap();
        let words = words.iter().copied().map(Some).collect::<Vec<_>>();
        assert!(program.bounded_work_for_inputs(&abi, &words).unwrap()>0);
        assert!(!program.time_dependent_for_inputs(&abi, &words).unwrap());
        assert!(!program.scattering_free_for_inputs(&abi, &words).unwrap());
        assert!(program.emission_free_for_inputs(&abi, &words).unwrap());
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn medium_work_proof_refuses_unknown_loops_even_with_bounded_coefficients() {
    let compiler=Compiler::discover();let abi=abi();
    for (body,bounded) in [
        ("float v=0.;for(uint j=0u;j<4u;j++){v+=c.position.y;}m.absorption=float3(abs(v));",true),
        ("float v=0.;while(v<c.position.x){v+=1.;}m.absorption=float3(clamp(v,0.,1.));",false),
    ] {
        let source=format!("struct Inputs {{}}; MediumCoefficients medium(MediumContext c,Inputs i){{MediumCoefficients m={{}};{body}return m;}}");
        let program=compiler.compile_medium(&abi,&source,Path::new("work.slang"),"workproof").unwrap();
        let proof=program.bounded_work_for_inputs(&abi,&[]);
        assert_eq!(proof.is_ok(),bounded,"{body}: {proof:?}");
        if let Err(error)=proof {assert!(format!("{error:#}").contains("loop count is not provably finite"),"{error:#}");}
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn exponential_world_proof_accepts_world_and_identity_local_coordinates() {
    let compiler=Compiler::discover();let abi=abi();
    for coordinate in ["c.position.y","c.world_position.y"] {
        let source=format!("struct Inputs {{}}; MediumCoefficients medium(MediumContext c,Inputs i){{MediumCoefficients m={{}};m.absorption=float3(.5*exp(-.7*{coordinate}));return m;}}");
        let program=compiler.compile_medium(&abi,&source,Path::new("world-height.slang"),"worldheight").unwrap();
        let (absorption,scattering,gradient)=program.exponential_for_inputs(&abi,&[]).unwrap().expect(coordinate);
        assert_eq!(absorption,[0.5;3]);assert_eq!(scattering,[0.;3]);
        assert_eq!(gradient,[0.,-0.7,0.]);
    }
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn height_medium_preserves_provably_vacuum_channels() {
    let compiler=Compiler::discover();let abi=abi();
    let program=compiler.compile_medium(&abi,include_str!("fixtures/fog-height.slang"),Path::new("height.slang"),"heightvacuum").unwrap();
    let inputs=serde_json::from_value(serde_json::json!({"absorption":[0,0.5,1],"scattering":[0,0,0],"emission":[0,0,0],"falloff":0.7})).unwrap();
    let words=program.inputs.pack(&inputs,|_,_|unreachable!()).unwrap();
    let proof=words.into_iter().map(Some).collect::<Vec<_>>();
    assert!(program.scattering_free_for_inputs(&abi,&proof).unwrap());
    assert!(program.emission_free_for_inputs(&abi,&proof).unwrap());
    assert!(program.exponential_for_inputs(&abi,&proof).unwrap().is_some());
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn exponential_world_proof_refuses_vanished_positive_amplitude() {
    let compiler=Compiler::discover();let abi=abi();
    let source="struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m={};m.absorption=float3(exp(-c.world_position.y-1000.));return m;}";
    let program=compiler.compile_medium(&abi,source,Path::new("thin-world.slang"),"thinworld").unwrap();
    assert!(program.exponential_for_inputs(&abi,&[]).unwrap().is_none());
}

#[test]
fn removed_imports_refuse_with_replacement() {
    let compiler = Compiler::discover();
    for (module, replacement) in [("post.white_balance", "std.white_balance"), ("std.particles", "fluid.particles")] {
        let source = format!("import {module}; struct Inputs {{}}; import std.view; ViewResult view(ViewContext c, Inputs i) {{return finish_view(float3(0.25),c,1.);}}");
        let error = match compiler.prepare_program(&abi(), &source, Path::new("removed.slang")) {
            Ok(_) => panic!("removed module {module} accepted"),
            Err(error) => error.to_string(),
        };
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(error.contains(replacement) && error.contains("vork help upgrade"), "{error}");
        let rewritten = source.replace(module, replacement);
        compiler.prepare_program(&abi(), &rewritten, Path::new("rewritten.slang")).unwrap();
        let nested = compiler.prepare_modules(&[("main", "import helper;"), ("helper", &format!("import {module};"))], "main", &[], &[]);
        let error = nested.err().expect("nested removed import accepted").to_string();
        assert_eq!(error.lines().count(), 1, "{error}");
        assert!(error.contains(replacement), "{error}");
        let inactive = format!("// import {module};\n#if 0\nimport {module};\n#endif\n{rewritten}");
        compiler.prepare_program(&abi(), &inactive, Path::new("inactive.slang")).unwrap();
    }
}

#[test]
fn bloom_compositor_compiles() -> anyhow::Result<()> {
    Compiler::discover().compile_composite(&abi(), vork_slang::module_source("std/bloom").unwrap(), Path::new("std/bloom.slang"), "bloom")?;
    Ok(())
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn blender_math_uses_rna_fract_and_refuses_fraction() {
    let compiler=Compiler::discover();
    let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:vork_slang::openpbr_fields()};
    let source="import blender.nodes; struct Inputs {}; Closure surface(ShadingContext ctx, Inputs i) { return unlit(float3(Math(MathOperation.FRACT,1.25,0.,0.,false))); }";
    compiler.compile_surface(&abi,source,Path::new("fract.slang"),"fract").unwrap();
    let error=compiler.prepare_program(&abi,&source.replace("FRACT,","FRACTION,"),Path::new("fraction.slang"));
    let error=match error {Ok(_)=>panic!("retired FRACTION accepted"),Err(error)=>error.to_string()};
    assert_eq!(error.lines().count(), 1, "{error}");
    assert!(error.contains("FRACTION"), "{error}");
    assert!(error.split(|c: char| !c.is_ascii_alphanumeric()).any(|token| token == "FRACT"), "{error}");
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn missing_enum_members_name_actual_supported_values() {
    let compiler = Compiler::discover();
    let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: vork_slang::openpbr_fields() };
    let source = "enum Finish { Matte, Glossy }; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return unlit(float3(int(Finish.Gloss))); }";
    let error = match compiler.prepare_program(&abi, source, Path::new("finish.slang")) {
        Ok(_) => panic!("unknown enum member accepted"),
        Err(error) => error.to_string(),
    };
    assert_eq!(error.lines().count(), 1, "{error}");
    assert!(error.contains("Gloss") && error.contains("Glossy"), "{error}");
    // A user's own enum can still declare the same spelling as a removed library member.
    let source = "enum MathOperation { FRACTION }; struct Inputs {}; Closure surface(ShadingContext c, Inputs i) { return unlit(float3(int(MathOperation.FRACTION))); }";
    assert!(compiler.prepare_program(&abi, source, Path::new("own-enum.slang")).is_ok());
}

#[test]
#[ignore = "requires pinned Slang toolchain"]
fn object_vectors_and_normals_retain_runtime_material_effects() {
    let mut abi = abi();
    abi.closure.push(Field::new("transmission_weight", Type::Float, Some("0.")));
    for helper in ["object_direction_to_world", "object_normal_to_world"] {
        let source = format!("import vork; struct Inputs {{}}; Closure surface(ShadingContext ctx, Inputs i) {{ Closure c={{}}; c.transmission_weight={helper}(ctx,float3(0.,1.,0.)).x; return c; }}");
        let program = Compiler::discover().compile_surface(&abi,&source,Path::new("object-vector.slang"),"vectorEffects").unwrap();
        assert!(program.capabilities.transmission, "{helper} depends on the installed object transform");
    }
}
