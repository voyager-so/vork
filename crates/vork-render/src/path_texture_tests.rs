//! Path texture filtering converges without changing normal-frame validity.
use super::*;
use serde_json::{Value, json};

const NORMAL_PROGRAM: &str = r#"import vork; import std.texture;
struct Inputs {Image normal;float base_weight=1.;float3 base_color=float3(.6);float specular_weight=0.;float normal_rotation=0.;float coat_weight=0.;float coat_roughness=.2;float coat_strength=0.;float coat_rotation=.7;};
TextureFootprint footprint(ShadingContext ctx,float angle) {
    let t=float2x2(cos(angle),sin(angle),-sin(angle),cos(angle));
    return texture_footprint(mul(ctx.TEXCOORD_0,t),mul(ctx.duv_dx[0],t),mul(ctx.duv_dy[0],t),ctx);
}
Closure surface(ShadingContext ctx,Inputs p) {
    Closure c={};c.base_weight=p.base_weight;c.base_color=p.base_color;c.specular_weight=p.specular_weight;c.coat_weight=p.coat_weight;
    float2 variance;float2 coat_variance;
    c.geometry_normal=texture_normal(p.normal,footprint(ctx,p.normal_rotation),ctx,float3(1.),p.normal_rotation!=0.,variance);
    c.geometry_coat_normal=texture_normal(p.normal,footprint(ctx,p.coat_rotation),ctx,float3(p.coat_strength,p.coat_strength,1.),p.coat_rotation!=0.,coat_variance);
    c.specular_roughness=filtered_ggx_roughness(c.specular_roughness,variance);
    c.coat_roughness=filtered_ggx_roughness(p.coat_roughness,coat_variance);return c;
}"#;

const HEIGHT_PROGRAM: &str = "import vork; import blender.vector; import blender.texture_coordinate; struct Inputs {}; struct Height:BumpHeight {ShadingContext context;float evaluate(float2 offset) {return .75*texture_coordinate(context,offset).Object.x;}}; Closure surface(ShadingContext ctx,Inputs inputs) {Height height={};height.context=ctx;Closure c={};c.geometry_opacity=1.;c.geometry_normal=bump(height,ctx.NORMAL,ctx.position_dx,ctx.position_dy,1.,1.,.1);c.base_weight=1.;c.base_color=float3(.6);c.specular_weight=0.;return c;}";

fn directory(name: &str) -> Result<PathBuf> {
    let root=std::env::temp_dir().join(format!("vork-path-texture-{name}-{}",std::process::id()));
    std::fs::create_dir_all(&root)?;
    // An image enables terminal environment NEE; a constant world does not.
    image::RgbImage::from_pixel(1,1,image::Rgb([255,255,255])).save(root.join("white.png"))?;
    Ok(root)
}

fn slab() -> Value {
    json!({"version":2,
        "composition":{"space":"3d","view":{"program":"std/standard","inputs":{}}},"render":{"dither":false,"resolution":[16,16],"background":[0,0,0],"settings":{"integrator":"path","denoiser":null,"samples":256}},
        "images":{"normal":{"file":"normal.png","color_space":"Non-Color",
            "mag_filter":"linear","min_filter":"linear"},"white":{"file":"white.png"}},
        "programs":{"mapped":{"source":NORMAL_PROGRAM}},
        "materials":{"environment_image":{"program":"std/environment_image","inputs":{"image":{"image":"white"}}},"m":{"program":"mapped","inputs":{"normal":{"image":"normal"}}}},
        "buffers":{"p":{"type":"vec3","data":[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]},
            "uv":{"type":"vec2","data":[[0,1],[1,1],[1,0],[0,0]]},
            "n":{"type":"vec3","data":[[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},
            "t":{"type":"vec4","data":[[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]},
            "i":{"type":"u32","data":[0,1,2,0,2,3]}},
        "geometry":{"quad":{"attributes":{"POSITION":"p","NORMAL":"n","TANGENT":"t","TEXCOORD_0":"uv"},"indices":"i"}},
        "nodes":[{"type":"mesh","geometry":"quad","materials":["m"]},
            {"type":"camera","translation":[0,0,3],"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":2}}},
            {"type":"light","shape":"environment","material":"environment_image","energy":1}]})
}

fn render(r: &Renderer, input: Value, root: &Path) -> Result<Vec<[f32;4]>> {
    let scene=r.install(vork_scene::v2::from_test_value(input)?,root)?;
    let mut slot=scene.create_frame_slot(r,ReadbackPrecision::Float)?;
    slot.submit_outputs(&scene,r,0.,ReadbackOutputs {float:FloatOutputs {color:true,..Default::default()},..Default::default()})?;
    slot.collect(r)?.linear_rgba.context("linear colour readback")
}
fn mean(pixels: &[[f32;4]]) -> f32 { pixels.iter().map(|p|p[0]).sum::<f32>()/pixels.len() as f32 }
fn tilted_map(root: &Path) -> Result<()> {
    image::ImageBuffer::<image::Rgb<u16>,_>::from_pixel(1,1,image::Rgb([204*257,128*257,230*257])).save(root.join("normal.png"))?;
    Ok(())
}

#[test]
#[ignore = "requires pinned Slang and Vulkan ray queries"]
fn minified_normals_converge_to_texel_response() -> Result<()> {
    let r = crate::test_renderer()?;
    let root=directory("minification")?;
    image::ImageBuffer::<image::Rgb<u16>,_>::from_fn(128,128,|x,_| {
        image::Rgb([if x%2==0 {204*257} else {51*257},128*257,230*257])
    }).save(root.join("normal.png"))?;
    let mut input=slab();
    let reference=render(&r,input.clone(),&root)?;
    input["images"]["normal"]["min_filter"]=json!("linear_mipmap_linear");
    let filtered=render(&r,input,&root)?;
    let reference_mean=mean(&reference);
    assert!(reference_mean>0.45 && reference_mean<0.59,"normal detail must change the white-world response: {reference_mean}");
    let rmse=(reference.iter().zip(&filtered).map(|(a,b)|(a[0]-b[0]).powi(2)).sum::<f32>()/reference.len() as f32).sqrt();
    eprintln!("minified normals: reference {reference_mean}, RMSE {rmse}");
    assert!(rmse<0.003,"normal minification retained a sample-independent bias: {rmse}");
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires pinned Slang and Vulkan ray queries"]
fn small_uv_frames_retain_constant_normal_maps() -> Result<()> {
    let r = crate::test_renderer()?;
    let root=directory("small-uv")?;tilted_map(&root)?;
    let reference=mean(&render(&r,slab(),&root)?);
    assert!(reference>0.45 && reference<0.55,"tilted map must be active: {reference}");
    for supplied_tangent in [false,true] {
        let mut input=slab();
        input["buffers"]["uv"]["data"]=json!([[0,0.0001],[0.0001,0.0001],[0.0001,0],[0,0]]);
        if supplied_tangent {
            // A turned map requires a derivative frame even with authored tangents.
            input["materials"]["m"]["inputs"]["normal_rotation"]=json!(0.3);
        } else {input["geometry"]["quad"]["attributes"].as_object_mut().unwrap().remove("TANGENT");}
        let measured=mean(&render(&r,input,&root)?);
        assert!((measured-reference).abs()<0.005,"small UV frame, supplied tangent={supplied_tangent}: {measured} vs {reference}");
    }
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires pinned Slang and Vulkan ray queries"]
fn translated_hits_retain_normal_and_bump_differentials() -> Result<()> {
    let r = crate::test_renderer()?;
    let root=directory("translated")?;tilted_map(&root)?;
    let interior_mean=|pixels: &[[f32;4]]| {
        (4..12).flat_map(|y|(4..12).map(move |x|pixels[y*16+x][0])).sum::<f32>()/64.
    };
    for procedural in [false,true] {
        let mut input=slab();
        // One pixel is exactly 1/512 world units, resolvable at this translation;
        // a 1/sqrt(1024) sample footprint is below the world-position spacing.
        input["render"]["settings"]["samples"]=json!(1024);
        input["nodes"][1]["material"]["inputs"]["ortho_scale"]=json!(0.03125);
        // A constant map needs a derivative frame when tangents are absent.
        input["geometry"]["quad"]["attributes"].as_object_mut().unwrap().remove("TANGENT");
        if procedural {
            input["programs"]=json!({"slope":{"source":HEIGHT_PROGRAM}});
            input["materials"]["m"]=json!({"program":"slope"});
        }
        let reference=interior_mean(&render(&r,input.clone(),&root)?);
        assert!(reference>0.45 && reference<0.55,"mapped response must be active: {reference}");
        vork_scene::v2::set_transform(&mut input["nodes"][0], json!({"translation":[4096,4096,0]}));
        vork_scene::v2::set_transform(&mut input["nodes"][1], json!({"translation":[4096,4096,3]}));
        let translated=interior_mean(&render(&r,input,&root)?);
        assert!((translated-reference).abs()<0.005,"translation erased hit differentials, procedural={procedural}: {translated} vs {reference}");
    }
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
#[ignore = "requires pinned Slang and Vulkan"]
fn coat_normal_is_independent_of_base_normal() -> Result<()> {
    let r = crate::test_renderer()?;
    let root=directory("coat-normal")?;tilted_map(&root)?;
    for integrator in ["raster","path"] {
        let mut input=slab();
        input["render"]["settings"]=json!({"integrator":integrator,"denoiser":null,"samples":16});
        input["render"]["settings"]["samples"]=json!(16);
        // Keep an independently mapped base throughout the comparison.
        let uncoated=render(&r,input.clone(),&root)?;
        input["materials"]["m"]["inputs"]["coat_strength"]=json!(1.);
        let mapped_uncoated=render(&r,input.clone(),&root)?;
        assert_eq!(uncoated,mapped_uncoated,"coat map changed an absent coat ({integrator})");
        input["materials"]["m"]["inputs"]["coat_weight"]=json!(1);
        input["materials"]["m"]["inputs"]["coat_roughness"]=json!(0.2);
        let mapped=render(&r,input.clone(),&root)?;
        input["materials"]["m"]["inputs"].as_object_mut().unwrap().remove("coat_strength");
        let plain=render(&r,input.clone(),&root)?;
        let difference=mapped.iter().zip(&plain).map(|(a,b)|(a[0]-b[0]).abs()).sum::<f32>();
        eprintln!("{integrator}: coat map absolute red difference {difference}; absent-coat pixels identical");
        assert!(difference>0.001,"coat map had no effect ({integrator}): {difference}");
        // A zero-strength map must restore the exact unbound pixels, independently
        // of the base map and the coat map's UV rotation.
        input["materials"]["m"]["inputs"]["coat_strength"]=json!(0.);
        crate::test_linear::assert_linear_eq(&plain,&render(&r,input,&root)?);
    }
    std::fs::remove_dir_all(root)?;
    Ok(())
}
