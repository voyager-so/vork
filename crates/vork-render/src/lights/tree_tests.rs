use super::Node;
use super::*;
use crate::*;

fn source(x: f32, y: f32, power: f64) -> Source {
    Source {
        kind: Kind::Analytic,
        identity: 0,
        bounds: (Vec3::new(x, y, 2.), Vec3::new(x, y, 2.)),
        power,
        infinite: false,
    }
}
fn branch(a: &Node, b: &Node, p: glam::DVec3) -> f64 {
    let weight = |n: &Node| {
        let lo = Vec3::from_slice(&n.lower_power).as_dvec3();
        let hi = Vec3::from_slice(&n.upper_infinite).as_dvec3();
        let distance = if n.upper_infinite[3] != 0. {
            1.
        } else {
            (p - (lo + hi) * 0.5)
                .length_squared()
                .max(((hi - lo) * 0.5).length_squared())
                .max(1e-12)
        };
        f64::from(n.lower_power[3]) / distance
    };
    let x = weight(a);
    let y = weight(b);
    let uniform = f64::from(a.links[3]) / f64::from(a.links[3] + b.links[3]);
    if x + y == 0. { uniform } else { x / (x + y) }
}
// Independent exhaustive propagation, not a traversal/replay mirror.
fn probabilities(tree: &Tree, p: Vec3) -> Vec<f64> {
    let mut mass = vec![0.; tree.nodes.len()];
    if mass.is_empty() {
        return mass;
    }
    mass[0] = 1.;
    for (i, n) in tree.nodes.iter().enumerate() {
        if n.links[0] != u32::MAX {
            let a = n.links[0] as usize;
            let b = n.links[1] as usize;
            let q = branch(&tree.nodes[a], &tree.nodes[b], p.as_dvec3());
            mass[a] = mass[i] * q;
            mass[b] = mass[i] * (1. - q);
        }
    }
    tree.leaves
        .iter()
        .map(|&i| 0.25 / tree.leaves.len() as f64 + 0.75 * mass[i as usize])
        .collect()
}
#[test]
fn topology_support_and_extremes() -> Result<()> {
    assert!(Tree::build(&[])?.nodes.is_empty());
    assert_eq!(std::mem::size_of::<Node>(), 64);
    for count in [1, 2, 3, 127, 1024] {
        let sources: Vec<_> = (0..count)
            .map(|i| source(i as f32, 0., if i % 3 == 0 { 0. } else { 1e30 }))
            .collect();
        let tree = Tree::build(&sources)?;
        assert_eq!(tree.nodes.len(), 2 * count - 1);
        for point in [Vec3::ZERO, Vec3::new(5., 0., 2.), Vec3::splat(1e15)] {
            let p = probabilities(&tree, point);
            assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-12);
            assert!(p.iter().all(|&v| v > 0. && v.is_finite()));
        }
    }
    assert!(Tree::build(&[source(0., 0., f64::NAN)]).is_err());
    let t = Tree::build(&[source(0., 0., 0.), source(1., 0., 0.), source(2., 0., 0.)])?;
    for p in probabilities(&t, Vec3::ZERO) {
        assert!((p - 1. / 3.).abs() < 1e-12);
    }
    Ok(())
}

fn dispatch(r: &Renderer, tree: &Tree, body: &str, count: u32) -> Result<Vec<[f32; 4]>> {
    let native = medium::Native {
        includes: vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders")],
        ..Default::default()
    };
    // Include the production implementation so boundary probes can inspect its
    // private partition helper without making that helper a public shader API.
    let source = format!(
        "#include \"light_tree.slang\"\nimport vork_constants;\n[[vk::binding(0,0)]] StructuredBuffer<LightTreeNode> nodes;\n[[vk::binding(1,0)]] RWStructuredBuffer<float4> output;\n{body}"
    );
    let entries = [
        pipelines::buffer_entry(
            0,
            wgpu::ShaderStages::COMPUTE,
            wgpu::BufferBindingType::Storage { read_only: true },
        ),
        pipelines::buffer_entry(
            1,
            wgpu::ShaderStages::COMPUTE,
            wgpu::BufferBindingType::Storage { read_only: false },
        ),
    ];
    let pipeline = medium::test_pipeline(r, &source, &native, &entries, (1, 1, 1))?;
    let nodes = init(
        &r.device,
        "light tree",
        bytemuck::cast_slice(&tree.nodes),
        wgpu::BufferUsages::STORAGE,
    );
    let size = u64::from(count) * 16;
    let output = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light tree results"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = r.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = r.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            entry(0, nodes.as_entire_binding()),
            entry(1, output.as_entire_binding()),
        ],
    });
    let mut encoder = r.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(count, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
    r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |v| {
        let _ = tx.send(v);
    });
    r.device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    rx.recv()??;
    let values = bytemuck::cast_slice(&readback.get_mapped_range(..)).to_vec();
    r.errors.check()?;
    Ok(values)
}
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn weighted_remapping_never_enters_a_zero_power_branch() -> Result<()> {
    let r = crate::test_renderer()?;
    // #1186: the old selector remapped a finite coordinate to 1, entered a
    // zero-power branch, and divided 0/0. A zero light contribution then became
    // NaN when divided by the selected probability. Each subtree here has one
    // powered and one dark leaf, so the weighted strategy must avoid dark leaves
    // even immediately below/above the root's partition boundary.
    let sources = [source(0., 0., 1.), source(0., 0., 0.),
        source(100., 0., 1.), source(100., 0., 0.)];
    let tree = Tree::build(&sources)?;
    let body = format!(r#"
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    float3 point=float3((float(id.x/2u)+.5)/32768.*400.-150.,1.,0.);
    uint2 children=nodes[0].links.xy;
    float split=light_tree_branch(nodes[children.x],nodes[children.y],point);
    bool left=(id.x&1u)==0u;
    float u=asfloat(left ? asuint(split)-1u : asuint(split)+1u);
    var pick=light_tree_pick(nodes,{}u,point,float2(u,.75));
    float expected=REG_LIGHT_TREE_UNIFORM_MIX/4.+(1.-REG_LIGHT_TREE_UNIFORM_MIX)*(left ? split : 1.-split);
    output[id.x]=float4(float(nodes[pick.leaf].source.z),pick.probability/expected,
        light_tree_probability(nodes,pick.leaf,point)/expected,left ? u/split : (u-split)/(1.-split));
}}
"#, tree.nodes.len());
    let values = dispatch(&r, &tree, &body, 65534)?;
    let mut rounded_to_one = 0;
    for (i, v) in values.iter().enumerate() {
        assert_eq!(v[0], if i % 2 == 0 { 0. } else { 2. },
            "weighted selection entered a dark branch at case {i}: {v:?}");
        assert!(v.iter().all(|x| x.is_finite()), "non-finite selection at {i}: {v:?}");
        assert!((v[1]-1.).abs()<3e-5 && (v[2]-1.).abs()<3e-5,
            "sample/replayed probability at {i}: {v:?}");
        rounded_to_one += usize::from(v[3] >= 1.);
    }
    eprintln!("light-tree boundary regression: {} cases, {rounded_to_one} unbounded remaps rounded to 1", values.len());
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn compiled_sampler_matches_brute_force_and_reduces_many_light_noise() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut sources = Vec::new();
    for y in 0..32 {
        for x in 0..32 {
            let mut s = source(x as f32 * 4. - 62., y as f32 * 4. - 62., 1.);
            s.identity = sources.len() as u32;
            sources.push(s);
        }
    }
    let tree = Tree::build(&sources)?;
    let body = format!(
        r#"
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    uint i=id.x; float3 point=float3(1.,3.,0.);
    var pick=light_tree_pick(nodes,{}u,point,float2((float(i%16383u)+.5)/16383.,(float(i/16383u)+.5)/4.));
    output[i]=float4(float(nodes[pick.leaf].source.z),pick.probability,light_tree_probability(nodes,pick.leaf,point),0.);
}}"#,
        tree.nodes.len()
    );
    // Below the 65535 one-dimensional dispatch limit.
    let values = dispatch(&r, &tree, &body, 65532)?;
    let expected = probabilities(&tree, Vec3::new(1., 3., 0.));
    let mut hits = vec![0; 1024];
    for v in values {
        let id = v[0] as usize;
        hits[id] += 1;
        assert!((f64::from(v[1]) / expected[id] - 1.).abs() < 3e-5);
        assert!((v[1] - v[2]).abs() < 1e-7);
    }
    for (n, p) in hits.iter().zip(expected) {
        assert!(
            (f64::from(*n) - 65532. * p).abs() < 8.,
            "frequency mismatch {n} {p}"
        );
    }
    for count in [1, 5, 513] {
        let mut edge: Vec<_> = (0..count)
            .map(|i| {
                source(
                    i as f32,
                    0.,
                    if count == 513 {
                        if i == 512 { 1e38 } else { 0. }
                    } else {
                        [0., 1e38, 1e-30, 2., 3.][i]
                    },
                )
            })
            .collect();
        if count > 1 {
            edge[3].infinite = true;
            edge[3].kind = Kind::Environment;
            edge[4].kind = Kind::Mesh;
        }
        let edge = Tree::build(&edge)?;
        let expected = probabilities(&edge, Vec3::new(0., 0., 2.));
        let body = format!(
            r#"
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
float u=id.x==0?0.:(id.x==1?0.9999999403953552:float(id.x)/4096.);
var p=light_tree_pick(nodes,{}u,float3(0.,0.,2.),float2(u,(float(id.x%4u)+.5)/4.));
output[id.x]=float4(float(nodes[p.leaf].source.z),p.probability,light_tree_probability(nodes,p.leaf,float3(0.,0.,2.)),0.);
}}"#,
            edge.nodes.len()
        );
        let mut seen = vec![false; count];
        for v in dispatch(&r, &edge, &body, 4096)? {
            seen[v[0] as usize] = true;
            assert!(
                (f64::from(v[1]) / expected[v[0] as usize] - 1.).abs() < 3e-5,
                "HDR probability: sample={v:?}, expected={expected:?}"
            );
            assert!((v[1] - v[2]).abs() < 1e-7);
        }
        assert!(
            seen.iter().all(|v| *v),
            "uniform strategy lost source support"
        );
        if count == 1 {
            let body = "[shader(\"compute\")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {var p=light_tree_pick(nodes,0,float3(0.),float2(0.));output[0]=float4(p.probability,p.leaf==0xffffffffu?1.:0.,0.,0.);}";
            assert_eq!(dispatch(&r, &edge, body, 1)?[0], [0., 1., 0., 0.]);
        }
    }
    // Direct-light images of a Lambertian plane, exact unoccluded point-source
    // integrand. Same source domain/budget for uniform and tree, exhaustive truth.
    for scene in 0..4 {
        let mut scene_sources = sources.clone();
        for (i, s) in scene_sources.iter_mut().enumerate() {
            if scene == 1 {
                s.bounds.0.x *= 0.08;
                s.bounds.1 = s.bounds.0;
            }
            if scene == 2 {
                s.power = if i % 13 == 0 { 100. } else { 0.01 };
            }
            if scene == 3 {
                // Unit rectangles, equivalently two equal-area mesh triangles.
                s.kind = if i % 2 == 0 {
                    Kind::Mesh
                } else {
                    Kind::Analytic
                };
                s.bounds.0 -= Vec3::new(0.5, 0.5, 0.);
                s.bounds.1 += Vec3::new(0.5, 0.5, 0.);
            }
        }
        let tree = Tree::build(&scene_sources)?;
        let body = format!(
            r#"
uint hash(uint x) {{x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}}
float value(float3 p,float3 light) {{float3 d=light-p;float r2=dot(d,d);return {scene}u==3u ? d.z*d.z/(r2*r2) : d.z/(r2*sqrt(r2));}}
float3 center(uint n) {{return .5*nodes[n].lower_power.xyz+.5*nodes[n].upper_infinite.xyz;}}
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    uint i=id.x;float3 p=float3((float(i%128u)+.5)/128.*120.-60.,(float(i/128u)+.5)/128.*120.-60.,0.);
    float truth=0.,second=0.;
    for(uint n=0;n<{}u;n++)if(nodes[n].links.x==0xffffffffu) {{
        float mean=0.,square=0.;uint count={scene}u==3u?64u:1u;
        for(uint j=0;j<count;j++) {{
            float3 offset=count==1u?float3(0.):float3((float(j%8u)+.5)/8.-.5,(float(j/8u)+.5)/8.-.5,0.);
            float f=value(p,center(n)+offset)*nodes[n].lower_power.w;mean+=f; square+=f*f;
        }}
        truth+=mean/float(count);second+=square/float(count)*1024.;
    }}
    float tree=0.,uniform=0.;
    for(uint s=0;s<64u;s++) {{
        float u=float(hash(i*64u+s)>>8)*(1./16777216.);
        float3 offset={scene}u==3u?float3(float(hash((i*64u+s)^0x68bc21ebu)>>8)*(1./16777216.)-.5,float(hash((i*64u+s)^0x02e5be93u)>>8)*(1./16777216.)-.5,0.):float3(0.);
        var pick=light_tree_pick(nodes,{}u,p,float2(u,float(hash((i*64u+s)^0x9e3779b9u)>>8)*(1./16777216.)));tree+=value(p,center(pick.leaf)+offset)*nodes[pick.leaf].lower_power.w/pick.probability;
        uint n=0;float v=u;while(nodes[n].links.x!=0xffffffffu) {{uint2 c=nodes[n].links.xy;float q=float(nodes[c.x].links.w)/float(nodes[n].links.w);bool left=v<q;v=left?v/q:(v-q)/(1.-q);n=left?c.x:c.y;}}
        uniform+=value(p,center(n)+offset)*nodes[n].lower_power.w*1024.;
    }}
    output[i]=float4(tree/64.,uniform/64.,truth,max(0.,second-truth*truth));
}}"#,
            tree.nodes.len(),
            tree.nodes.len()
        );
        let values = dispatch(&r, &tree, &body, 128 * 128)?;
        let mut sums = [0.; 3];
        let mut mse = [0.; 2];
        let mut uniform_variance = 0.;
        for v in &values {
            uniform_variance += f64::from(v[3]) / 64.;
            for c in 0..3 {
                sums[c] += f64::from(v[c]);
            }
            for c in 0..2 {
                mse[c] += (f64::from(v[c]) - f64::from(v[2])).powi(2);
            }
        }
        eprintln!(
            "many-light scene {scene} 1024 sources 64 spp: tree/reference={} uniform/reference={} tree/uniform MSE={} uniform energy standard error={}",
            sums[0] / sums[2],
            sums[1] / sums[2],
            mse[0] / mse[1],
            uniform_variance.sqrt() / sums[2]
        );
        assert!((sums[0] / sums[2] - 1.).abs() < 0.02);
        // The deliberately bad uniform proposal can have large mean error at
        // 64 spp. Test against its independently enumerated sampling variance,
        // not a fixed energy tolerance that mistakes variance for bias.
        assert!((sums[1] - sums[2]).abs() < 5. * uniform_variance.sqrt());
        assert!(mse[0] < mse[1] * 0.3);
        if let Ok(dir) = std::env::var("VORK_LIGHT_TREE_EVIDENCE") {
            std::fs::create_dir_all(&dir)?;
            let mut image = image::RgbImage::new(384, 128);
            for (i, v) in values.iter().enumerate() {
                for c in 0..3 {
                    let x = (v[c].max(0.) / (1. + v[c].max(0.))).sqrt();
                    image.put_pixel(
                        (i % 128 + c * 128) as u32,
                        (i / 128) as u32,
                        image::Rgb([(x * 255.) as u8; 3]),
                    );
                }
            }
            image.save(
                std::path::Path::new(&dir)
                    .join(format!("scene-{scene}-tree-uniform-reference.png")),
            )?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn installed_scene_source_identities_and_shapes() -> Result<()> {
    let r = crate::test_renderer()?;
    let mut input: serde_json::Value =
        serde_json::from_str(include_str!("../../tests/fixtures/analytic.json"))?;
    input["animation"] = serde_json::json!([]);
    input["render"] = serde_json::json!({"resolution":[8,8],"settings":{"samples":1,"integrator":"path","denoiser":null}});
    input["materials"] = serde_json::json!({"emitter":{"inputs":{"emission_color":[1,0.5,0.2],"emission_luminance":3}}});
    let mut nodes = vec![
        serde_json::json!({"type":"camera","translation":[0,0,5],"material":{"program":"std/camera"}}),
        serde_json::json!({"type":"mesh","geometry":"shape_box","materials":["emitter"]}),
    ];
    for (i, shape) in ["sun", "point", "spot", "rectangle", "ellipse"]
        .iter()
        .enumerate()
    {
        nodes.push(serde_json::json!({"type":"light","shape":shape,"translation":[i as f32,2,3],"energy":i+1}));
    }
    input["buffers"]["field"] = serde_json::json!({"type":"f32","dims":[4,4,4],"data":(0..64).map(|i|{let p=Vec3::new((i%4) as f32,(i/4%4) as f32,(i/16) as f32)*0.5-Vec3::splat(0.75);p.length()-0.6}).collect::<Vec<_>>()});
    input["geometry"]["field"] =
        serde_json::json!({"topology":"implicit","attributes":{"phi":"field"}});
    input["materials"]["transparent"] = serde_json::json!({"opacity":"transmittance","inputs":{"geometry_opacity":0.5,"emission_luminance":3}});
    nodes.push(serde_json::json!({"type":"mesh","geometry":"field","materials":["emitter"],"translation":[3,0,0]}));
    nodes.push(serde_json::json!({"type":"mesh","geometry":"field","materials":["transparent"],"translation":[-3,0,0]}));
    nodes.push(serde_json::json!({"type":"light","shape":"environment","energy":1}));
    input["nodes"] = nodes.into();
    let scene = r.install(
        vork_scene::v2::from_test_value(input)?,
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
    )?;
    let worlds: Vec<_> = scene.nodes.iter().map(|n| n.world).collect();
    let lights = scene
        .light_plan
        .evaluate(&scene.nodes, &worlds, &scene.links)?;
    assert_eq!(lights.len(), 5);
    let rays = scene.rays.as_ref().unwrap();
    let bounds: Vec<_> = rays
        .instances
        .iter()
        .map(|&(d, o)| {
            crate::shadow_envelope::transformed(
                scene.objects[o as usize].model,
                scene.draws[d].bounds,
            )
        })
        .collect();
    let tree = from_scene(&scene, &lights, &bounds, &[], 1.)?;
    let leaf: Vec<_> = tree
        .leaves
        .iter()
        .map(|&n| tree.nodes[n as usize].source)
        .collect();
    assert_eq!(
        leaf.iter()
            .filter(|s| s[0] == Kind::Analytic as u32)
            .count(),
        5
    );
    assert_eq!(
        leaf.iter()
            .filter(|s| s[0] == Kind::Environment as u32)
            .count(),
        1
    );
    assert!(leaf.iter().any(|s| s[0] == Kind::Mesh as u32));
    assert_eq!(
        leaf.iter()
            .filter(|s| s[0] == Kind::Implicit as u32)
            .count(),
        1
    );
    for record in &leaf {
        if record[0] == Kind::Mesh as u32 {
            assert!((record[1] as usize) < rays.instances.len());
        }
    }
    let p = probabilities(&tree, Vec3::ZERO);
    assert!((p.iter().sum::<f64>() - 1.).abs() < 1e-12);
    Ok(())
}
