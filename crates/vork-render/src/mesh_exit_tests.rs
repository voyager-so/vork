use super::*;

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn strand_tree_matches_brute_force_for_bends_caps_and_parallel_rays() -> Result<()> {
    let renderer = crate::test_renderer()?;
    let mut points = Vec::new();
    for strand in 0..2 {
        let x = (strand as f32 - 11.5) * 0.095;
        for i in 0..64 {
            let t = i as f32 / 63.;
            points.push([
                x + 0.10 * (2. * std::f32::consts::PI * t + strand as f32 * 0.12).sin(),
                2. * t,
                -0.15 + 0.05 * (std::f32::consts::PI * t).sin(),
            ]);
        }
    }
    let geometry = crate::curves::tubes(&points, &[0.035; 128], &[64, 64], 12)?;
    let words = crate::hair::geometry_words(&geometry)?;
    let mut rays = Vec::<[[f32; 4]; 3]>::new();
    let mut expected = Vec::new();
    for first_vertex in [0, 64 * 13] {
        let range = geometry.vertices[first_vertex].TEXCOORD_1;
        let header = range[0] as usize * 4;
        let first = words[header];
        let nodes = words[header + 1];
        let triangles = words[header + 2];
        let count = words[header + 3] as usize;
        let corners: &[[ExitVertex; 3]] =
            bytemuck::cast_slice(&words[triangles as usize..triangles as usize + count * 12]);
        let x = if first_vertex == 0 { -1.0925 } else { -0.9975 };
        for triangle in corners {
            let [a, b, c] = triangle.map(|v| Vec3::from(v.value));
            let normal = (b - a).cross(c - a).normalize();
            for light in 0..8 {
                let angle = (light as f32 + 0.25) * std::f32::consts::PI / 8.;
                let tilt: f32 = if light % 2 == 0 { 1.1 } else { 0.45 };
                let q = glam::Quat::from_xyzw(
                    tilt.sin() * angle.cos(),
                    tilt.sin() * angle.sin(),
                    0.,
                    tilt.cos(),
                );
                let d = q * Vec3::Z;
                if normal.dot(d) >= 0. {
                    continue;
                }
                for offset in [-1e-7, 0., 1e-7] {
                    rays.push([
                        ((a + b + c) / 3. + normal * offset).extend(1.).to_array(),
                        d.extend(0.).to_array(),
                        [
                            f32::from_bits(first),
                            f32::from_bits(triangles),
                            f32::from_bits(nodes),
                            f32::from_bits(count as u32),
                        ],
                    ]);
                    expected.push(f32::NAN);
                }
            }
        }
        for y in [-0.1, 0., 0.3, 1., 1.8, 2., 2.1] {
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let p = Vec3::new(x, y, 0.);
                        let d = Vec3::new(dx as f32, dy as f32, dz as f32).normalize();
                        // Independent brute-force oracle, deliberately without bounds.
                        let mut nearest = 1e30_f32;
                        for triangle in corners {
                            let [a, b, c] = triangle.map(|corner| Vec3::from(corner.value));
                            let e1 = b - a;
                            let e2 = c - a;
                            if e1.cross(e2).dot(d) <= 0. {
                                continue;
                            }
                            let q = d.cross(e2);
                            let det = e1.dot(q);
                            if det.abs() < 1e-20 {
                                continue;
                            }
                            let delta = p - a;
                            let u = delta.dot(q) / det;
                            let r = delta.cross(e1);
                            let v = d.dot(r) / det;
                            let distance = e2.dot(r) / det;
                            if u >= -1e-6 && v >= -1e-6 && u + v <= 1. + 1e-6 && distance >= 0. {
                                nearest = nearest.min(distance);
                            }
                        }
                        expected.push(nearest);
                        rays.push([
                            p.extend(1.).to_array(),
                            d.extend(0.).to_array(),
                            [
                                f32::from_bits(first as u32),
                                f32::from_bits(triangles),
                                f32::from_bits(nodes),
                                f32::from_bits(count as u32),
                            ],
                        ]);
                    }
                }
            }
        }
    }
    let source = format!(
        r#"
import mesh_exit;
import vork_constants;
[[vk::binding(0,0)]] StructuredBuffer<uint4> nodes;
struct Ray {{float4 origin;float4 direction;float4 range;}};
[[vk::binding(1,0)]] StructuredBuffer<Ray> rays;
[[vk::binding(2,0)]] RWStructuredBuffer<float2> hits;
struct Geometry : IExitGeometry {{
    uint2 base;
    __init(uint2 base) {{this.base=base;}}
    bool bounded() {{return true;}}
    uint4 word(uint node,uint field) {{return nodes[base.x/4u+node*REG_EXIT_NODE_VEC4S+field];}}
    float3 vertex(uint triangle,uint corner) {{return asfloat(nodes[base.y/4u+triangle*REG_EXIT_TRIANGLE_VEC4S+corner].xyz);}}
}};
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {{
    if (id.x>={count}u) return;
    let ray=rays[id.x];let geometry=Geometry(asuint(ray.range.xy));
    let hit=mesh_exit<Geometry>(geometry,asuint(ray.range.z),
        ray.origin.xyz,ray.direction.xyz,1e30,1e-6);
    var brute=1e30;
    let count=asuint(ray.range.w);
    // Exact pre-acceleration intersection on GPU, including surface roundoff.
    for (uint t=0;t<count;t++) {{
        let a=geometry.vertex(t,0u);
        let e1=geometry.vertex(t,1u)-a;
        let e2=geometry.vertex(t,2u)-a;
        let d=ray.direction.xyz;
        if(dot(cross(e1,e2),d)<=0.) continue;
        let p=cross(d,e2);let det=dot(e1,p);
        if(abs(det)<1e-20) continue;
        let delta=ray.origin.xyz-a;let u=dot(delta,p)/det;
        let q=cross(delta,e1);let v=dot(d,q)/det;let distance=dot(e2,q)/det;
        if(u>=-1e-6 && v>=-1e-6 && u+v<=1.+1e-6 && distance>=0.) brute=min(brute,distance);
    }}
    hits[id.x]=float2(hit.distance,brute);
}}
"#,
        count = rays.len()
    );
    let d = &renderer.device;
    let entries = [
        pipelines::buffer_entry(
            0,
            wgpu::ShaderStages::COMPUTE,
            wgpu::BufferBindingType::Storage { read_only: true },
        ),
        pipelines::buffer_entry(
            1,
            wgpu::ShaderStages::COMPUTE,
            wgpu::BufferBindingType::Storage { read_only: true },
        ),
        pipelines::buffer_entry(
            2,
            wgpu::ShaderStages::COMPUTE,
            wgpu::BufferBindingType::Storage { read_only: false },
        ),
    ];
    let layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &entries,
    });
    let pipeline = medium::test_pipeline(&renderer, &source, &medium::Native::default(), &entries, (64, 1, 1))?;
    let tree_buffer = init(
        d,
        "strand test trees",
        bytemuck::cast_slice(&words),
        wgpu::BufferUsages::STORAGE,
    );
    let ray_buffer = init(
        d,
        "strand test rays",
        bytemuck::cast_slice(&rays),
        wgpu::BufferUsages::STORAGE,
    );
    let bytes = (rays.len() * 8) as u64;
    let output = crate::gpu_memory::buffer(
        d,
        &wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        },
    );
    let readback = crate::gpu_memory::buffer(
        d,
        &wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        },
    );
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            entry(0, tree_buffer.as_entire_binding()),
            entry(1, ray_buffer.as_entire_binding()),
            entry(2, output.as_entire_binding()),
        ],
    });
    let mut encoder = d.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    let submitted = renderer.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    d.poll(wgpu::PollType::Wait {
        submission_index: Some(submitted),
        timeout: None,
    })?;
    rx.recv()??;
    renderer.errors.check()?;
    let mapped = readback.get_mapped_range(..);
    let actual: &[[f32; 2]] = bytemuck::cast_slice(&mapped);
    for (i, (&[actual, brute], &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            ((actual - brute).abs() <= 1e-5 * brute.max(1.))
                && (expected.is_nan() || (actual - expected).abs() <= 1e-5 * expected.max(1.)),
            "ray {i}: tree {actual}, GPU brute {brute}, CPU {expected}"
        );
    }
    Ok(())
}
