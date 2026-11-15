//! Numerical checks of the compiled fibre sampler against analytic energy and moments.
use crate::*;

#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn chiang_samples_preserve_energy_and_longitudinal_moments() -> Result<()> {
    let r = crate::test_renderer()?;
    let native = medium::Native {
        includes: vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../libraries")],
        ..Default::default()
    };
    let source = r#"
import std.hair_bsdf;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> output;
uint hash(uint v) {v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);}
float unit(uint v) {return (float(hash(v)>>8)+.5)/16777216.;}
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    uint index=id.x;uint c=id.y;
    float roughness[4]={.02,.3,.8,1.};
    var h=principled_hair(float3(.5,.25,.08),roughness[c%4],.35,.3,1.55,c<4?0.:.17);
    let t=normalize(float3(.2,1.,-.3));
    let n=normalize(cross(t,float3(1.,0.,0.)));
    let wo=c>=8 && c<12 ? t : normalize(.3*t+sqrt(.91)*n);
    if(c==12) {h.ior=1.;h.absorption_coefficient=float3(1e4);}
    let s=hair_sample(h,t,n,wo,float3(unit(index*3+1),unit(index*3+2),unit(index*3+3)));
    let at=(c*16384u+index)*2u;
    output[at]=float4(s.response/max(s.pdf,1e-20),dot(s.direction,t));
    output[at+1]=float4(hair_uniform_world(h,t,n,wo),length(s.direction));
}"#;
    let entries = [pipelines::buffer_entry(
        0,
        wgpu::ShaderStages::COMPUTE,
        wgpu::BufferBindingType::Storage { read_only: false },
    )];
    let pipeline = medium::test_pipeline(&r, source, &native, &entries, (64, 1, 1))?;
    let d = &r.device;
    let bytes = 13 * 16384 * 2 * 16;
    let output = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hair sample moments"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("hair sample readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[entry(0, output.as_entire_binding())],
    });
    let mut encoder = d.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(256, 13, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |v| {
        let _ = tx.send(v);
    });
    d.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    })?;
    rx.recv()??;
    let mapped = readback.get_mapped_range(..);
    let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    for (case, samples) in values.chunks_exact(16384 * 2).enumerate() {
        let mut mean = [0.; 4];
        for pair in samples.chunks_exact(2) {
            for channel in 0..4 {
                assert!(
                    pair[0][channel].is_finite(),
                    "case {case}: non-finite sample"
                );
                mean[channel] += f64::from(pair[0][channel]) / 16384.;
            }
            assert!(
                (pair[1][3] - 1.).abs() < 2e-5,
                "case {case}: non-unit direction {}",
                pair[1][3]
            );
        }
        for channel in 0..3 {
            let expected = f64::from(samples[1][channel]);
            assert!(
                (mean[channel] - expected).abs() < 0.015 * expected.max(1e-6),
                "case {case} channel {channel}: sampled {} != analytic {expected}",
                mean[channel]
            );
        }
        // For alpha=0, all lobes share the same longitudinal mean factor
        // E[sin(theta_i)] = -sin(theta_o) (coth(1/v)-v), mixed by attenuation.
        // The broadest case is close to isotropic; a uniform-sphere sampler
        // would incorrectly produce this zero moment for every roughness.
        if case == 0 {
            assert!(
                (mean[3] + 0.3).abs() < 0.01,
                "smooth longitudinal mean: {}",
                mean[3]
            );
        }
        if case == 3 {
            assert!(mean[3].abs() < 0.03, "rough longitudinal mean: {}", mean[3]);
        }
    }
    Ok(())
}
