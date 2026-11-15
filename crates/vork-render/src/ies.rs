//! LM-63 Type C photometry as a linear angular image. Strength follows Cycles
//! IES Texture: D65 luminous efficacy and no peak normalization.
use anyhow::{Context, Result, ensure};

/// Original angular knots and physically scaled values, shared by exact emitters
/// and the explicitly resampled image adapter.
pub(crate) struct Profile { pub vertical: Vec<f32>, pub horizontal: Vec<f32>, pub radiance: Vec<f32> }
impl Profile {
    pub(crate) fn words(&self) -> Vec<u32> {
        let mut words=vec![self.vertical.len() as u32,self.horizontal.len() as u32];
        words.extend(self.vertical.iter().chain(&self.horizontal).chain(&self.radiance).map(|x|x.to_bits()));
        words
    }
}
pub(crate) fn parse(bytes: &[u8]) -> Result<Profile> {
    let text = std::str::from_utf8(bytes).context("IES must be UTF-8/ASCII")?;
    let mut lines = text.lines();
    let tilt = lines
        .find(|line| line.trim().starts_with("TILT="))
        .context("IES missing TILT declaration")?;
    ensure!(tilt.trim() == "TILT=NONE", "IES supports TILT=NONE only; re-export the profile without a tilt table");
    let values = lines
        .collect::<Vec<_>>()
        .join("\n")
        .split(|c: char| c.is_ascii_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(str::parse::<f32>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("invalid IES numeric data")?;
    ensure!(
        values.len() >= 13 && values.iter().all(|v| v.is_finite()),
        "invalid IES header or nonfinite data"
    );
    let count = |i: usize| -> Result<usize> {
        let n = values[i];
        ensure!(
            n >= 1. && n <= 65536. && n.fract() == 0.,
            "invalid IES angle count"
        );
        Ok(n as usize)
    };
    let nv = count(3)?;
    let nh = count(4)?;
    ensure!(values[5] == 1., "IES supports Type C photometry only; re-export the profile as LM-63 Type C");
    ensure!(values[2] >= 0., "negative IES candela multiplier");
    let size = nv.checked_mul(nh).context("IES table size overflow")?;
    ensure!(
        values.len() == 13 + nv + nh + size,
        "IES table size mismatch"
    );
    let multiplier = values[2] * values[10] * values[11] * crate::constants::IES_CANDELA_TO_CYCLES_WATTS;
    ensure!(
        values[10] >= 0. && values[11] >= 0. && multiplier.is_finite(),
        "invalid IES ballast/photometric factor"
    );
    let vertical = &values[13..13 + nv];
    let horizontal = &values[13 + nv..13 + nv + nh];
    let candela = &values[13 + nv + nh..];
    for angles in [vertical, horizontal] {
        ensure!(
            angles.windows(2).all(|p| p[0] < p[1]),
            "IES angles must increase strictly"
        );
    }
    ensure!(
        vertical[0] >= 0. && vertical[nv - 1] <= 180.,
        "IES vertical angles outside 0..180"
    );
    ensure!(
        horizontal[0] == 0. && matches!(horizontal[nh - 1], 0. | 90. | 180. | 360.),
        "unsupported IES horizontal symmetry"
    );
    ensure!(
        candela
            .iter()
            .all(|v| *v >= 0. && (*v * multiplier).is_finite()),
        "invalid IES candela values"
    );
    Ok(Profile { vertical: vertical.to_vec(), horizontal: horizontal.to_vec(), radiance: candela.iter().map(|v|v*multiplier).collect() })
}

pub(crate) fn decode(bytes: &[u8]) -> Result<image::DynamicImage> {
    let profile=parse(bytes)?;
    let vertical=&profile.vertical;let horizontal=&profile.horizontal;let candela=&profile.radiance;
    let nv=vertical.len();let nh=horizontal.len();
    let bracket = |angles: &[f32], angle: f32| {
        let hi = angles.partition_point(|v| *v < angle).min(angles.len() - 1);
        let lo = hi.saturating_sub(1);
        let t = if hi == lo {
            0.
        } else {
            ((angle - angles[lo]) / (angles[hi] - angles[lo])).clamp(0., 1.)
        };
        (lo, hi, t)
    };
    // Texel centres match sample(Image, uv): u=azimuth/360, v=polar/180.
    let [width, height] = crate::constants::IES_IMAGE_SIZE;
    let image = image::Rgb32FImage::from_fn(width, height, |x, y| {
        let polar = (y as f32 + 0.5) * 180. / height as f32;
        if polar < vertical[0] || polar > vertical[nv - 1] {
            return image::Rgb([0.; 3]);
        }
        let mut azimuth = (x as f32 + 0.5) * 360. / width as f32;
        match horizontal[nh - 1] {
            0. => azimuth = 0.,
            90. => {
                azimuth %= 180.;
                if azimuth > 90. {
                    azimuth = 180. - azimuth;
                }
            }
            180. => {
                if azimuth > 180. {
                    azimuth = 360. - azimuth;
                }
            }
            _ => {}
        }
        let (v0, v1, vt) = bracket(vertical, polar);
        let (h0, h1, ht) = bracket(horizontal, azimuth);
        let row = |h| candela[h * nv + v0] * (1. - vt) + candela[h * nv + v1] * vt;
        let value = row(h0) * (1. - ht) + row(h1) * ht;
        image::Rgb([value; 3])
    });
    Ok(image::DynamicImage::ImageRgb32F(image))
}

#[cfg(test)]
mod tests {
    #[test]
    fn constant_profile_keeps_cycles_strength_and_both_photometric_factors() {
        let input=b"IESNA:LM-63-2002\nTILT=NONE\n1 1000 2 2 1 1 1 0 0 0 0.5 3 10\n0 180\n0\n100 100\n";
        let image=super::decode(input).unwrap().into_rgb32f();
        let expected=300. * (4. * std::f32::consts::PI / 177.83);
        for p in image.pixels() { assert!((p.0[0]-expected).abs()<1e-4); }
    }
    #[test]
    fn source_knots_preserve_narrow_lobes_and_absolute_strength() {
        let profile=super::parse(include_bytes!("../tests/fixtures/emitter-knot.ies")).unwrap();
        assert_eq!(profile.vertical,[0.,20.,20.01,20.02,60.,90.,180.]);
        assert_eq!(profile.horizontal,[0.,90.,180.,360.]);
        let words=profile.words();assert_eq!(&words[..2],&[7,4]);
        assert!((profile.radiance[2]-100.*crate::constants::IES_CANDELA_TO_CYCLES_WATTS).abs()<1e-5);
    }
    #[test]
    #[ignore="requires pinned Slang and Vulkan (lavapipe works)"]
    fn exact_knots_and_source_rotations_on_gpu() -> anyhow::Result<()> {
        use crate::*;
        let r = crate::test_renderer()?;
        let profile=super::parse(include_bytes!("../tests/fixtures/emitter-knot.ies"))?;
        let mut data=vec![0u32];data.extend(profile.words());
        let source=r#"
import emitter;
import frame;
import vork_constants;
[[vk::binding(0,0)]] StructuredBuffer<uint> data;
[[vk::binding(1,0)]] RWStructuredBuffer<float4> result;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let i=id.x;
    if(i>=8u) {
        LightValues spot={};spot.position=float4(0.,0.,1.,2.);spot.direction=float4(0.,0.,-1.,cos(REG_PI/6.));spot.radiance.w=cos(REG_PI/18.);
        result[i]=float4(emitter_window(spot,float3(0.),normalize(float3(i==9u ? 0. : 1.,0.,1.))),0.,0.,0.);return;
    }
    var polar=20.01;var azimuth=90.;
    if(i==1u) {polar=20.005;azimuth=45.;}
    if(i==2u) {azimuth=180.;}
    if(i==4u) {polar=20.02;}
    if(i==5u) {polar=180.;}
    if(i==6u) {polar=19.;}
    let theta=polar*REG_PI/180.;let phi=azimuth*REG_PI/180.;
    var outgoing=float3(sin(theta)*cos(phi),sin(theta)*sin(phi),-cos(theta));
    LightValues l={};l.emission.w=asfloat(1u);
    if(i==2u) {l.emission.z=sqrt(.5);}
    if(i==3u) {l.emission.y=sqrt(.5);outgoing=float3(outgoing.z,outgoing.y,-outgoing.x);}
    if(i==7u) {l.emission.w=0.;}
    result[i]=float4(emitter_profile_words(l,outgoing,data),0.,0.,0.);
}
"#;
        let entries=[crate::pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),crate::pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false})];
        let pipeline=crate::medium::test_pipeline(&r,source,&Default::default(),&entries, (1, 1, 1))?;
        let d=&r.device;
        let input=init(d,"source IES knots",bytemuck::cast_slice(&data),wgpu::BufferUsages::STORAGE);
        let output=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:160,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:false});
        let readback=d.create_buffer(&wgpu::BufferDescriptor {label:None,size:160,usage:wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST,mapped_at_creation:false});
        let bind=d.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,input.as_entire_binding()),entry(1,output.as_entire_binding())]});
        let mut encoder=d.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(10,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&readback,0,160);let submission=r.queue.submit([encoder.finish()]);
        let(tx,rx)=std::sync::mpsc::channel();readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});
        d.poll(wgpu::PollType::Wait {submission_index:Some(submission),timeout:None})?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);let values:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        for(i,(value,expected))in values.iter().zip([200.,75.,200.,200.,0.,0.,0.,0.,0.,1.]).enumerate() {
            let expected=if i==7 {1.}else if i>=8 {expected}else{expected*crate::constants::IES_CANDELA_TO_CYCLES_WATTS};
            assert!((value[0]-expected).abs()<0.015,"IES case {i}: {} vs {expected}",value[0]);
        }
        Ok(())
    }
    #[test]
    fn unsupported_photometry_and_tilt_are_explicit_errors() {
        assert!(super::decode(b"IESNA:LM-63-2002\nTILT=INCLUDE\n").is_err());
        assert!(super::decode(b"IESNA:LM-63-2002\nTILT=NONE\n1 1000 1 2 1 2 1 0 0 0 1 1 10\n0 180\n0\n100 100\n").is_err());
    }
}
