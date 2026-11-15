//! Deterministic per-frame sample positions. No frame or slot state enters the sequence.
use glam::Mat4;

/// Projective acceleration aperture: metres and NDC scale, independent of optics.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Lens {
    pub radius: f32,
    pub screen_scale: [f32;2],
    pub focus_distance: f32,
    pub aperture_blades: u32,
    pub aperture_rotation: f32,
    pub aperture_ratio: f32,
}
impl Lens {
    pub(crate) fn circle_scale(&self, height: f32) -> f32 {
        self.radius*self.screen_scale[1]*height/(2.*self.focus_distance)
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
fn coprime_step(count: usize, mut step: usize) -> usize {
    while gcd(step, count) != 1 {
        step += 1;
    }
    step
}
fn lattice_step(count: usize) -> usize {
    coprime_step(
        count,
        ((count as f32 * crate::constants::SAMPLE_LATTICE_RATIO).round() as usize).max(1),
    )
}
/// Immutable setup for every pixel permutation of a finite sequence. The seed
/// contributes only its upper 16 bits to the coprime search.
pub(crate) fn sequence_parameters(count: usize) -> Vec<u32> {
    let count = count.max(1);
    let mut result = Vec::with_capacity(2 + count.min(65536));
    result.extend([count as u32, lattice_step(count) as u32]);
    result.extend((0..count.min(65536)).map(|seed| coprime_step(count, 1 + seed) as u32));
    result
}

/// Immutable sequence payloads shared by screen effects and image captures.
pub(crate) struct Parameters {
    pub(crate) buffer: wgpu::Buffer,
    offsets: std::collections::BTreeMap<usize,u32>,
    film_offsets: std::collections::BTreeMap<usize,u32>,
}
impl Parameters {
    pub(crate) fn new(d: &wgpu::Device, scene: &vork_scene::Scene, render: &vork_scene::Render) -> Self {
        let mut counts = std::collections::BTreeSet::from([1usize, vork_slang::registry::MOTION_TAPS as usize]);
        let mut film_counts=std::collections::BTreeSet::from([1usize]);
        for settings in std::iter::once(render.clone()).chain(
            scene.images.values().filter_map(|i| i.resolved_render(render))
        ) {
            counts.insert(settings.settings.samples as usize);
            film_counts.insert(settings.settings.samples as usize);
            let (steps, spatial) = crate::plan::sampling_counts(&settings, true);
            counts.insert(steps as usize * spatial as usize);
            film_counts.insert(spatial as usize);
        }
        let (mut words, offsets) = parameter_table(counts, render.settings.seed);
        let mut film_offsets=std::collections::BTreeMap::new();
        for count in film_counts {
            film_offsets.insert(count,words.len() as u32);
            if render.settings.seed == 0 {
                words.extend((0..count).flat_map(|i| offset(i,count).map(f32::to_bits)));
            } else {
                let angle = (sample_hash(render.settings.seed) as f64 / (u32::MAX as f64 + 1.)) * std::f64::consts::TAU;
                let (sin,cos) = (angle as f32).sin_cos();
                words.extend((0..count).flat_map(|i| {
                    let [x,y] = offset(i,count);
                    [x*cos-y*sin,x*sin+y*cos].map(f32::to_bits)
                }));
            }
        }
        let buffer = crate::init(d, "shared screen sampling parameters", bytemuck::cast_slice(&words), wgpu::BufferUsages::STORAGE);
        Self {buffer, offsets, film_offsets}
    }
    /// Resolve uniform view budgets on the host, never in each pixel's shader.
    pub(crate) fn uniform(&self, count: usize, visibility: usize) -> [u32;4] {
        [self.offsets[&count], self.offsets[&(vork_slang::registry::MOTION_TAPS as usize)], self.film_offsets[&visibility], 0]
    }
}
fn parameter_table(counts: std::collections::BTreeSet<usize>, seed: u32) -> (Vec<u32>,std::collections::BTreeMap<usize,u32>) {
    let mut words = Vec::new();
    let mut offsets = std::collections::BTreeMap::new();
    for count in counts {
        offsets.insert(count, words.len() as u32);
        let mut parameters = sequence_parameters(count);
        if seed != 0 && count > 1 {
            let start = 1 + (lattice_step(count) - 1 + seed as usize) % (count - 1);
            parameters[1] = coprime_step(count, start) as u32;
        }
        words.extend(parameters);
    }
    (words, offsets)
}

/// The shared finite-exposure rank-one lattice, matching vork_sampling.slang.
/// No effect owns a sequence: effects supply a stable stream and transform this
/// unit square to their integration domain.
pub(crate) fn sample_lattice(index: usize, count: usize) -> [f32; 2] {
    let count=count.max(1);let index=index%count;
    [(index as f32+0.5)/count as f32,
     (((index*lattice_step(count))%count) as f32+0.5)/count as f32]
}
pub(crate) fn sample_hash(mut value: u32) -> u32 {
    let mix=vork_slang::registry::SAMPLE_HASH_MIX;
    value=(value^(value>>16)).wrapping_mul(mix);
    value=(value^(value>>16)).wrapping_mul(mix);
    value^(value>>16)
}
/// CPU reference for the shader `sample_2d` sequence.
#[cfg(test)]
pub(crate) fn sample_2d(pixel: [u32; 2], index: usize, count: usize, stream: u32) -> [f32; 2] {
    let count=count.max(1);
    let spatial=sample_hash(pixel[0].wrapping_mul(vork_slang::registry::SAMPLE_HASH_MIX)^pixel[1]);
    let seed=sample_hash(spatial^sample_hash(stream));
    let step=coprime_step(count,1+(seed>>16) as usize%count);
    let order=((index%count)*step+(seed&65535) as usize)%count;
    let point=sample_lattice(order,count);
    let rotation=[(seed&65535) as f32/65536.,(sample_hash(seed)&65535) as f32/65536.];
    [(point[0]+rotation[0]).fract(),(point[1]+rotation[1]).fract()]
}
/// EEVEE film density: radial Gaussian (sigma 0.284 * 1.5 px), truncated
/// at 1.5 px. Equal-weight importance samples need no film resolve pass.
pub(crate) fn offset(index: usize,count: usize) -> [f32;2] {
    if count<=1 {return [0.,0.];}
    // Opposite pairs keep every finite exposure centered; an odd budget adds
    // one central sample. The half-disk lattice covers radius-squared and angle.
    let index=index%count;
    let pairs=count/2;
    if index == pairs*2 { return [0.;2]; }
    let [u,v]=sample_lattice(index/2,pairs);
    let a=0.5/0.284_f32.powi(2);
    let radius=1.5*(-(1.-u*(1.-(-a).exp())).ln()/a).sqrt();
    // A diagonal two-sample pair gives both screen axes coverage.
    let angle=std::f32::consts::PI*(v-0.25);
    let sign=if index%2==0 {1.} else {-1.};
    [sign*radius*angle.cos(),sign*radius*angle.sin()]
}

/// Analytic CDF of Cycles' separable Blackman–Harris film kernel at width 1.5.
/// Cycles doubles the width internally: support is [-1.5, 1.5] pixels.
fn path_filter_cdf(x: f64) -> f64 {
    let t = std::f64::consts::PI*x/1.5;
    0.5 + (0.35875*x + 1.5/std::f64::consts::PI *
        (0.48829*t.sin()+0.14128*(2.*t).sin()/2.+0.01168*(3.*t).sin()/3.)) / (3.*0.35875)
}

pub(crate) type PathFilterTable = [f32;vork_slang::registry::PATH_FILTER_INTERVALS+1];

/// Inverse CDF, retained once in the installed ray table. Two interpolated lookups
/// per camera ray; no filter evaluation or extra transport samples on the GPU.
pub(crate) fn path_filter_table() -> &'static PathFilterTable {
    static TABLE: std::sync::OnceLock<PathFilterTable> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| std::array::from_fn(|i| {
        let n = vork_slang::registry::PATH_FILTER_INTERVALS;
        if i == 0 { return -1.5; }
        if i == n { return 1.5; }
        let u = i as f64/n as f64;
        let (mut lo,mut hi) = (-1.5,1.5);
        for _ in 0..48 {
            let mid = (lo+hi)*0.5;
            if path_filter_cdf(mid)<u {lo=mid;} else {hi=mid;}
        }
        ((lo+hi)*0.5) as f32
    }))
}

// Global camera dimensions need a non-affine permutation: a rotation or affine
// lattice remapping can leave pixel, shutter and aperture on a lower-dimensional
// lattice. Cycle walking a keyed bijection preserves one visit per stratum for
// arbitrary budgets without tying their ordering to the reconstruction jitter.
fn camera_permutation(index: usize, count: usize, stream: u32) -> usize {
    let count = count.max(1);
    let mask = count.next_power_of_two() - 1;
    let shift = ((count.next_power_of_two().trailing_zeros() / 2) as usize).max(1);
    let mut value = index % count;
    loop {
        for round in 0..4 {
            let key = sample_hash(stream.wrapping_add(round));
            value = (value ^ key as usize) & mask;
            value = value.wrapping_mul((sample_hash(key) | 1) as usize) & mask;
            value ^= value >> shift;
        }
        if value < count { return value; }
    }
}

/// A separate stratified shutter dimension, centered for a one-sample exposure.
pub(crate) fn shutter_index(index:usize,count:usize)->usize {camera_permutation(index,count.max(1),0x73687574)}
#[cfg(test)]
pub(crate) fn shutter_time(index:usize,count:usize)->f32 {
    (shutter_index(index,count) as f32+0.5)/count.max(1) as f32
}


pub(crate) fn projection(
    mut projection: Mat4,
    resolution: [u32; 2],
    index: usize,
    count: usize,
) -> Mat4 {
    if count > 1 {
        let [x, y] = clip_offset(resolution, index, count);
        // A clip-space translation works for perspective and orthographic cameras.
        let mut shift = Mat4::IDENTITY;
        shift.w_axis.x = x;
        shift.w_axis.y = y;
        projection = shift * projection;
    }
    projection
}
/// Sample `index`'s sub-pixel offset as a clip-space translation.
pub(crate) fn clip_offset(resolution: [u32; 2], index: usize, count: usize) -> [f32; 2] {
    let [x, y] = offset(index, count);
    [x * 2. / resolution[0] as f32, y * 2. / resolution[1] as f32]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_selects_a_repeatable_stratified_sequence() {
        let counts = std::collections::BTreeSet::from([64]);
        let a = parameter_table(counts.clone(), 0).0;
        let b = parameter_table(counts.clone(), 7).0;
        assert_ne!(a,b);
        assert_eq!(b,parameter_table(counts,7).0);
        assert_eq!(gcd(b[1] as usize,64),1);
        for count in [1, 3, 64, 4096, 4294967291] {
            let payload = parameter_table(std::collections::BTreeSet::from([count]), u32::MAX).0;
            assert_eq!(gcd(payload[1] as usize,count),1);
            assert!(count == 1 || (payload[1] as usize) < count);
        }
    }
    #[test]
    fn sampled_motion_blur_converges_to_film_convolved_with_shutter_box() {
        // Moving step edge: integrate the radial film independently of shutter time.
        for count in [1024, 4096, 16384] {
            for velocity in [0.8_f32, 1., 1.2] {
                for axis in 0..2 {
                    for edge in [-0.75_f32, -0.5, -0.25, 0., 0.25, 0.5, 0.75] {
                        let coverage = (0..count).filter(|&i| {
                            offset(i, count)[axis] - velocity * (shutter_time(i, count) - 0.5) < edge
                        }).count() as f32 / count as f32;
                        // Independent 2D quadrature of the radial film density,
                        // analytically integrating the uniform shutter dimension.
                        let mut mass = 0.; let mut total = 0.;
                        for y in 0..200 { for x in 0..200 {
                            let px = (x as f32+0.5)*3./200.-1.5;
                            let py = (y as f32+0.5)*3./200.-1.5;
                            let r2 = (px*px+py*py)/2.25;
                            if r2 >= 1. {continue;}
                            let w = (-0.5*r2/0.284_f32.powi(2)).exp();
                            total += w;
                            mass += w*((edge-px)/velocity+0.5).clamp(0.,1.);
                        }}
                        let expected = mass/total;
                        assert!((coverage - expected).abs() < 1.25 / (count as f32).sqrt(),
                            "count={count} v={velocity} axis={axis} edge={edge}: {coverage} vs {expected}");
                    }
                }
            }
        }
    }

    #[test]
    fn film_samples_match_radial_gaussian_mass() {
        let count=65536;
        let a=0.5/0.284_f32.powi(2);
        for radius in [0.2_f32,0.5,0.8,1.2,1.5] {
            let mass=(0..count).filter(|&i| {
                let [x,y]=offset(i,count);
                x*x+y*y <= radius*radius
            }).count() as f32/count as f32;
            let expected=(1.-(-a*(radius/1.5).powi(2)).exp())/(1.-(-a).exp());
            assert!((mass-expected).abs() < 2./count as f32);
        }
    }

    #[test]
    fn path_filter_lookup_matches_blackman_harris_cdf() {
        let table=path_filter_table();
        let n=vork_slang::registry::PATH_FILTER_INTERVALS;
        let at=|i:usize| table[i];
        assert_eq!(at(0),-1.5); assert_eq!(at(n),1.5);
        for i in 0..n {
            assert!(at(i)<=at(i+1));
            assert!((at(i)+at(n-i)).abs()<1e-6);
            for fraction in [0.,0.25,0.5,0.75] {
                let x=at(i) as f64*(1.-fraction)+at(i+1) as f64*fraction;
                let u=(i as f64+fraction)/n as f64;
                assert!((path_filter_cdf(x)-u).abs()<0.0007);
            }
        }
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn shutter_strata_render_convolved_motion_edge() -> anyhow::Result<()> {
        use serde_json::json;
        let renderer = crate::test_renderer()?;
        for velocity in [-1., 1.] {
            // One world unit per pixel; the edge crosses one pixel per shutter.
            let scene = json!({
                "version": 2,
                "composition":{"space":"3d","view":{"program":"std/agx","inputs":{}}},"render": {"dither":false,"resolution": [16,16], "fps": 1, "duration": 1, "background": "transparent", "motion_blur": {"shutter": 1, "position": "center"}, "settings": {"integrator":"raster","samples":256,"reflections":"environment"}},
                "nodes": [
                    {"type": "camera","material":{"program":"std/camera","inputs":{"projection": "ortho","ortho_scale": 16}}},
                    {"type": "mesh", "name": "edge", "geometry": "quad", "materials": ["white"]}
                ],
                "materials": {"white": {"inputs": {"base_weight": 0,
                    "specular_weight": 0, "emission_luminance": 1}}},
                "buffers": {
                    "p": {"type": "vec3", "data": [[-20,-20,-2],[0,-20,-2],[0,20,-2],[-20,20,-2]]},
                    "n": {"type": "vec3", "data": [[0,0,1],[0,0,1],[0,0,1],[0,0,1]]},
                    "i": {"type": "u32", "data": [0,1,2,0,2,3]}
                },
                "geometry": {"quad": {"attributes": {"POSITION": "p", "NORMAL": "n"}, "indices": "i"}},
                "animation": [{"tracks": [{"path": ["nodes","edge","translation"],
                    "keys": [[0,[-0.5*velocity,0,0]],[1,[0.5*velocity,0,0]]],
                    "interpolation": "LINEAR"}]}]
            });
            let mut installed = renderer.install(vork_scene::v2::from_test_value(scene)?, std::path::Path::new("."))?;
            let mut slot = installed.create_frame_slot(&renderer, Default::default())?;
            let rgba = installed.render_frame(&renderer, &mut slot, 0.5)?.rgba;
            // Gaussian film convolved with the one-pixel shutter; outer tails
            // are small and the two edge pixels retain complementary coverage.
            for (x, expected) in [(6,0.999), (7,0.832), (8,0.168), (9,0.001)] {
                let alpha = rgba[(8*16+x)*4+3] as f32 / 255.;
                assert!((alpha - expected).abs() < 0.06,
                    "velocity={velocity} x={x}: alpha={alpha}, expected={expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn camera_dimensions_visit_every_stratum() {
        for count in 1..=1024 {
            for stream in [0x73687574, 0x6c656e73] {
                let mut visited = vec![false; count];
                for i in 0..count {
                    let rank = camera_permutation(i, count, stream);
                    assert!(!visited[rank], "count={count} stream={stream}");
                    visited[rank] = true;
                }
            }
        }
        assert_eq!(shutter_time(0, 1), 0.5);
    }

    #[test]
    fn aperture_is_decorrelated_from_pixel_and_shutter() {
        let count = 16384;
        // A centered disk must stay centered in each pixel/shutter quartile,
        // and each quartile must see the same mean squared aperture radius.
        for dimension in 0..3 {
            let mut sums = [glam::Vec3::ZERO; 4];
            let mut counts = [0; 4];
            for i in 0..count {
                let coordinate = if dimension < 2 { sample_lattice(i, count)[dimension] }
                    else { shutter_time(i, count) };
                let bin = (coordinate * 4.) as usize;
                let disk = lens_disk(i, count);
                sums[bin] += glam::Vec3::new(disk.x, disk.y, disk.length_squared());
                counts[bin] += 1;
            }
            for bin in 0..4 {
                let mean = sums[bin] / counts[bin] as f32;
                assert!(mean.truncate().length() < 0.035 && (mean.z - 0.5).abs() < 0.035,
                    "dimension={dimension} bin={bin}: {mean:?}");
            }
        }
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn shader_sequence_matches_host_for_exposure_wide_budgets() -> anyhow::Result<()> {
        use crate::*;
        let r = crate::test_renderer()?;
        let source = r#"
import vork_sampling;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> output;
[[vk::binding(1,0)]] StructuredBuffer<uint> parameters;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    // Also compare both paths at legal shadow budgets; 10641 (3547*3)
    // exercises a half-integer lattice rounding followed by a coprime search.
    uint counts[6]={4096u,65536u,405504u,16u,10641u,16384u};
    uint indices[5]={0u,1u,65535u,131071u,405503u};
    uint shadow_counts[3]={16u,10641u,16384u};
    let row=id.x/5u;let count=counts[row];let index=indices[id.x%5u];
    let seed=screen_ray_hash(uint2(17u,53u),0u,2u);
    let base=parameters[row];let shadow_count=shadow_counts[row%3u];
    output[id.x]=float4(sample_2d(uint2(17u,53u),index,count,2u),
        sample_2d_precomputed(seed,index,shadow_count,parameters[base+2u+(seed>>16u)%shadow_count],parameters[base+1u]));
}"#;
        let pipeline = medium::test_pipeline(&r, source, &medium::Native::default(), &[
            pipelines::buffer_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage {read_only:false}),
            pipelines::buffer_entry(1, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage {read_only:true}),
        ], (1, 1, 1))?;
        let buffer = |usage| r.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shared sample sequence regression"), size:30*16, usage, mapped_at_creation:false,
        });
        let output=buffer(wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let readback=buffer(wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let mut words=vec![0u32;6];
        for (i,count) in [16,10641,16384,16,10641,16384].into_iter().enumerate() {
            words[i]=words.len() as u32;
            words.extend(sequence_parameters(count));
        }
        let parameters=init(&r.device,"sequence setup regression",bytemuck::cast_slice(&words),wgpu::BufferUsages::STORAGE);
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding()),entry(1,parameters.as_entire_binding())],
        });
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(30,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&readback,0,30*16);
        r.queue.submit([encoder.finish()]);
        let (tx,rx)=std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move |v| {let _=tx.send(v);});
        r.device.poll(wgpu::PollType::Wait {submission_index:None,timeout:None})?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);let points:&[[f32;4]]=bytemuck::cast_slice(&mapped);
        for (row,&count) in [4096,65536,405504,16,10641,16384].iter().enumerate() {
            for (column,index) in [0,1,65535,131071,405503].into_iter().enumerate() {
                let expected=sample_2d([17,53],index,count,2);
                let shadow_expected=sample_2d([17,53],index,[16,10641,16384][row%3],2);
                let expected=[expected[0],expected[1],shadow_expected[0],shadow_expected[1]];
                for axis in 0..4 {assert!((points[row*5+column][axis]-expected[axis]).abs()<1e-6,
                    "count={count}, index={index}: GPU {:?}, host {expected:?}",points[row*5+column]);}
            }
        }
        r.errors.check()?;
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn contiguous_shader_sequence_matches_individual_points() -> anyhow::Result<()> {
        use crate::*;
        let r = crate::test_renderer()?;
        let source=r#"
import vork_sampling;
import vork_constants;
[[vk::binding(0,0)]] RWStructuredBuffer<uint4> output;
[[vk::binding(1,0)]] StructuredBuffer<uint> parameters;
[shader("compute")][numthreads(1,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    uint counts[8]={1u,3u,16u,65u,4096u,65536u,405504u,4294967291u};
    uint starts[6]={0u,1u,31u,65535u,131071u,405503u};
    let count=counts[id.x/6u];let start=starts[id.x%6u];
    let pixel=uint2(17u+id.x,53u);let stream=2u+id.x;
    var sequence=sample_2d_sequence_installed(parameters,parameters[id.x/6u],pixel,start,count,stream);
    var local_sequence=sample_2d_sequence(pixel,start,count,stream);
    let grouped_points=8u*REG_ANALYTIC_LIGHT_SAMPLES;
    let output_base=id.x*(33u+grouped_points);
    for (var j=0u;j<16u;j++) {
        let expected=asuint(sample_2d(pixel,start+j,count,stream));
        output[output_base+j]=uint4(expected,asuint(sample_2d_next(sequence)));
        output[output_base+16u+j]=uint4(expected,asuint(sample_2d_next(local_sequence)));
    }
    // The analytic stream restarts at each covered sample, including sparse,
    // nonmonotonic sample identities. Zero coverage skips an entire group;
    // per-light rejections consume their point even at the group's endpoints.
    uint exposures[6]={1u,3u,16u,65u,4096u,65536u};
    uint covered_samples[8]={start,0u,start+3u,start+1u,start+65u,1u,start+31u,start+7u};
    let source_seed=pixel_hash(id.x);let source_key=pixel^uint2(source_seed,pixel_hash(source_seed));
    let source_count=exposures[id.x%6u]*REG_ANALYTIC_LIGHT_SAMPLES;
    let rotation_bits=screen_ray_hash(source_key,0u,REG_SAMPLING_STREAM_ANALYTIC_SOURCE);
    let rotation=(float(rotation_bits>>16u)+.5)*(1./4294967296.);
    Sample2DSequence source_sequence;var initialized=0u;var advanced=0u;
    for(var next=0u;next<grouped_points;next++) {
        let tap=next/REG_ANALYTIC_LIGHT_SAMPLES;let pick=next%REG_ANALYTIC_LIGHT_SAMPLES;
        let output_index=output_base+32u+next;output[output_index]=uint4(0u);
        if(tap%3u==0u) {continue;}
        let covered_sample=covered_samples[tap];
        if(pick==0u) {
            source_sequence=sample_2d_sequence(source_key,covered_sample*REG_ANALYTIC_LIGHT_SAMPLES,
                source_count,REG_SAMPLING_STREAM_ANALYTIC_SOURCE);
            initialized++;
        }
        var actual=sample_2d_next(source_sequence);advanced++;
        if(pick==0u || pick==3u || pick==REG_ANALYTIC_LIGHT_SAMPLES-1u) {continue;}
        var expected=sample_2d(source_key,covered_sample*REG_ANALYTIC_LIGHT_SAMPLES+pick,
            source_count,REG_SAMPLING_STREAM_ANALYTIC_SOURCE);
        actual.x=frac(actual.x+rotation);expected.x=frac(expected.x+rotation);
        output[output_index]=uint4(asuint(expected),asuint(actual));
    }
    output[output_base+32u+grouped_points]=uint4(5u,5u*REG_ANALYTIC_LIGHT_SAMPLES,initialized,advanced);
}"#;
        let pipeline=medium::test_pipeline(&r,source,&medium::Native::default(),&[
            pipelines::buffer_entry(0,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:false}),
            pipelines::buffer_entry(1,wgpu::ShaderStages::COMPUTE,wgpu::BufferBindingType::Storage {read_only:true}),
        ], (1, 1, 1))?;
        let (payload,offsets)=parameter_table(std::collections::BTreeSet::from([1,3,16,65,4096,65536,405504,4294967291]), 0);
        let mut words:Vec<u32>=offsets.values().map(|offset|offset+8).collect();
        words.extend(payload);
        let parameters=init(&r.device,"installed sequence regression",bytemuck::cast_slice(&words),wgpu::BufferUsages::STORAGE);
        let row_points=33+8*u64::from(vork_slang::renderer_registry::REG_ANALYTIC_LIGHT_SAMPLES);
        let size=48*row_points*16;
        let buffer=|usage|r.device.create_buffer(&wgpu::BufferDescriptor {label:Some("contiguous shared sequence"),size,usage,mapped_at_creation:false});
        let output=buffer(wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let readback=buffer(wgpu::BufferUsages::MAP_READ|wgpu::BufferUsages::COPY_DST);
        let bind=r.device.create_bind_group(&wgpu::BindGroupDescriptor {label:None,layout:&pipeline.get_bind_group_layout(0),entries:&[entry(0,output.as_entire_binding()),entry(1,parameters.as_entire_binding())]});
        let mut encoder=r.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&pipeline);pass.set_bind_group(0,&bind,&[]);pass.dispatch_workgroups(48,1,1);}
        encoder.copy_buffer_to_buffer(&output,0,&readback,0,size);r.queue.submit([encoder.finish()]);
        let (tx,rx)=std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read,..,move|v|{let _=tx.send(v);});
        r.device.poll(wgpu::PollType::Wait {submission_index:None,timeout:None})?;rx.recv()??;
        let mapped=readback.get_mapped_range(..);let points:&[[u32;4]]=bytemuck::cast_slice(&mapped);
        for (index,p) in points.iter().enumerate() {
            assert_eq!(&p[..2],&p[2..],"sequence point bits or group counters {index} changed");
        }
        r.errors.check()?;
        Ok(())
    }

    #[test]
    fn precomputed_permutations_match_every_seed() {
        for count in [1, 2, 3, 32, 65, 257, 4096, 65536, 405504] {
            let parameters=sequence_parameters(count);
            assert_eq!(parameters[1] as usize,lattice_step(count));
            for seed in 0..65536 {
                assert_eq!(parameters[2+seed%count] as usize,coprime_step(count,1+seed%count));
            }
        }
    }

    #[test]
    fn sampled_shadow_products_fit_u32_for_all_valid_budgets() {
        let maximum=vork_scene::MAX_RENDER_SAMPLES as usize*4;
        // A coprime search beginning in 1..=count stops by count+1,
        // which is always coprime. This bounds both products and the offset
        // for every validated budget, without a shader fast/slow threshold.
        let count=maximum as u64;
        assert!((count-1)*(count+1)+(count-1)<=u32::MAX as u64);
    }

    #[test]
    fn shared_sequence_stratifies_every_budget_and_stream() {
        for count in 1..=4096 {
            for stream in [0,2,7,31] {
                let mut strata=[vec![false;count],vec![false;count]];
                for index in 0..count {
                    let p=sample_2d([17,53],index,count,stream);
                    for axis in 0..2 {
                        assert!((0.0..1.0).contains(&p[axis]));
                        // A common rotation translates the stratum boundaries.
                        let first=sample_2d([17,53],0,count,stream)[axis];
                        let rank=(((p[axis]-first).rem_euclid(1.)*count as f32).round() as usize)%count;
                        assert!(!strata[axis][rank], "repeated stratum count={count} stream={stream}");
                        strata[axis][rank]=true;
                    }
                }
            }
        }
    }

    #[test]
    fn shared_sequence_integrates_disk_area_and_polynomials() {
        for count in [257,1024,4096] {
            let mut integral=[0.;3];
            for i in 0..count {
                let [u,v]=sample_2d([17,53],i,count,2);
                integral[0]+=u/count as f32;
                integral[1]+=u*v/count as f32;
                integral[2]+=f32::from(u<0.25)/count as f32;
            }
            for (actual,expected) in integral.into_iter().zip([0.5,0.25,0.25]) {
                assert!((actual-expected).abs()<0.012,"count={count}: {actual} vs {expected}");
            }
        }
    }

    #[test]
    fn filtered_circle_matches_projected_lens_at_finite_focus() {
        let camera = Lens { radius: 0.085/(2.*0.45), screen_scale: [170./36.;2],
            focus_distance: 0.62, aperture_blades: 0,
            aperture_rotation: 0., aperture_ratio: 1. };
        let fov = 2.*(1./camera.screen_scale[1]).atan();
        let projection = Mat4::perspective_rh(fov, 1., 0.01, 100.);
        let height = 384.;
        for depth in [0.4, camera.focus_distance, 0.95, 100.] {
            let point = glam::Vec3::new(0., 0., -depth);
            let nominal = projection.project_point3(point);
            let radius = camera.circle_scale(height)*(1.-camera.focus_distance/depth);
            for index in 0..64 {
                let (p,v) = lens(projection, Mat4::IDENTITY, Some(&camera), index, 64);
                let offset = (p*v).project_point3(point)-nominal;
                let expected = lens_disk(index,64).length()*radius.abs();
                assert!((offset.truncate().length()*height*0.5-expected).abs() < 2e-4,
                    "depth {depth}: projected lens differs from filtered circle");
            }
        }
    }

    #[test]
    fn visibility_lens_clip_translation_matches_full_camera_and_culling_envelope() {
        let camera=Lens {radius:0.085/(2.*0.45),screen_scale:[170./36.,170./24.],focus_distance:0.62,aperture_blades:0,aperture_rotation:0.,aperture_ratio:1.};
        let [w,h]=[480,320];let near=0.02;let far=100.;
        let projection=Mat4::perspective_rh(2.*(24_f32/(2.*85.)).atan(),w as f32/h as f32,near,far);
        let scale=camera.circle_scale(h as f32);
        let selected=selection_projection(projection,[w,h],lens_selection_radius(scale,camera.focus_distance,near,far));
        for depth in [near,0.1,0.62,1.,far] {
            for index in 0..64 {
                let p=glam::Vec3::new(0.03,-0.02,-depth).extend(1.);
                let nominal=projection*p;
                let disk=lens_disk(index,64);
                let expected=nominal+glam::Vec4::new(disk.x*2./w as f32,disk.y*2./h as f32,0.,0.)*scale*(nominal.w-camera.focus_distance);
                let (lp,lv)=lens(projection,Mat4::IDENTITY,Some(&camera),index,64);
                assert!(((lp*lv*p)-expected).length()<2e-4);
                // A point at the shifted frustum's right edge stays within the
                // nominal conservative selection frustum for every lens sample.
                let edge=(lp*lv).inverse()*glam::Vec4::new(nominal.w,0.,nominal.z,nominal.w);
                let clip=selected*edge;
                assert!(clip.x.abs()<=clip.w+1e-4);
            }
        }
    }

    #[test]
    fn thin_lens_keeps_focus_and_has_centered_bounded_aperture() {
        let camera = Lens {
            radius: 0.05/(2.*1.4),
            screen_scale: [100./36.,100./24.],
            focus_distance: 2.,
            aperture_blades: 0,
            aperture_rotation: 0.,
            aperture_ratio: 1.,
        };
        let projection = Mat4::perspective_rh(0.8, 1.5, 0.1, 100.);
        let point = glam::Vec3::new(0.3, -0.2, -camera.focus_distance);
        let expected = projection.project_point3(point);
        let limit = camera.radius;
        for count in [2, 3, 4, 5, 8, 65] {
            let mut sum = glam::Vec3::ZERO;
            for i in 0..count {
                let (p, v) = lens(projection, Mat4::IDENTITY, Some(&camera), i, count);
                assert!(((p * v).project_point3(point) - expected).length() < 1e-6);
                let offset = v.inverse().transform_point3(glam::Vec3::ZERO);
                assert!(offset.length() <= limit + 1e-6);
                sum += offset;
            }
            assert!(sum.length() < 1e-6, "biased lens budget {count}");
        }
    }

    #[test]
    fn every_sample_budget_has_centered_film_support() {
        for count in [1, 2, 3, 5, 6, 7, 8, 31, 64, 65] {
            let mut sum = glam::Vec2::ZERO;
            for i in 0..count {
                let p = projection(Mat4::IDENTITY, [2, 2], i, count);
                sum += p.w_axis.truncate().truncate();
                assert_eq!(offset(i,count),offset(i+count,count));
                assert!(p.w_axis.truncate().truncate().length() <= 1.5);
            }
            assert!(sum.length() < 1e-5, "biased budget {count}: {sum}");

        }
    }
}

/// Thin-lens aperture radius in metres; zero for a pinhole.
pub(crate) fn aperture_radius(camera: &Lens) -> f32 {
    camera.radius
}

/// Exposure sample `index`'s shaped aperture point in camera metres (x right, y up).
pub(crate) fn lens_offset(physical: Option<&Lens>, index: usize, count: usize) -> glam::Vec2 {
    let Some(camera) = physical.filter(|p| p.radius>0.) else { return glam::Vec2::ZERO; };
    let mut disk = lens_disk(index, count);
    if camera.aperture_blades >= 3 {
        let n = camera.aperture_blades as f32;
        let sector = std::f32::consts::TAU / n;
        let coordinate = disk.y.atan2(disk.x).rem_euclid(std::f32::consts::TAU) / sector;
        let angle = coordinate.floor() * sector + camera.aperture_rotation;
        let a = glam::Vec2::new(angle.cos(),angle.sin());
        let b = glam::Vec2::new((angle+sector).cos(),(angle+sector).sin());
        disk = a.lerp(b,coordinate.fract()) * disk.length();
    }
    disk.x /= camera.aperture_ratio.max(1.);
    disk.y *= camera.aperture_ratio.min(1.);
    disk * aperture_radius(camera)
}

/// The camera sample's lens: translate the eye by the aperture point and shift
/// the projection so the focus plane stays fixed.
pub(crate) fn lens(
    projection: Mat4,
    view: Mat4,
    physical: Option<&Lens>,
    index: usize,
    count: usize,
) -> (Mat4, Mat4) {
    let Some(camera) = physical.filter(|p| p.radius>0.) else { return (projection, view); };
    let offset = lens_offset(physical, index, count).extend(0.);
    let mut focus_shift = Mat4::IDENTITY;
    focus_shift.w_axis.x = projection.x_axis.x * offset.x / camera.focus_distance;
    focus_shift.w_axis.y = projection.y_axis.y * offset.y / camera.focus_distance;
    (
        focus_shift * projection,
        Mat4::from_translation(-offset) * view,
    )
}

pub(crate) fn lens_disk(index: usize, count: usize) -> glam::Vec2 {
    let index = camera_permutation(index, count, 0x6c656e73);
    if count < 2 || (count % 2 == 1 && index == count / 2) {
        return glam::Vec2::ZERO;
    }
    fn radical(mut value: usize, base: usize) -> f32 {
        let mut result = 0.;
        let mut factor = 1.;
        while value > 0 {
            factor /= base as f32;
            result += (value % base) as f32 * factor;
            value /= base;
        }
        result
    }
    let paired_index = if count % 2 == 1 && index > count / 2 {
        index - 1
    } else {
        index
    };
    let pairs = count / 2;
    let step = lattice_step(pairs);
    let pair = ((paired_index / 2) * step) % pairs + 1;
    let angle = std::f32::consts::TAU * radical(pair, crate::constants::LENS_AZIMUTH_BASE);
    let radius = radical(pair, crate::constants::LENS_RADIUS_BASE).sqrt();
    let point = glam::Vec2::new(angle.cos(), angle.sin()) * radius;
    if paired_index % 2 == 0 { point } else { -point }
}

/// Film samples reach 1.5 pixels; temporal shifts add at most half a pixel.
/// Expand the nominal frustum by that envelope plus the aperture radius; retaining
/// extra instances does not change raster order or coverage. The regular bounds
/// test supplies its existing arithmetic-roundoff margin as well.
pub(crate) fn selection_projection(vp: Mat4, internal: [u32;2], radius: f32) -> Mat4 {
    Mat4::from_scale(glam::Vec3::new(
        1. / (1. + 2. * (radius + 2.) / internal[0] as f32),
        1. / (1. + 2. * (radius + 2.) / internal[1] as f32), 1.)) * vp
}

/// Maximum physical aperture displacement over the camera's visible depth range.
/// Polygonal/anamorphic aperture shaping stays within the unit disk.
pub(crate) fn lens_selection_radius(scale: f32, focus: f32, near: f32, far: f32) -> f32 {
    scale.abs() * (1.-focus/near).abs().max((1.-focus/far).abs())
}
