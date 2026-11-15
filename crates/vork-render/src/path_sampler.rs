//! CPU reference of the path sampler (shaders/path_sampler.slang) and its contracts:
//! net stratification of every power-of-two prefix, joint uniformity of each group
//! over the scramble, decorrelated groups and pixels, faster-than-Monte-Carlo
//! convergence on a smooth integrand, and bit-exact GPU parity.
use crate::*;
use vork_slang::registry::{SAMPLE_HASH_MIX, RNG_PIXEL_MIX};

/// Sobol generator matrices of dimensions 1-3 (dimension 0 is the identity), from
/// Joe and Kuo's primitive polynomials and initial direction numbers
/// (new-joe-kuo-6.21201, dimensions 2-4): degree s, coefficients a, m_1..m_s.
/// Built here from the published table, independently of the shader's recurrences.
const fn sobol_directions() -> [[u32; 32]; 3] {
    const POLYNOMIALS: [(usize, u32, [u32; 3]); 3] = [(1, 0, [1, 0, 0]), (2, 1, [1, 3, 0]), (3, 1, [1, 3, 1])];
    let mut table = [[0u32; 32]; 3];
    let mut d = 0;
    while d < 3 {
        let (s, a, m) = POLYNOMIALS[d];
        let mut k = 0;
        while k < 32 {
            table[d][k] = if k < s { m[k] << (31 - k) } else {
                let mut value = table[d][k - s] ^ (table[d][k - s] >> s);
                let mut j = 1;
                while j < s {
                    if (a >> (s - 1 - j)) & 1 != 0 { value ^= table[d][k - j]; }
                    j += 1;
                }
                value
            };
            k += 1;
        }
        d += 1;
    }
    table
}
const DIRECTIONS: [[u32; 32]; 3] = sobol_directions();

use crate::sampling::sample_hash as pixel_hash;
fn screen_ray_hash(pixel: [u32; 2], index: u32, stream: u32) -> u32 {
    let spatial = pixel_hash(pixel[0].wrapping_mul(SAMPLE_HASH_MIX) ^ pixel[1]);
    pixel_hash(spatial ^ pixel_hash(index.wrapping_mul(RNG_PIXEL_MIX).wrapping_add(stream)))
}
fn owen_reversed(mut x: u32, seed: u32) -> u32 {
    let seed = pixel_hash(seed);
    x ^= x.wrapping_mul(0x3d20adea);
    x = x.wrapping_add(seed);
    x = x.wrapping_mul((seed >> 16) | 1);
    x ^= x.wrapping_mul(0x05526c56);
    x ^= x.wrapping_mul(0x53a22864);
    x
}
fn unit(x: u32) -> f32 { ((x >> 9) as f32 + 0.5) * (1. / 8388608.) }

#[derive(Clone, Copy)]
struct Sampler { index: u32, seed: u32, mask: u32 }
fn sampler(pixel: [u32; 2], index: u32, count: u32, stream: u32) -> Sampler {
    let mask = if count <= 1 { 0 } else { (2u32 << (31 - (count - 1).leading_zeros())).wrapping_sub(1) };
    Sampler { index, seed: screen_ray_hash(pixel, 0, stream), mask }
}
/// The four scrambled 32-bit fractions of one group.
fn group_bits(s: Sampler, group: u32) -> [u32; 4] {
    let seed = pixel_hash(s.seed ^ pixel_hash(group.wrapping_add(1)));
    let index = owen_reversed(s.index.reverse_bits(), seed ^ 0x6c8e9cf5).reverse_bits() & s.mask;
    let mut x = sobol(index).map(u32::reverse_bits);
    x[0] = index;
    let seeds = [0xa511e9b3, 0x63d83595, 0x2f7c4a1b, 0x9e3779b9];
    std::array::from_fn(|c| owen_reversed(x[c], seed ^ seeds[c]).reverse_bits())
}
fn group(s: Sampler, group: u32) -> [f32; 4] { group_bits(s, group).map(unit) }

/// Unscrambled Sobol fractions of dimensions 0-3.
fn sobol(index: u32) -> [u32; 4] {
    let mut x = [index.reverse_bits(), 0, 0, 0];
    for k in 0..32 {
        if index >> k & 1 != 0 {
            for d in 0..3 { x[d + 1] ^= DIRECTIONS[d][k]; }
        }
    }
    x
}

#[test]
fn sobol_table_matches_published_direction_numbers() {
    // Joe-Kuo dimension 4's first direction numbers, as published.
    let published = [0x80000000u32, 0xc0000000, 0x20000000, 0x50000000, 0xf8000000, 0x74000000, 0xa2000000, 0x93000000];
    for (k, v) in published.iter().enumerate() {
        assert_eq!(DIRECTIONS[2][k], *v, "direction {k}");
    }
    // First points of the van der Corput and Sobol dimensions 2-4, in eighths.
    let expected = [[0, 0, 0, 0], [4, 4, 4, 4], [2, 6, 6, 6], [6, 2, 2, 2], [1, 5, 3, 1], [5, 1, 7, 5], [3, 3, 5, 7], [7, 7, 1, 3]];
    for (i, row) in expected.iter().enumerate() {
        assert_eq!(sobol(i as u32).map(|v| v >> 29), *row, "point {i}");
    }
}

/// Owen scrambling and the index shuffle keep every power-of-two prefix a net:
/// each coordinate visits every 1/N interval once, and each 2D pair of a group
/// is a (t,k,2)-net: every elementary interval of area 2^t/N holds 2^t points.
/// t is the unscrambled Sobol pair's worst value up to N=1024: 0 for xy.
#[test]
fn every_power_of_two_prefix_is_a_net() {
    let pairs = [(0usize, 1usize, 0u32), (0, 2, 1), (0, 3, 2), (1, 2, 1), (1, 3, 2), (2, 3, 1)];
    for (pixel, stream, groups) in [([0u32, 0u32], 0u32, 0..3u32), ([913, 77], 0x5eed, 5..7), ([u32::MAX, 3], 1, 40..41)] {
        for g in groups {
            for count in [1u32, 2, 4, 16, 64, 256, 1024] {
                let s = |i| group_bits(Sampler { index: i, ..sampler(pixel, 0, count, stream) }, g);
                let points: Vec<[u32; 4]> = (0..count).map(s).collect();
                let k = count.trailing_zeros();
                for c in 0..4 {
                    let mut seen = vec![false; count as usize];
                    for p in &points {
                        let cell = if k == 0 { 0 } else { (p[c] >> (32 - k)) as usize };
                        assert!(!seen[cell], "pixel {pixel:?} group {g} count {count} coordinate {c}");
                        seen[cell] = true;
                    }
                }
                for &(a, b, t) in &pairs {
                    if k < t { continue; }
                    for split in 0..=(k - t) {
                        let (ka, kb) = (split, k - t - split);
                        let mut hits = vec![0u32; 1usize << (ka + kb)];
                        for p in &points {
                            let ca = if ka == 0 { 0 } else { p[a] >> (32 - ka) };
                            let cb = if kb == 0 { 0 } else { p[b] >> (32 - kb) };
                            hits[((ca << kb) | cb) as usize] += 1;
                        }
                        assert!(hits.iter().all(|&h| h == 1 << t),
                            "pixel {pixel:?} group {g} count {count} pair ({a},{b}) split {ka}x{kb}");
                    }
                }
            }
        }
    }
}

/// Groups at any depth and neighbouring pixels are decorrelated: the sample
/// correlation of every coordinate pair stays at Monte Carlo size.
#[test]
fn groups_and_pixels_are_decorrelated() {
    let count = 64u32;
    let mut series: Vec<Vec<f32>> = Vec::new();
    for pixel in [[10u32, 10u32], [11, 10], [10, 11]] {
        for g in [0u32, 1, 2, 5, 400, 4001] {
            for c in 0..4 {
                let mut values = Vec::new();
                for i in 0..count { values.push(group(Sampler { index: i, ..sampler(pixel, 0, count, 7) }, g)[c]); }
                series.push(values);
            }
        }
    }
    // Pool correlations over many pixels so each pair's mean is well resolved.
    let pixels = 256u32;
    let correlation = |a: usize, b: usize, ga: u32, gb: u32| {
        let mut sum = 0.;
        for p in 0..pixels {
            let s = sampler([p, 3 * p + 1], 0, count, 7);
            for i in 0..count {
                let x = group(Sampler { index: i, ..s }, ga)[a] - 0.5;
                let y = group(Sampler { index: i, ..s }, gb)[b] - 0.5;
                sum += f64::from(x * y) * 12.;
            }
        }
        sum / f64::from(pixels * count)
    };
    // Independent uniforms: correlation has standard deviation 1/sqrt(n).
    let bound = 5. / f64::from(pixels * count).sqrt();
    for (ga, gb) in [(0, 1), (1, 2), (1, 5), (2, 6), (17, 400)] {
        for a in 0..4 { for b in 0..4 {
            let r = correlation(a, b, ga, gb);
            assert!(r.abs() < bound, "groups {ga},{gb} coordinates {a},{b}: correlation {r}");
        }}
    }
    // Different pixels never repeat a sequence.
    for i in 0..series.len() { for j in 0..i { assert_ne!(series[i], series[j]); } }
}

/// A smooth 2D integrand converges faster than N^-1/2, where white noise would not.
#[test]
fn smooth_integrand_converges_faster_than_monte_carlo() {
    let f = |x: f32, y: f32| (3. * x).sin() * (2. * y + x).cos() + x * y;
    // Exact integral over the unit square, by fine midpoint quadrature.
    let n = 2048;
    let exact = (0..n * n).map(|i| {
        let x = (i % n) as f64 + 0.5; let y = (i / n) as f64 + 0.5;
        f64::from(f((x / n as f64) as f32, (y / n as f64) as f32))
    }).sum::<f64>() / (n * n) as f64;
    let rms = |count: u32, g: u32| {
        let mut squared = 0.;
        let pixels = 512u32;
        for p in 0..pixels {
            let s = sampler([p % 32, p / 32], 0, count, 11);
            let mean = (0..count).map(|i| {
                let u = group(Sampler { index: i, ..s }, g);
                f64::from(f(u[0], u[1]))
            }).sum::<f64>() / f64::from(count);
            squared += (mean - exact).powi(2);
        }
        (squared / f64::from(pixels)).sqrt()
    };
    for g in [0u32, 3, 77] {
        let (e16, e256) = (rms(16, g), rms(256, g));
        // Monte Carlo falls 4x from 16 to 256 samples; scrambled Sobol at
        // N^-1.5 falls 64x. Require at least 16x (slope steeper than -1).
        assert!(e16 / e256 > 16., "group {g}: rms {e16} at 16, {e256} at 256");
    }
}

/// Each sample, over the random pixel seed and group, is uniform on the whole
/// [0,1)^4, at every sample count: chi-square on 16x16 cells of every coordinate
/// pair of one group, on 4^4 cells of all four, and on like coordinates of adjacent
/// groups, with a random index below the count. Per-pixel stratification
/// alone cannot show this: correlated scrambles keep every prefix a net.
#[test]
fn every_coordinate_pair_is_jointly_uniform_over_seeds() {
    let draws = 256 * 200u32;
    for count in [1u32, 2, 4, 16, 64] {
        let mut cells = [[0u32; 256]; 9];
        for k in 0..draws {
            let h = pixel_hash(k.wrapping_mul(0x9e3779b9) ^ count);
            let s = sampler([h & 4095, h >> 12], 0, count, 0x5eed);
            let (index, g) = (pixel_hash(h) % count, pixel_hash(h ^ 0x51ed) % 50);
            let u = group(Sampler { index, ..s }, g);
            // Adjacent groups of one pixel and sample (scatter and light of a vertex).
            let next = group(Sampler { index, ..s }, g + 1);
            let c = u.map(|v| ((v * 16.) as usize).min(15));
            for (p, (a, b)) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].into_iter().enumerate() {
                cells[p][c[a] * 16 + c[b]] += 1;
            }
            // The full 4D joint on 4x4x4x4 cells.
            cells[6][c.iter().fold(0, |cell, &v| cell * 4 + v / 4)] += 1;
            let n = next.map(|v| ((v * 16.) as usize).min(15));
            cells[7][c[0] * 16 + n[0]] += 1;
            cells[8][c[2] * 16 + n[2]] += 1;
        }
        // 255 degrees of freedom: mean 255, standard deviation 22.6.
        let expected = f64::from(draws) / 256.;
        for (p, pair) in cells.iter().enumerate() {
            let chi2: f64 = pair.iter().map(|&o| (f64::from(o) - expected).powi(2) / expected).sum();
            assert!(chi2 < 255. + 6. * 22.6, "count {count}, pair {p}: chi-square {chi2}");
        }
    }
}

/// The light group picks with z and places the point with xy: given either
/// light of two, the point must still cover its whole parameter square.
#[test]
fn light_point_is_independent_of_light_pick() {
    for count in [1u32, 2, 4, 16] {
        let (mut picked, mut low_x, mut low_y) = (0u32, 0u32, 0u32);
        for k in 0..40000u32 {
            let h = pixel_hash(k ^ count.wrapping_mul(0x2545f491));
            let s = sampler([h & 1023, h >> 10], 0, count, 3);
            let u = group(Sampler { index: pixel_hash(h) % count, ..s }, 2);
            if u[2] < 0.5 { picked += 1; low_x += u32::from(u[0] < 0.5); low_y += u32::from(u[1] < 0.5); }
        }
        for (name, n) in [("x", low_x), ("y", low_y)] {
            let p = f64::from(n) / f64::from(picked);
            assert!((p - 0.5).abs() < 5. * (0.25 / f64::from(picked)).sqrt(), "count {count}: P({name}<.5 | light 0) = {p}");
        }
    }
}

/// The compiled shader draws exactly the reference's bits.
#[test]
#[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
fn shader_matches_cpu_reference() -> Result<()> {
    let r = crate::test_renderer()?;
    let source = r#"
import path_sampler;
[[vk::binding(0,0)]] RWStructuredBuffer<float4> output;
[shader("compute")][numthreads(64,1,1)] void numerical_test(uint3 id:SV_DispatchThreadID) {
    let index=id.x%300u;let pixel=uint2(id.x/300u*7u,id.x/300u*13u+1u);
    var s=path_sampler(pixel,index,300u,0x5eedu);s.vertex=id.x%5u;
    output[id.x*4u]=path_camera_sample(s);
    output[id.x*4u+1u]=path_scatter(s);
    output[id.x*4u+2u]=float4(path_light(s).xyz,path_environment(s).x);
    output[id.x*4u+3u]=float4(path_triangle(s).x,path_opacity(s),path_subsurface_entry(s),path_roulette(s));
}"#;
    let entries = [pipelines::buffer_entry(0, wgpu::ShaderStages::COMPUTE, wgpu::BufferBindingType::Storage { read_only: false })];
    let pipeline = medium::test_pipeline(&r, source, &medium::Native::default(), &entries, (64, 1, 1))?;
    let d = &r.device;
    let invocations = 64 * 47u64;
    let bytes = invocations * 64;
    let output = d.create_buffer(&wgpu::BufferDescriptor { label: Some("path sampler"), size: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let readback = d.create_buffer(&wgpu::BufferDescriptor { label: Some("path sampler readback"), size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let bind = d.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0),
        entries: &[entry(0, output.as_entire_binding())] });
    let mut encoder = d.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(47, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
    r.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |v| { let _ = tx.send(v); });
    d.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })?;
    rx.recv()??;
    let mapped = readback.get_mapped_range(..);
    let values: &[[f32; 4]] = bytemuck::cast_slice(&mapped);
    for id in 0..invocations as u32 {
        let (index, p) = (id % 300, id / 300);
        let s = sampler([p * 7, p * 13 + 1], index, 300, 0x5eed);
        // Groups: 0 camera, then per vertex scatter, light, mesh, environment, decisions.
        let g = |slot: u32| group(s, 1 + (id % 5) * 5 + slot);
        let at = id as usize * 4;
        assert_eq!(values[at], group(s, 0), "camera, invocation {id}");
        assert_eq!(values[at + 1], g(0), "scatter, invocation {id}");
        let (light, environment, mesh, decisions) = (g(1), g(3), g(2), g(4));
        assert_eq!(values[at + 2], [light[0], light[1], light[2], environment[0]], "light, invocation {id}");
        assert_eq!(values[at + 3], [mesh[2], decisions[0], decisions[1], decisions[2]], "mesh and decisions, invocation {id}");
    }
    r.errors.check()?;
    Ok(())
}
