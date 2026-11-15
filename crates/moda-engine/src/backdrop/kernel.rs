//! Separable-Gaussian kernel math — line-for-line port of
//! `WebGPUBlurUtils.generateGaussianWeights` / `generateOptimizedGaussianKernel`
//! (`renderers/webgpu/utils/WebGPUBlurUtils.ts:36-117`).
//!
//! Quirk contract (DECISION §1.3): σ = max(radius/3, 0.5); kernel size
//! `min(ceil(6σ) | 1, 31)`; weights normalized; pairs folded into
//! linear-sampling (weight, offset) taps, ≤ 16 taps per side
//! (`MAX_BLUR_KERNEL_SIZE` in `WebGPUBackdropBlurRenderer.ts:5`).
//!
//! The TS stores into `Float32Array` (f32) but computes in f64; mirrored here:
//! all arithmetic in f64, each store rounds to f32.
//!
//! `radius` here is the blur's 3σ reach. A node's blur radius R follows Figma
//! (σ = R / 2), so the planners pass 1.5 R (`BLUR_REACH_PER_RADIUS`).

/// `MAX_BLUR_KERNEL_SIZE` (`WebGPUBackdropBlurRenderer.ts:5`).
pub const MAX_BLUR_KERNEL_SIZE: usize = 16;

#[derive(Debug, Clone, PartialEq)]
pub struct BlurKernel {
    pub weights: Vec<f32>,
    pub offsets: Vec<f32>,
    pub size: usize,
}

/// `generateGaussianWeights(sigma, kernelSize)` — TS stores each weight into a
/// Float32Array as it goes, then normalizes **the f32 values** with f64
/// division; mirror that exactly.
fn generate_gaussian_weights(sigma: f64, kernel_size: usize) -> Vec<f32> {
    let mut weights = vec![0f32; kernel_size];
    let center = (kernel_size / 2) as f64; // Math.floor(kernelSize / 2)
                                           // TS: `weights[i] = weight; sum += weight;` — the sum accumulates the f64
                                           // weights (the Float32Array store is a separate rounding).
    let mut sum = 0f64;
    for (i, w) in weights.iter_mut().enumerate() {
        let x = i as f64 - center;
        let weight = (-(x * x) / (2.0 * sigma * sigma)).exp();
        *w = weight as f32; // Float32Array store
        sum += weight;
    }
    for w in weights.iter_mut() {
        // TS: `weights[i] /= sum` — reads back the f32 store, divides in f64,
        // stores f32.
        *w = ((*w as f64) / sum) as f32;
    }
    weights
}

/// `generateOptimizedGaussianKernel(radius)`.
pub fn generate_optimized_gaussian_kernel(radius: f64) -> BlurKernel {
    // sigma = max(radius / 3, 0.5)
    let sigma = (radius / 3.0).max(0.5);
    // kernelSize = min(ceil(sigma * 6) | 1, 31)  — JS `| 1` on the ceil'd
    // number (int32 coercion + bitwise-or).
    let kernel_size = (((sigma * 6.0).ceil() as i64) | 1).min(31) as usize;

    let base_weights = generate_gaussian_weights(sigma, kernel_size);

    let optimized_size = kernel_size.div_ceil(2); // Math.ceil(kernelSize / 2)
    let mut weights = vec![0f32; optimized_size];
    let mut offsets = vec![0f32; optimized_size];

    let center = kernel_size / 2; // Math.floor

    weights[0] = base_weights[center];
    offsets[0] = 0.0;

    let mut weight_index = 1usize;
    let mut i = 1usize;
    while i <= center {
        if i == center && kernel_size % 2 == 0 {
            // Edge case for even kernel sizes (unreachable: kernelSize is odd
            // by construction — kept for line-fidelity).
            weights[weight_index] = base_weights[center - i];
            offsets[weight_index] = i as f32;
        } else if i + 1 <= center {
            let w1 = base_weights[center - i] as f64;
            let w2 = base_weights[center - i - 1] as f64;
            let combined = w1 + w2;
            weights[weight_index] = combined as f32;
            offsets[weight_index] = (i as f64 + w2 / combined) as f32;
        } else {
            weights[weight_index] = base_weights[center - i];
            offsets[weight_index] = i as f32;
        }
        weight_index += 1;
        i += 2;
    }

    BlurKernel {
        weights: weights[..weight_index].to_vec(),
        offsets: offsets[..weight_index].to_vec(),
        size: weight_index,
    }
}

/// Largest σ (in texels) the single 31-tap kernel represents exactly: its taps
/// reach ±15 texels = 3σ.
pub const SIGMA_DIRECT_MAX: f64 = 5.0;

/// One separable pass of a large backdrop blur: `(weight, offset)` taps in
/// texels, folded like [`BlurKernel`] (tap 0 is the centre, every other tap is
/// sampled at ±offset).
pub type BlurPass = BlurKernel;

/// The passes (per direction) that blur by σ = max(radius / 3, 0.5).
///
/// Up to σ = 5 this is the single production kernel, unchanged. Beyond it the
/// 31-tap kernel would be truncated at ±15 texels (a radius-60 blur drew as
/// as σ ≈ 8.6 instead of 20), so the blur continues as a ladder of passes
/// whose variances add up to σ²: the first is the σ = 5 kernel, each later one
/// samples every `stride` texels with ≤ 16 taps per side at σ ≤ 5 strides. The
/// stride is at most the blur accumulated so far (which already removed the
/// detail a coarser tap spacing would alias) and at most the pass's own σ (so
/// its Gaussian is sampled finely enough). σ = 66 (the 200 px cap) takes 3
/// passes.
pub fn backdrop_blur_passes(radius: f64) -> Vec<BlurPass> {
    let sigma = (radius / 3.0).max(0.5);
    if sigma <= SIGMA_DIRECT_MAX {
        return vec![generate_optimized_gaussian_kernel(radius)];
    }
    let mut passes = vec![generate_optimized_gaussian_kernel(SIGMA_DIRECT_MAX * 3.0)];
    let mut acc = SIGMA_DIRECT_MAX;
    while sigma * sigma - acc * acc > 1e-6 {
        let rest = (sigma * sigma - acc * acc).sqrt();
        // A stride wider than the pass's own σ would sample it too coarsely.
        let stride = acc.min(rest).floor().max(1.0);
        let pass_sigma = rest.min(SIGMA_DIRECT_MAX * stride);
        passes.push(strided_gaussian_pass(pass_sigma / stride, stride));
        acc = (acc * acc + pass_sigma * pass_sigma).sqrt();
    }
    passes
}

/// A Gaussian of `sigma_steps` (in units of `stride` texels) sampled at every
/// `stride`-th texel, 16 taps per side, normalised.
fn strided_gaussian_pass(sigma_steps: f64, stride: f64) -> BlurPass {
    let n = MAX_BLUR_KERNEL_SIZE;
    let raw: Vec<f64> = (0..n)
        .map(|i| (-((i * i) as f64) / (2.0 * sigma_steps * sigma_steps)).exp())
        .collect();
    let sum = raw[0] + 2.0 * raw[1..].iter().sum::<f64>();
    BlurKernel {
        weights: raw.iter().map(|w| (w / sum) as f32).collect(),
        offsets: (0..n).map(|i| (i as f64 * stride) as f32).collect(),
        size: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference values produced by running the production TS in Node:
    /// ```text
    /// npx tsx -e "import { WebGPUBlurUtils } from './renderers/webgpu/utils/WebGPUBlurUtils';
    ///   for (const r of [0.5, 24, 60]) console.log(JSON.stringify(
    ///     WebGPUBlurUtils.generateOptimizedGaussianKernel(r)))"
    /// ```
    /// Exact f32 equality is expected (same f64 math, f32 stores).
    fn assert_kernel(radius: f64, weights: &[f32], offsets: &[f32]) {
        let k = generate_optimized_gaussian_kernel(radius);
        assert_eq!(k.size, weights.len(), "size for r={radius}");
        for i in 0..weights.len() {
            assert!(
                (k.weights[i] - weights[i]).abs() <= 1e-7,
                "r={radius} weight[{i}]: {} vs TS {}",
                k.weights[i],
                weights[i]
            );
            assert!(
                (k.offsets[i] - offsets[i]).abs() <= 1e-6,
                "r={radius} offset[{i}]: {} vs TS {}",
                k.offsets[i],
                offsets[i]
            );
        }
    }

    /// The σ a list of passes adds up to (per direction), from the taps.
    fn passes_sigma(passes: &[BlurPass]) -> f64 {
        passes
            .iter()
            .map(|p| {
                // Linear-sampling taps fold two texels into one offset; their
                // second moment is within a fraction of a texel² of the texels'.
                (0..p.size)
                    .map(|i| {
                        let m = if i == 0 { 1.0 } else { 2.0 };
                        m * p.weights[i] as f64 * (p.offsets[i] as f64).powi(2)
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn small_backdrop_blurs_keep_the_single_production_kernel() {
        for r in [0.5, 4.0, 12.0, 15.0] {
            assert_eq!(
                backdrop_blur_passes(r),
                vec![generate_optimized_gaussian_kernel(r)]
            );
        }
    }

    #[test]
    fn large_backdrop_blurs_reach_the_full_sigma() {
        // Before: one truncated 31-tap pass, whatever the radius.
        let truncated = passes_sigma(&[generate_optimized_gaussian_kernel(60.0)]);
        assert!(
            truncated < 9.0,
            "the single pass tops out near σ 8.6: {truncated}"
        );
        for (r, max_passes) in [(16.0, 2), (24.0, 2), (60.0, 2), (200.0, 3)] {
            let passes = backdrop_blur_passes(r);
            let sigma = passes_sigma(&passes);
            let want = r / 3.0;
            assert!(
                (sigma - want).abs() / want < 0.03,
                "r={r}: σ {sigma} vs {want}"
            );
            assert!(passes.len() <= max_passes, "r={r}: {} passes", passes.len());
            for p in &passes {
                assert!(p.size <= MAX_BLUR_KERNEL_SIZE);
                let total = p.weights[0] as f64
                    + 2.0 * p.weights[1..p.size].iter().map(|w| *w as f64).sum::<f64>();
                assert!((total - 1.0).abs() < 1e-5, "normalised: {total}");
            }
        }
    }

    #[test]
    fn kernel_matches_ts_reference_radius_0_5() {
        assert_kernel(0.5, &[0.786_986_05, 0.106_506_98], &[0.0, 1.0]);
    }

    #[test]
    fn kernel_matches_ts_reference_radius_24() {
        // bu-blur-rect radius 24: sigma 8, kernelSize 31 (cap), 9 taps.
        assert_kernel(
            24.0,
            &[
                0.052_632_59,
                0.103_236_26,
                0.095_507_06,
                0.083_023_55,
                0.067_815_59,
                0.052_049_91,
                0.037_538_12,
                0.025_438_227,
                0.009_074_996,
            ],
            &[
                0.0,
                1.494_140_9,
                3.486_331_5,
                5.478_529,
                7.470_736_5,
                9.462_958,
                11.455_198,
                13.447_46,
                15.0,
            ],
        );
    }

    #[test]
    fn kernel_matches_ts_reference_radius_60() {
        // bu-blur-ellipse radius 60: sigma 20, kernelSize 31 (cap).
        assert_kernel(
            60.0,
            &[
                0.035_511_535,
                0.070_801_59,
                0.069_922_626,
                0.068_367_89,
                0.066_183,
                0.063_430_846,
                0.060_188_606,
                0.056_544_166,
                0.026_805_51,
            ],
            &[
                0.0,
                1.499_062_5,
                3.497_812_5,
                5.496_562_5,
                7.495_312_7,
                9.494_062,
                11.492_813,
                13.491_563,
                15.0,
            ],
        );
    }
}
