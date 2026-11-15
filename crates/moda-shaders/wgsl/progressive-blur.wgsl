// progressive-blur — (native-only; hand-authored, no TS-renderer twin).
//
// One direction of a separable Gaussian whose radius varies per pixel: Figma's
// progressive blur. The radius ramps linearly from `radii.x` at `line.xy` to
// `radii.y` at `line.zw` along the start→end line (constant beyond either
// point; a zero-length line uses the end radius). The radii are 3σ reaches
// (a node's blur radius R, σ = R / 2, arrives as 1.5 R). σ = radius / 3, like the
// uniform blur (`backdrop::kernel`). Run H then V; the V pass reads σ at its
// own pixel, the usual separable approximation of a spatially varying blur.
//
// Any radius works at full resolution: the taps span ±3σ and their spacing
// grows past MAX_TAPS per side (spacing ≤ σ/10.7, far below the Gaussian's
// width, and the linear sampler interpolates between texels). A pixel whose
// σ is below 0.2 px passes through untouched.
//
// Positions are texture px from the fragment position (top-left origin), so
// the pass is independent of any UV flip. Premultiplied in, premultiplied out.

struct ProgressiveBlurUniforms {
  // x: direction (0 = horizontal, 1 = vertical), y: unused, zw: texture size in px.
  params: vec4<f32>,
  // xy: start point, zw: end point, in texture px.
  line: vec4<f32>,
  // x: radius at start, y: radius at end, in texture px; zw unused.
  radii: vec4<f32>,
};

@group(0) @binding(0) var<uniform> blur: ProgressiveBlurUniforms;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;

const MAX_TAPS: i32 = 32;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

// The blur radius at `p` (texture px).
fn radiusAt(p: vec2<f32>) -> f32 {
  let d = blur.line.zw - blur.line.xy;
  let len2 = dot(d, d);
  var t = 1.0;
  if (len2 > 1e-6) {
    t = clamp(dot(p - blur.line.xy, d) / len2, 0.0, 1.0);
  }
  return max(mix(blur.radii.x, blur.radii.y, t), 0.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let size = blur.params.zw;
  let p = frag.xy;
  let uv = p / size;
  let sigma = radiusAt(p) / 3.0;
  if (sigma < 0.2) {
    return textureSampleLevel(inputTexture, inputSampler, uv, 0.0);
  }
  let dir = select(vec2<f32>(1.0 / size.x, 0.0), vec2<f32>(0.0, 1.0 / size.y), blur.params.x > 0.5);
  let reach = ceil(3.0 * sigma);
  let taps = min(reach, f32(MAX_TAPS));
  let spacing = reach / taps;
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);
  var acc = vec4<f32>(0.0);
  var wsum = 0.0;
  for (var i = -MAX_TAPS; i <= MAX_TAPS; i++) {
    let fi = f32(i);
    if (abs(fi) > taps) {
      continue;
    }
    let x = fi * spacing;
    let w = exp(-x * x * inv2s2);
    acc += textureSampleLevel(inputTexture, inputSampler, uv + dir * x, 0.0) * w;
    wsum += w;
  }
  return acc / wsum;
}
