// layer-filter-texture — (native-only; hand-authored, no TS-renderer twin).
//
// Figma's texture effect over a layer-filter capture (moda-engine
// layer_blur/exec.rs): every pixel samples the capture at a displaced
// position, roughening the node's edges. The displacement is a smooth value
// noise with feature size `params.x`, two independent channels for x and y,
// each in [-1, 1) × radius (`params.y`). The lattice is anchored at the node's
// origin (`frame.zw`), so the grain sticks to the node under any camera.
//
// clipToShape (`params.z` = 1) keeps the node's own outline, as Figma does: the
// result has the UNdisplaced capture's coverage, and where the displaced
// sample reaches past the outline (it is partly transparent) the node's own
// colour fills in behind it. A solid shape keeps a clean edge; only its
// content is grained.
//
// Lattice values come from the integer-exact `hashU32` (lowbias32, see
// moda-shaders/examples/hash_gpu_parity.rs): the corner ids are integers, so
// every backend draws the same grain. Premultiplied in, premultiplied out.

struct TextureUniforms {
  // xy: texture size in px, zw: the node's origin in texture px.
  frame: vec4<f32>,
  // x: feature size (texture px), y: radius (texture px), z: clip (0 | 1), w: seed.
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> tex: TextureUniforms;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn hashU32(x: u32) -> u32 {
  var h = x;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

// Top 24 bits -> [0, 1): exactly representable in f32, no rounding.
fn hashToUnit(h: u32) -> f32 {
  return f32(h >> 8u) * (1.0 / 16777216.0);
}

// The lattice value at integer cell (x, y) for channel `salt`, in [0, 1).
fn cellHash(cell: vec2<i32>, salt: u32) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(cell.x) ^ hashU32(bitcast<u32>(cell.y) ^ hashU32(salt))));
}

// Smooth (C1) value noise in [0, 1).
fn valueNoise(q: vec2<f32>, salt: u32) -> f32 {
  let i = floor(q);
  let f = q - i;
  let u = f * f * (3.0 - 2.0 * f);
  let c = vec2<i32>(i);
  let a = cellHash(c, salt);
  let b = cellHash(c + vec2<i32>(1, 0), salt);
  let d = cellHash(c + vec2<i32>(0, 1), salt);
  let e = cellHash(c + vec2<i32>(1, 1), salt);
  return mix(mix(a, b, u.x), mix(d, e, u.x), u.y);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let size = tex.frame.xy;
  let p = frag.xy;
  let q = (p - tex.frame.zw) / max(tex.params.x, 1e-3);
  let seed = u32(tex.params.w) * 2u;
  let n = vec2<f32>(valueNoise(q, seed), valueNoise(q, seed + 1u)) * 2.0 - 1.0;
  let displaced = textureSampleLevel(inputTexture, inputSampler, (p + n * tex.params.y) / size, 0.0);
  if (tex.params.z > 0.5) {
    let own = textureSampleLevel(inputTexture, inputSampler, p / size, 0.0);
    if (own.a <= 0.0) {
      return vec4<f32>(0.0);
    }
    // The displaced colour over the node's own (straight) colour, at the
    // node's own coverage.
    let colour = displaced.rgb + (own.rgb / own.a) * (1.0 - displaced.a);
    return vec4<f32>(colour * own.a, own.a);
  }
  return displaced;
}
