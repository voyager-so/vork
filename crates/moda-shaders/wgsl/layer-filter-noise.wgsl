// layer-filter-noise — (native-only; hand-authored, no TS-renderer twin).
//
// Figma's noise effect over a layer-filter capture (moda-engine
// layer_blur/exec.rs). Measured on Figma's own 2x export (the 03-effects
// capture, docs/qa/paste-fidelity.md §2.1): the grain is square cells of
// `size` OUTPUT pixels (Figma keeps the grain at screen size: a 1.5 grain in a
// 2x export is 1.5 export pixels, not 3), and each cell's paint opacity is a
// ramp of its random value:
// a cell with value h paints at (h − t) / (1 − t) above the threshold
// t = 1 − onShare(density), nothing below. Its histogram fits Figma's (24 / 17
// / 5 % of pixels past 12.5 / 37.5 / 75 % strength, against 24 / 17 / 7 %).
//
// Cells are anchored at the node's origin (`frame.zw`), `params.x` texture px
// wide; a pixel takes the cell its centre falls in. Mode (`params.w`):
//   0 mono  — cells paint `color`;
//   1 duo   — cells paint `color`, the rest paints `secondary`
//             (`secondary.a` = 1 when there is one);
//   2 multi — each cell paints a random colour (three more hash channels).
// Paint covers the layer at `params.z` opacity, clipped to the layer's shape:
// `coverageTexture`'s alpha (the node's silhouette at its opacity, so grain
// shows over a transparent or missing fill, as in Figma), or the capture's own
// alpha when the engine binds the capture there too. Over the painted part it
// is source-atop (alpha unchanged); the shape's unpainted rest gains the
// grain's alpha.
//
// The cell id is an integer and the hash is the integer-exact `hashU32`
// (lowbias32, see moda-shaders/examples/hash_gpu_parity.rs), so every backend
// draws the same cells (`noise_cell_hash` in exec.rs is its CPU twin).
// `seed` (params2.x) separates two noises on one node. Colours are the
// frame's own encoding (straight sRGB values).

struct NoiseUniforms {
  // xy: texture size in px, zw: the node's origin in texture px.
  frame: vec4<f32>,
  // x: grain size (texture px), y: density 0..1, z: opacity 0..1, w: mode.
  params: vec4<f32>,
  color: vec4<f32>,
  secondary: vec4<f32>,
  // x: seed, y: the ON threshold of the value noise; zw unused.
  params2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> noise: NoiseUniforms;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;
@group(0) @binding(3) var coverageTexture: texture_2d<f32>;

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

// The value of integer lattice point (x, y) for channel `salt`, in [0, 1).
// Mirrored bit-for-bit by `noise_cell_hash` in moda-engine layer_blur/exec.rs.
fn cellHash(cell: vec2<i32>, salt: u32) -> f32 {
  return hashToUnit(hashU32(bitcast<u32>(cell.x) ^ hashU32(bitcast<u32>(cell.y) ^ hashU32(salt))));
}

// A cell's paint opacity: its value's ramp above the threshold `t`.
fn cellAlpha(cell: vec2<i32>, salt: u32, t: f32) -> f32 {
  return clamp((cellHash(cell, salt) - t) / max(1.0 - t, 1e-6), 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let src = textureLoad(inputTexture, vec2<i32>(frag.xy), 0);
  let size = max(noise.params.x, 1e-3);
  let salt = u32(noise.params2.x) * 4u;
  let t = noise.params2.y;
  let mode = i32(noise.params.w + 0.5);
  // One sample per pixel, at its centre: a pixel takes the cell its centre is in.
  let cell = vec2<i32>(floor((frag.xy - noise.frame.zw) / size));
  let cover = cellAlpha(cell, salt, t);
  var paint = noise.color.rgb * cover;
  if (mode == 2) {
    paint = vec3<f32>(cellHash(cell, salt + 1u), cellHash(cell, salt + 2u), cellHash(cell, salt + 3u)) * cover;
  }
  var k = noise.params.z * cover;
  var colour = paint / max(cover, 1e-6);
  if (mode == 1 && noise.secondary.a > 0.5) {
    // Duo: the cells over the secondary colour, everywhere.
    colour = paint + noise.secondary.rgb * (1.0 - cover);
    k = noise.params.z;
  }
  // The shape's coverage: at least what the paint covers.
  let shape = max(textureLoad(coverageTexture, vec2<i32>(frag.xy), 0).a, src.a);
  return vec4<f32>(src.rgb * (1.0 - k) + colour * (k * shape), src.a + k * (shape - src.a));
}
