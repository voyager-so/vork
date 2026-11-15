// #include "working-color.wgsl"

struct DrawUniforms {
  model: mat4x4<f32>,
  color: vec4<f32>,
  params: vec4<f32>,   // x=zDepth, y=opacity, z=blur(px), w=stroke(px) — z/w solid analytic only
  pageClip: vec4<f32>, // x, y, w, h in world space
  // Node distort warp: rows of the local-px 3×3 homography. distort0.w is the
  // enabled flag (1 = warp); disabled draws carry identity rows.
  distort0: vec4<f32>,
  distort1: vec4<f32>,
  distort2: vec4<f32>,
};

// Warp a local-space position through the node's distort homography, returning the
// HOMOGENEOUS result (x·w, y·w, w). Callers keep w through the model/frame transforms
// (both are affine in xy with last row [0,0,0,1]) and let the hardware perspective
// divide + perspective-correct varying interpolation do the rest — this is what makes
// glyph-space (em) varyings exact per fragment across a warped quad. Validation only
// proves w > 0 inside the node's box; glyph geometry that overflows the box (bearings,
// descenders) could in principle cross the horizon (w ≤ 0) under an extreme warp and
// get hardware-clipped — accepted: plausible matrices keep the horizon far outside.
fn vectorWarpLocal(local: vec2<f32>) -> vec3<f32> {
  if (draw.distort0.w < 0.5) {
    return vec3<f32>(local, 1.0);
  }
  let p = vec3<f32>(local, 1.0);
  return vec3<f32>(dot(draw.distort0.xyz, p), dot(draw.distort1.xyz, p), dot(draw.distort2.xyz, p));
}
@group(0) @binding(0) var<uniform> frame: mat4x4<f32>;
@group(0) @binding(1) var<uniform> draw: DrawUniforms;

// world is the true (post-divide) world position — affine in SCREEN space — so it
// must interpolate WITHOUT perspective correction or page-clip edges bend on warped
// geometry (identical to default interpolation when the warp is disabled, w == 1).
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(linear) world: vec2<f32> };

@vertex fn vs(@location(0) pos: vec2<f32>) -> VSOut {
  var o: VSOut;
  let h = vectorWarpLocal(pos);
  let world = draw.model * vec4<f32>(h.xy, 0.0, h.z);
  o.pos = frame * world;
  o.pos.z = draw.params.x * h.z; // hardware divide restores params.x
  o.world = world.xy / h.z;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  let b = draw.pageClip;
  if (in.world.x < b.x || in.world.x > b.x + b.z || in.world.y < b.y || in.world.y > b.y + b.w) {
    discard;
  }
  let a = draw.color.a * draw.params.y;
  return vec4<f32>(workingRgb(draw.color.rgb), a);
}