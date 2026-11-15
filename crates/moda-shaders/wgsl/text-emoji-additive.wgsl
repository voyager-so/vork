// The emoji image shader for a text glow's copy of an emoji: emitted light, added to what is
// below like vector-text-analytic-additive's glyph glow. Same source as text-emoji.wgsl; only
// the manifest blend differs (color src-alpha + one, alpha one + one).
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, @location(1) world: vec2f };
@group(0) @binding(0) var<uniform> frame: mat4x4<f32>;
@group(0) @binding(1) var<uniform> model: mat4x4<f32>;
@group(0) @binding(2) var<uniform> color: vec4f; // keep interface parity; used for alpha only if needed
@group(0) @binding(3) var<uniform> params: vec4f; // x=zDepth, y=nodeOpacity
@group(0) @binding(4) var<uniform> pageClip: vec4f; // x, y, w, h in world space
@group(1) @binding(0) var samp: sampler;
@group(1) @binding(1) var texImg: texture_2d<f32>;

@vertex fn vs(@location(0) pos: vec2f, @location(1) uv: vec2f) -> VSOut {
  var o: VSOut;
  let world = model * vec4f(pos, 0.0, 1.0);
  o.pos = frame * world;
  o.pos.z = params.x;
  o.uv = uv;
  o.world = world.xy;
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let bounds = pageClip;
  if (in.world.x < bounds.x || in.world.x > bounds.x + bounds.z || in.world.y < bounds.y || in.world.y > bounds.y + bounds.w) {
    discard;
  }
  let rgba = textureSample(texImg, samp, in.uv);
  let finalAlpha = rgba.a * params.y;
  return vec4f(rgba.rgb, finalAlpha);
}
