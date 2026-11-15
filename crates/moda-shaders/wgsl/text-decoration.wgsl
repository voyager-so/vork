// #include "working-color.wgsl"

struct VSOut { @builtin(position) pos: vec4f, @location(0) world: vec2f };
@group(0) @binding(0) var<uniform> frame: mat4x4<f32>;
@group(0) @binding(1) var<uniform> model: mat4x4<f32>;
@group(0) @binding(2) var<uniform> color: vec4f;
@group(0) @binding(3) var<uniform> params: vec4f; // x=zDepth, y=nodeOpacity
@group(0) @binding(4) var<uniform> pageClip: vec4f; // x, y, w, h in world space

@vertex fn vs(@location(0) pos: vec2f, @location(1) _uv: vec2f) -> VSOut {
  var o: VSOut;
  let world = model * vec4f(pos, 0.0, 1.0);
  o.pos = frame * world;
  o.pos.z = params.x; // enforce scene depth
  o.world = world.xy;
  return o;
}

@fragment fn fsClip(in: VSOut) -> @location(0) vec4f {
  let bounds = pageClip;
  if (in.world.x < bounds.x || in.world.x > bounds.x + bounds.z || in.world.y < bounds.y || in.world.y > bounds.y + bounds.w) {
    discard;
  }
  // Straight alpha output, apply node opacity to alpha only (match MSDF pipeline semantics)
  // NOTE: Blur not supported for solid shapes (bullets, decorations) - would require geometry expansion
  let finalAlpha = color.a * params.y;
  return vec4f(workingRgb(color.rgb), finalAlpha);
}
    