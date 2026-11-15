// #include "working-color.wgsl"
// layer-filter-shadow — (native-only; hand-authored, no TS-renderer twin).
//
// Drop and inner shadows of ANY node shape, from its coverage (moda-engine
// layer_blur/exec.rs `shadow_capture`). The capture is the node's silhouette
// (every visible fill and its widest stroke painted opaque) rendered into a
// transparent texture; its alpha is the shape's coverage.
//
// Two entry points, both fullscreen triangles over texture px:
//
// `fs_prepare` — one shadow's source, shifted by its offset:
//   drop:  a(p − offset)
//   inner: 1 − a(p − offset)       (the outside of the shape, CSS inset)
//   inner glow from the centre: a(p) (the shape itself, clipped to it below)
// written to alpha (rgb 0). The engine then blurs it with two
// `progressive-blur` passes at a constant radius (σ = blur / 2, any size).
// Samples past the capture clamp to its edge texel: the capture is padded by
// the blur reach, so for a drop shadow that edge is empty (0) and for an inner
// shadow it is the outside (1) — both right.
//
// `fs_composite` — one shadow painted from its blurred source (binding 2)
// and the capture (binding 3), premultiplied, drawn with source-over into the
// result, one shadow after another in list order (effects[0] furthest back):
//   drop:  color × s × (showBehind ? 1 : 1 − a)   (knocked out by the shape)
//   inner: color × s × a                           (clipped to the shape)
// `color.a` is the shadow's opacity × the node's opacity.
//
// `fs_source` — the capture itself (binding 2, premultiplied), drawn
// source-over on the shadows `fs_composite` drew: a filter shadow's result is
// the layer's picture over its own shadows (CSS `filter: drop-shadow()`),
// where a shadow group's is the shadows alone. Reads only `frame.xy`.
//
// Illustrator's Feather over a layer filter capture, two entry points:
//
// `fs_feather_source` — the layer's hard alpha, min(1, 127·a) (Figma's
// `hardAlpha`, the SVG export's too): wherever the layer draws, translucent
// or not, is inside. The engine blurs it (σ = radius / 2).
//
// `fs_feather` — binding 2 is that blur s, binding 3 the layer itself
// (premultiplied), scaled by clamp(2·s − 1, 0, 1): along a straight edge s is
// ½ at the outline, so the layer fades to 0 there and back to full about one
// radius inside (erf(d / (σ√2)) at depth d). A translucent layer keeps its own
// alpha inside. Both read only `frame.xy`.

struct ShadowUniforms {
  // xy: texture size in px, zw: offset in texture px.
  frame: vec4<f32>,
  // Straight rgb + alpha (opacity).
  color: vec4<f32>,
  // x: 1 = inner (clipped to the shape), y: 1 = show behind the node (drop
  // only), z: 1 = an inner glow from the centre (its source is the shape, not
  // its outside), w unused.
  params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> shadow: ShadowUniforms;
@group(0) @binding(1) var inputSampler: sampler;
@group(0) @binding(2) var inputTexture: texture_2d<f32>;
@group(0) @binding(3) var captureTexture: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_prepare(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let size = shadow.frame.xy;
  let uv = (frag.xy - shadow.frame.zw) / size;
  let a = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).a;
  let s = select(a, 1.0 - a, shadow.params.x > 0.5 && shadow.params.z < 0.5);
  return vec4<f32>(0.0, 0.0, 0.0, s);
}

@fragment
fn fs_composite(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy / shadow.frame.xy;
  let s = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).a;
  let a = textureSampleLevel(captureTexture, inputSampler, uv, 0.0).a;
  var k = 1.0 - a;
  if (shadow.params.x > 0.5) {
    k = a;
  } else if (shadow.params.y > 0.5) {
    k = 1.0;
  }
  let alpha = clamp(shadow.color.a * s * k, 0.0, 1.0);
  return vec4<f32>(workingRgb(shadow.color.rgb) * alpha, alpha);
}

@fragment
fn fs_source(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  return textureSampleLevel(inputTexture, inputSampler, frag.xy / shadow.frame.xy, 0.0);
}

@fragment
fn fs_feather_source(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy / shadow.frame.xy;
  let a = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).a;
  return vec4<f32>(0.0, 0.0, 0.0, min(1.0, 127.0 * a));
}

@fragment
fn fs_feather(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = frag.xy / shadow.frame.xy;
  let s = textureSampleLevel(inputTexture, inputSampler, uv, 0.0).a;
  let c = textureSampleLevel(captureTexture, inputSampler, uv, 0.0);
  return c * clamp(2.0 * s - 1.0, 0.0, 1.0);
}
