// layer-blur-composite — (native-only; hand-authored, NOT extracted
// from the TS renderer, which has no layer-blur implementation).
//
// THIS FILE IS THE CANONICAL SOURCE. The extractor
// (tools/extract-wgsl/extractor.ts §8c) copies it verbatim into
// crates/moda-shaders/wgsl/ and writes its manifest entry, so the drift CI
// gate covers it like every extracted module — edit HERE, then re-run
// `npx tsx native-renderer/tools/extract-wgsl/extract.ts`. (Precedent:
// distortPatch.ts — the extractor is the drift-gated seam where native
// shader source deliberately diverges from the TS renderer.)
//
// Draws one blurred-capture quad in the main scene pass at the diverted
// node's z. The quad is positioned in CANVAS space (rect = the padded capture
// region) and projected by the shared frame view-projection, so it lands
// correctly under any camera; UVs span the capture texture 1:1 over the rect,
// giving the bilinear upsample for reduced-scale captures. Page clip is the
// standard canvas-space fragment test against worldPos (same idiom as
// shape-solid). The capture is premultiplied; the pipeline blend is
// (one, one-minus-src-alpha). Depth is the precomputed linear depth for the
// node's z (depth::linear_depth), depth-write OFF (transparent phase).

struct LayerBlurCompositeUniforms {
  // Canvas-space capture rect [x, y, w, h].
  rect: vec4<f32>,
  // Canvas-space page clip bounds [x, y, w, h].
  clip: vec4<f32>,
  // x = linear depth for the node's z, y = opacity multiplier, zw reserved.
  params: vec4<f32>,
  // Canvas-space rounded clip [x, y, w, h] (a capture shadow's / glow's
  // innermost rounded clipping frame); w or h <= 0 = none.
  rrect: vec4<f32>,
  // Its corner radii, canvas units: top-left, top-right, bottom-right,
  // bottom-left.
  rrectRadii: vec4<f32>,
};

@group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>;
@group(1) @binding(0) var<uniform> composite: LayerBlurCompositeUniforms;
@group(1) @binding(1) var captureSampler: sampler;
@group(1) @binding(2) var captureTexture: texture_2d<f32>;

// The node's clip mask (a clip-mask group above it), the same group-2 binding
// and fragment contract as every path shader: the capture is taken unmasked
// and the mask clips the filtered result, so a blur stays soft inside the
// mask and nothing draws outside it. The inactive group (maskActive = 0)
// leaves the quad unmasked.
struct ClipMaskUniforms {
  maskActive: u32,
  maskOrigin: vec2<f32>,
  maskInvSize: vec2<f32>,
}
@group(2) @binding(0) var clipMaskSampler: sampler;
@group(2) @binding(1) var clipMaskTexture: texture_2d<f32>;
@group(2) @binding(2) var<uniform> clipMask: ClipMaskUniforms;

fn clipMaskCoverage(worldPos: vec2<f32>) -> f32 {
  if (clipMask.maskActive == 0u) {
    return 1.0;
  }
  let uv = (worldPos - clipMask.maskOrigin) * clipMask.maskInvSize;
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    return 0.0;
  }
  return textureSampleLevel(clipMaskTexture, clipMaskSampler, uv, 0.0).r;
}

struct VSOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) worldPos: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
  // Two triangles forming a unit quad in [0,1] (same winding as shape-solid).
  var corner: vec2<f32>;
  switch (vid) {
    case 0u: { corner = vec2<f32>(0.0, 0.0); }
    case 1u: { corner = vec2<f32>(1.0, 0.0); }
    case 2u: { corner = vec2<f32>(0.0, 1.0); }
    case 3u: { corner = vec2<f32>(0.0, 1.0); }
    case 4u: { corner = vec2<f32>(1.0, 0.0); }
    default: { corner = vec2<f32>(1.0, 1.0); }
  }
  let world = composite.rect.xy + corner * composite.rect.zw;
  var clip = frameUniforms * vec4<f32>(world, 0.0, 1.0);
  clip.z = composite.params.x; // precomputed linear depth for the node's z
  var out: VSOut;
  out.position = clip;
  out.uv = corner;
  out.worldPos = world;
  return out;
}

// Signed distance to a rounded box centred on the origin (radii: top-left,
// top-right, bottom-right, bottom-left; y down).
fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
  var r = select(radii.x, radii.y, p.x > 0.0);
  if (p.y > 0.0) {
    r = select(radii.w, radii.z, p.x > 0.0);
  }
  let q = abs(p) - halfSize + r;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

@fragment
fn fs(inp: VSOut) -> @location(0) vec4<f32> {
  // Page clip (canvas space, same test as every other instance).
  let bounds = composite.clip;
  if (inp.worldPos.x < bounds.x ||
      inp.worldPos.x > bounds.x + bounds.z ||
      inp.worldPos.y < bounds.y ||
      inp.worldPos.y > bounds.y + bounds.w) {
    discard;
  }
  // Premultiplied capture × opacity (opacity is 1.0 in the MVP — the captured
  // items already bake node opacity — but the slot keeps the P3 isolation-
  // opacity path a uniform write, not a shader change).
  let color = textureSample(captureTexture, captureSampler, inp.uv) * composite.params.y;
  // Rounded clip, antialiased over one target pixel (the distance's screen
  // derivative), like the frame's own edge.
  let rr = composite.rrect;
  let halfSize = rr.zw * 0.5;
  let d = sdRoundedBox(inp.worldPos - (rr.xy + halfSize), halfSize, composite.rrectRadii);
  let aa = max(fwidth(d), 1e-6);
  let coverage = select(1.0, clamp(0.5 - d / aa, 0.0, 1.0), rr.z > 0.0 && rr.w > 0.0);
  return color * coverage * clipMaskCoverage(inp.worldPos);
}
