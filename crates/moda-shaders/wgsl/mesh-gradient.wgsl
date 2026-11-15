// Common constants

// Mathematical constants
const INVERSE_SQRT_2: f32 = 0.7071067811865476;

// Shader constants
const AA_PIXEL_EXPANSION: f32 = 0.5;
const MIN_BLUR_RADIUS: f32 = 0.01;
const MIN_GRADIENT_STOP_DIFF: f32 = 0.0001;


// Common geometry utilities

// Generate a unit quad (0,0) to (1,1) - used for fill rendering
fn getUnitQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(0.0, 0.0); }
    case 1u: { pos = vec2<f32>(1.0, 0.0); }
    case 2u: { pos = vec2<f32>(0.0, 1.0); }
    case 3u: { pos = vec2<f32>(0.0, 1.0); }
    case 4u: { pos = vec2<f32>(1.0, 0.0); }
    default: { pos = vec2<f32>(1.0, 1.0); }
  }
  return pos;
}

// Generate a normalized quad (-1,-1) to (1,1) - used for stroke rendering
fn getNormalizedQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(-1.0, -1.0); }
    case 1u: { pos = vec2<f32>( 1.0, -1.0); }
    case 2u: { pos = vec2<f32>(-1.0,  1.0); }
    case 3u: { pos = vec2<f32>(-1.0,  1.0); }
    case 4u: { pos = vec2<f32>( 1.0, -1.0); }
    default: { pos = vec2<f32>( 1.0,  1.0); }
  }
  return pos;
}

// Simple 2x2 rotation helper
fn rotate2(p: vec2<f32>, angle: f32) -> vec2<f32> {
  let s = sin(angle);
  let c = cos(angle);
  return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

// Generate UV coordinates for a quad (0,0) to (1,1)
fn getQuadUV(vertexIndex: u32) -> vec2<f32> {
  var uv: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { uv = vec2<f32>(0.0, 0.0); }
    case 1u: { uv = vec2<f32>(1.0, 0.0); }
    case 2u: { uv = vec2<f32>(0.0, 1.0); }
    case 3u: { uv = vec2<f32>(0.0, 1.0); }
    case 4u: { uv = vec2<f32>(1.0, 0.0); }
    default: { uv = vec2<f32>(1.0, 1.0); }
  }
  return uv;
}

// Generate a centered unit quad (-0.5,-0.5) to (0.5,0.5)
fn getCenteredQuadPosition(vertexIndex: u32) -> vec2<f32> {
  var pos: vec2<f32>;
  switch (vertexIndex) {
    case 0u: { pos = vec2<f32>(-0.5, -0.5); }
    case 1u: { pos = vec2<f32>( 0.5, -0.5); }
    case 2u: { pos = vec2<f32>(-0.5,  0.5); }
    case 3u: { pos = vec2<f32>(-0.5,  0.5); }
    case 4u: { pos = vec2<f32>( 0.5, -0.5); }
    default: { pos = vec2<f32>( 0.5,  0.5); }
  }
  return pos;
}

// Common color utilities

// Premultiply RGB by alpha for correct blending with premultiplied alpha mode
fn premultiplyAlpha(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(color.rgb * color.a, color.a);
}

// Convert from sRGB to linear color space
fn srgbToLinear(srgb: vec3<f32>) -> vec3<f32> {
    return pow(srgb, vec3<f32>(2.2));
}

// Convert from linear to sRGB color space
fn linearToSrgb(linear: vec3<f32>) -> vec3<f32> {
    return pow(linear, vec3<f32>(1.0 / 2.2));
}

// Luminance calculation using standard weights
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

// Desaturate a color by a given amount (0 = full color, 1 = grayscale)
fn desaturate(color: vec3<f32>, amount: f32) -> vec3<f32> {
    let gray = vec3<f32>(luminance(color));
    return mix(color, gray, amount);
}

// Apply opacity to a color (non-premultiplied)
fn applyOpacity(color: vec4<f32>, opacity: f32) -> vec4<f32> {
    return vec4<f32>(color.rgb, color.a * opacity);
}

// Blend two colors using normal blend mode
fn blendNormal(base: vec4<f32>, blend: vec4<f32>) -> vec4<f32> {
    let alpha = blend.a + base.a * (1.0 - blend.a);
    let rgb = (blend.rgb * blend.a + base.rgb * base.a * (1.0 - blend.a)) / alpha;
    return vec4<f32>(rgb, alpha);
}


// Main shader code

// Frame uniforms
@group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>;

struct CustomShaderData {
  model: mat4x4<f32>,
  fillColor: vec4<f32>,
  strokeColor: vec4<f32>,
  pathInfo: vec4<f32>,
  bounds: vec4<f32>,
  pageClipBounds: vec4<f32>,
  shaderParams: array<vec4<f32>, 16>,
  colorData: array<vec4<f32>, 16>,
}

@group(1) @binding(0) var<storage, read> paths: array<CustomShaderData>;

const STROKE_DEPTH_OFFSET: f32 = -0.000001490116119384765625;

struct VertexInput {
  @location(0) position: vec2<f32>,
  @location(1) vertexType: f32,
}
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) instanceId: u32,
  @location(1) @interpolate(flat) isStroke: f32,
  @location(2) localPos: vec2<f32>,
  @location(3) worldPos: vec2<f32>,
}

@vertex
fn vertexMain(vertex: VertexInput, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
  var o: VertexOutput;
  let path = paths[instanceIndex];
  let lp = vertex.position;
  let wp = path.model * vec4<f32>(lp.x, lp.y, 0.0, 1.0);
  o.worldPos = wp.xy;
  o.position = frameUniforms * wp;
  let depthOffset = select(0.0, STROKE_DEPTH_OFFSET, vertex.vertexType > 0.5);
  o.position.z = path.pathInfo.w + depthOffset;
  o.instanceId = instanceIndex;
  o.isStroke = vertex.vertexType;
  o.localPos = lp;
  return o;
}


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


      // Baked constants for optimal visual effect
      const TWO_PI: f32 = 6.283185307179586;
      const DISTORTION: f32 = 0.08;
      const SWIRL: f32 = 0.8;
      const SWIRL_MULT: f32 = 3.0;
      const WAVE: f32 = 0.25;
      const WAVE_PHASE: f32 = 0.15 * TWO_PI;
      const WAVE_Y_TIME_SCALE: f32 = 0.2;
      const POSITION_SEED: f32 = 25.0;
      const GRAIN_JITTER: f32 = 0.1;
      const GRAIN_OVERLAY: f32 = 0.24;

      // Spot animation frequency coefficients (module-level for guaranteed single allocation)
      const SPOT_FREQ_X: array<f32, 10> = array<f32, 10>(0.6, 0.9, 1.2, 0.6, 0.9, 1.2, 0.6, 0.9, 1.2, 0.6);
      const SPOT_FREQ_Y: array<f32, 10> = array<f32, 10>(0.8, 1.05, 1.3, 0.8, 1.05, 1.3, 0.8, 1.05, 1.3, 0.8);

      // Integer-exact hash core (lowbias32): bit-exact across WGSL compilers and devices,
      // unlike the old fract(sin(x) * K) hash.
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

      // Procedural position for i-th spot
      fn spotPos(i: u32, t: f32) -> vec2<f32> {
        let fi = f32(i);
        let a = fi * 0.37;
        let phase = t + a;
        return 0.5 + 0.5 * vec2<f32>(sin(phase * SPOT_FREQ_X[i]), cos(phase * SPOT_FREQ_Y[i] + a * 0.5));
      }

      fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
        // NOTE: This shader does not support strokes - designed for full-page backgrounds.
        // Stroke vertices will render as the mesh gradient (which is fine for backgrounds).

        // UV computation
        let b = path.bounds;
        let uv = clamp((input.localPos - b.xy) / b.zw, vec2<f32>(0.001), vec2<f32>(0.999));

        // Params: count, time, grain (minimal params)
        let params0 = path.shaderParams[0];
        let count = u32(params0.x + 0.5);
        let t = params0.y;
        let grain = params0.z;

        // Distortion - single pass, baked params
        let centerVec = uv - 0.5;
        let radius = length(centerVec);
        let center = 1.0 - radius;
        
        var shape_uv = uv;
        let sy = uv.y * uv.y * (3.0 - 2.0 * uv.y); // inline smoothstep(0,1,y)
        let sx = uv.x * uv.x * (3.0 - 2.0 * uv.x); // inline smoothstep(0,1,x)
        shape_uv.x += WAVE * center * cos(t + WAVE_PHASE + 2.0 * sy);
        shape_uv.y += WAVE * center * cos(WAVE_Y_TIME_SCALE * t + WAVE_PHASE + 2.0 * sx);
        shape_uv += DISTORTION * center * vec2<f32>(sin(t + 0.4 * shape_uv.y), cos(t + 0.6 * shape_uv.x)) * 0.5;

        // Swirl - always applied with fixed strength
        var uvRot = shape_uv - 0.5;
        let swirlAngle = SWIRL_MULT * SWIRL * radius;
        let ca = cos(-swirlAngle);
        let sa = sin(-swirlAngle);
        uvRot = vec2<f32>(uvRot.x * ca - uvRot.y * sa, uvRot.x * sa + uvRot.y * ca);
        uvRot += 0.5;

        // Hash for grain (computed once). Framebuffer coords: @builtin(position)
        // is bit-exact across rasterizers (interpolated varyings like localPos are not).
        let grainUV = input.position.xy;
        let h = hashToUnit(hashU32(bitcast<u32>(grainUV.x) ^ hashU32(bitcast<u32>(grainUV.y))));
        let grainOffset = GRAIN_JITTER * grain * (h - 0.5);

        // Color accumulation - branchless, fixed power (inverse distance)
        var color = vec3<f32>(0.0);
        var wsum = 0.0;
        let maxN = min(count, 10u);
        
        for (var i: u32 = 0u; i < maxN; i = i + 1u) {
          let pos = spotPos(i, t + POSITION_SEED) + grainOffset;
          let col = path.colorData[i].rgb;
          let d = uvRot - pos;
          let r2 = dot(d, d);
          // Fixed power=2 (halfPower=1) - simple inverse square falloff
          let w = 1.0 / (r2 + 1e-3);
          color += col * w;
          wsum += w;
        }

        // Normalize color (wsum always > 0 when count > 0)
        color = color / max(wsum, 1e-6);

        // Apply grain overlay
        let grainColor = vec3<f32>(h, fract(h * 127.1), fract(h * 311.7)) - 0.5;
        color = color + grainColor * grain * GRAIN_OVERLAY;

        // Straight-alpha output; the framework epilogue applies fillColor.a + clip/mask coverage.
        return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
      }
      

fn shadeStroke(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  return vec4<f32>(path.strokeColor.rgb, 1.0);
}


fn pageClip(worldPos: vec2<f32>, path: CustomShaderData) -> f32 {
  let clip = path.pageClipBounds;
  return step(clip.x, worldPos.x) * step(worldPos.x, clip.x + clip.z) *
         step(clip.y, worldPos.y) * step(worldPos.y, clip.y + clip.w);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
  let path = paths[input.instanceId];
  let clipIn = pageClip(input.worldPos, path) * clipMaskCoverage(input.worldPos);
  // Page-clip / clip-mask early-out before the (possibly expensive) shade. A discard (not a return)
  // keeps shadeMain below in uniform control flow, so its derivative builtins (fwidth/dpdx/dpdy) stay valid.
  if (clipIn <= 0.0) { discard; }
  let c = shadeMain(input, path);
  // Stroke fragments paint the stroke shader (default: node strokeColor) instead of letting the shader
  // fill bleed across the border. Resolved after shadeMain so the uniformity above holds. Mirrors the fill
  // path: shadeStroke returns intrinsic straight-alpha and the epilogue applies the node strokeColor.a.
  if (input.isStroke > 0.5) {
    let sc = shadeStroke(input, path);
    let strokeAlpha = sc.a * path.strokeColor.a * clipIn;
    if (strokeAlpha <= 0.0) { discard; }
    return premultiplyAlpha(vec4<f32>(sc.rgb, strokeAlpha));
  }
  let baseAlpha = path.fillColor.a * clipIn;
  let outA = c.a * baseAlpha;
  if (outA <= 0.0) { discard; }
  return premultiplyAlpha(vec4<f32>(c.rgb, outA));
}
