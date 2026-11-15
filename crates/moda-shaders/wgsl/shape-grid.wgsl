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


// Constants
const TAU: f32 = 6.28318530718;
const SQRT3_2: f32 = 0.866025404;  // sqrt(3) / 2
const AA_WIDTH: f32 = 0.002;
const DEG60: f32 = 1.0471975512;   // π/3 radians

fn encode_int(v: f32) -> u32 { return u32(v + 0.5); }

// Hash utility for palette index selection (integer-based, avoids expensive sin())
fn hash12(p: vec2<f32>) -> f32 {
  var n = u32(p.x * 127.1 + p.y * 311.7);
  n = (n << 13u) ^ n;
  n = n * (n * n * 15731u + 789221u) + 1376312589u;
  return f32(n & 0x7fffffffu) / f32(0x7fffffff);
}

fn rot2(a: f32) -> mat2x2<f32> { let s = sin(a); let c = cos(a); return mat2x2<f32>(c,-s,s,c); }

struct LatticeInfo {
  cell: vec2<f32>,   // integer cell id (as float)
  local: vec2<f32>,  // 0..1 within cell
}

// Map uv -> square lattice cell/local based on cellSize
fn square_lattice(uv: vec2<f32>, cellSize: f32) -> LatticeInfo {
  let g = uv / max(cellSize, 1e-4);
  let cid = floor(g);
  let fracp = fract(g);
  return LatticeInfo(cid, fracp);
}

// Simple staggered rows lattice approximating hex packing for dot grids
fn hex_lattice(uv: vec2<f32>, cellSize: f32) -> LatticeInfo {
  let cs = max(cellSize, 1e-4);
  let rowHeight = cs * SQRT3_2;
  let gy = uv.y / rowHeight;
  let row = floor(gy);
  let isOdd = i32(row) & 1;
  let offsetX = select(0.0, cs * 0.5, isOdd == 1);
  let gx = (uv.x - offsetX) / cs;
  let col = floor(gx);
  let local = vec2<f32>(fract(gx), fract(gy));
  return LatticeInfo(vec2<f32>(col, row), local);
}

// Triangular lattice via skewed square (fallback to square-like behavior)
fn tri_lattice(uv: vec2<f32>, cellSize: f32) -> LatticeInfo {
  // Skew: shear x by 0.5 along y
  let M = mat2x2<f32>(1.0, 0.5, 0.0, 1.0);
  let p = M * uv;
  return square_lattice(p, cellSize);
}

// Signed distance helpers (working in local 0..1 cell space)
fn sdf_circle(local: vec2<f32>, radius: f32) -> f32 {
  let p = local - vec2<f32>(0.5, 0.5);
  return length(p) - radius;
}

fn sdf_box(local: vec2<f32>, half: vec2<f32>) -> f32 {
  let p = local - vec2<f32>(0.5, 0.5);
  let d = abs(p) - half;
  return max(max(d.x, d.y), 0.0) + min(max(d.x, d.y), 0.0);
}

// Utility: rotate 2D by angle
fn rot(a: f32) -> mat2x2<f32> { let s = sin(a); let c = cos(a); return mat2x2<f32>(c,-s,s,c); }

// Triangular grid outlines via 3 line families (0, +60, -60 degrees)
// Returns distance in the same units as input coordinates to nearest grid line
fn tri_grid_edge_dist(p: vec2<f32>, spacing: f32) -> f32 {
  let R60 = rot(DEG60);
  let Rm60 = rot(-DEG60);
  let g0 = abs(fract(p.x / spacing) * spacing - 0.5 * spacing);
  let p1 = R60 * p;
  let g1 = abs(fract(p1.x / spacing) * spacing - 0.5 * spacing);
  let p2 = Rm60 * p;
  let g2 = abs(fract(p2.x / spacing) * spacing - 0.5 * spacing);
  return min(g0, min(g1, g2));
}

// Triangular cell parity using projections onto 3 axes
fn tri_cell_parity(p: vec2<f32>, spacing: f32) -> u32 {
  let R60 = rot(DEG60);
  let Rm60 = rot(-DEG60);
  let u = floor(p.x / spacing + 0.5);
  let v = floor((R60 * p).x / spacing + 0.5);
  let w = floor((Rm60 * p).x / spacing + 0.5);
  let s = i32(u + v + w);
  return u32(abs(s) & 1);
}

// Palette sampling: linear blend across provided colors (RGBA)
fn sample_palette(path: CustomShaderData, t: f32, count: u32) -> vec4<f32> {
  let c = clamp(count, 1u, 8u);
  if (c == 1u) { return path.colorData[0]; }
  let x = clamp(t, 0.0, 1.0) * (f32(c) - 1.0);
  let i0 = u32(floor(x));
  let i1 = min(i0 + 1u, c - 1u);
  let f = fract(x);
  let a = path.colorData[i0];
  let b = path.colorData[i1];
  return mix(a, b, f);
}

// Helper for piecewise-linear interpolation between two stops
fn linear_interp_segment(t: f32, p_start: f32, p_end: f32, v_start: f32, v_end: f32) -> f32 {
  let u = clamp((t - p_start) / max(p_end - p_start, 1e-5), 0.0, 1.0);
  return mix(v_start, v_end, u);
}

fn shadeMain(input: VertexOutput, path: CustomShaderData) -> vec4<f32> {
  let b = path.bounds;
  var uv = (input.localPos - vec2<f32>(b.x, b.y)) / vec2<f32>(b.z, b.w);
  uv = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));

  // Aspect-correct UV: keep shapes isotropic regardless of node aspect ratio
  let aspect = b.z / max(b.w, 1e-4);
  var uvA = uv;
  if (aspect > 1.0) {
    uvA = vec2<f32>((uv.x - 0.5) * aspect + 0.5, uv.y);
  } else {
    uvA = vec2<f32>(uv.x, (uv.y - 0.5) / aspect + 0.5);
  }

  // Params (dots-only)
  let lattice = encode_int(path.shaderParams[0].y);    // 0=square,1=hex,2=tri
  let shape = encode_int(path.shaderParams[0].z);      // 0=circle,1=square
  let time = path.shaderParams[0].w;

  let cellSize = max(path.shaderParams[1].x, 1e-4);
  let jitter = clamp(path.shaderParams[1].z, 0.0, 1.0);
  let strokeWidth = max(path.shaderParams[1].w, 0.0);

  let radiusBase = clamp(path.shaderParams[2].x, 0.0, 1.0);
  let radiusGradAmt = path.shaderParams[2].y;
  // Precomputed cos/sin for radius gradient direction
  let radiusGradDir = vec2<f32>(path.shaderParams[2].z, path.shaderParams[2].w);
  let rippleAmp = path.shaderParams[7].x;

  let rippleWave = max(path.shaderParams[3].x, 1e-4);
  let rippleSpeed = path.shaderParams[3].y;
  // Precomputed cos/sin for ripple direction
  let rippleDir = vec2<f32>(path.shaderParams[3].z, path.shaderParams[3].w);
  let seed = path.shaderParams[7].y;

  let colorMode = encode_int(path.shaderParams[4].x);  // 0=uniform,1=gradient,2=hashed
  // Precomputed cos/sin for color gradient direction
  let gradDir = vec2<f32>(path.shaderParams[4].y, path.shaderParams[7].z);
  let gradRepeat = max(path.shaderParams[4].z, 1e-4);
  let colorCount = encode_int(path.shaderParams[4].w);

  // Size gradient: 0=uniform,1=linear,2=radial
  let sizeGradMode = encode_int(path.shaderParams[5].x);

  // Lattice mapping
  var latt: LatticeInfo;
  switch(i32(lattice)) {
    case 0: { latt = square_lattice(uvA, cellSize); break; }
    case 1: { latt = hex_lattice(uvA, cellSize); break; }
    default: { latt = tri_lattice(uvA, cellSize); break; }
  }

  // Per-cell jitter
  if (jitter > 0.0) {
    let j = (hash12(latt.cell + vec2<f32>(seed, seed*1.713)) - 0.5) * jitter;
    latt.local = fract(latt.local + vec2<f32>(j, j));
  }

  // Radius with gradient + ripple, then optional size gradient fade
  // Using precomputed direction vectors instead of cos/sin calls
  let grad = radiusGradAmt * dot(uvA - vec2<f32>(0.5), radiusGradDir);
  let wave = rippleAmp * sin((dot(uvA, rippleDir) * (TAU / rippleWave)) - time * rippleSpeed);
  var baseRadius = clamp(radiusBase + grad + wave, 0.0, 1.0);
  // Only compute size gradient when mode is non-uniform (skip expensive calculations otherwise)
  if (sizeGradMode != 0u) {
    let baseRadius0 = baseRadius;
    // Compute t for linear or radial modes
    let sgStart = vec2<f32>(path.shaderParams[6].x, path.shaderParams[6].y);
    let sgEnd = vec2<f32>(path.shaderParams[6].z, path.shaderParams[6].w);
    var t = 0.0;
    if (sizeGradMode == 1u) {
      let v = sgEnd - sgStart;
      let vLen2 = max(dot(v, v), 1e-5);
      t = clamp(dot(uvA - sgStart, v) / vLen2, 0.0, 1.0);
    } else {
      // radial: t = distance from center to uv divided by radius
      let center = sgStart;
      let radius = max(length(sgEnd - sgStart), 1e-5);
      t = clamp(length(uvA - center) / radius, 0.0, 1.0);
    }

    // Read stop data
    let stopCount = u32(max(path.shaderParams[8].x, 2.0));
    let p0 = clamp(path.shaderParams[8].y, 0.0, 1.0);
    let v0 = clamp(path.shaderParams[8].z, 0.0, 1.0);
    let p1 = clamp(path.shaderParams[8].w, 0.0, 1.0);
    let v1 = clamp(path.shaderParams[9].x, 0.0, 1.0);
    let p2 = clamp(path.shaderParams[9].y, 0.0, 1.0);
    let v2 = clamp(path.shaderParams[9].z, 0.0, 1.0);
    let p3 = clamp(path.shaderParams[9].w, 0.0, 1.0);
    let v3 = clamp(path.shaderParams[10].x, 0.0, 1.0);

    // Piecewise linear interpolation across stops
    var mult = v0;
    if (stopCount <= 2u) {
      mult = linear_interp_segment(t, p0, p1, v0, v1);
    } else if (stopCount == 3u) {
      if (t <= p1) {
        mult = linear_interp_segment(t, p0, p1, v0, v1);
      } else {
        mult = linear_interp_segment(t, p1, p2, v1, v2);
      }
    } else { // 4 stops
      if (t <= p1) {
        mult = linear_interp_segment(t, p0, p1, v0, v1);
      } else if (t <= p2) {
        mult = linear_interp_segment(t, p1, p2, v1, v2);
      } else {
        mult = linear_interp_segment(t, p2, p3, v2, v3);
      }
    }

    baseRadius = baseRadius0 * mult;
  }

  // Local coordinates within cell
  var local = latt.local;
  // Hex lattice has shorter vertical spacing (rowHeight = cs * sqrt(3)/2), which squashes shapes.
  // Compensate by scaling local.y so SDFs see isotropic coordinates.
  if (lattice == 1u) { // hex
    local = vec2<f32>(local.x, (local.y - 0.5) * SQRT3_2 + 0.5);
  }
  // Shape SDF: branchless selection (both SDFs are cheap, avoids thread divergence)
  let r = baseRadius * 0.5;
  let circle_dist = sdf_circle(local, r);
  let box_dist = sdf_box(local, vec2<f32>(r));
  let dist = select(circle_dist, box_dist, shape == 1u);

  // Stroke/fill alpha from SDF
  var alpha = 1.0 - smoothstep(0.0, AA_WIDTH, dist);
  // Dots-only: regular stroke handling
  if (strokeWidth > 0.0) {
    let inner = 1.0 - smoothstep(0.0, AA_WIDTH, dist + strokeWidth);
    alpha = clamp(alpha - inner, 0.0, 1.0);
  }

  // Color selection
  var col: vec4<f32>;
  switch(i32(colorMode)) {
    case 0: { // uniform
      col = path.colorData[0];
      break;
    }
    case 1: { // gradient along angle (seamless via sinusoidal mapping)
      // Using precomputed direction vector
      let s = dot(uvA - vec2<f32>(0.5), gradDir) * gradRepeat * TAU;
      let t = 0.5 + 0.5 * sin(s);
      col = sample_palette(path, t, colorCount);
      break;
    }
    default: { // hashed per cell
      let h = hash12(latt.cell + vec2<f32>(seed * 3.17, 1.91));
      col = sample_palette(path, h, colorCount);
      break;
    }
  }

  // Background compositing using parameterized RGBA background (colorData[14])
  let bg = path.colorData[14];
  // Alpha: use computed fill alpha only
  let srcA = clamp(alpha * col.a, 0.0, 1.0);
  let outColor = col.rgb * srcA + bg.rgb * bg.a * (1.0 - srcA);
  let outA = srcA + bg.a * (1.0 - srcA);

  // Premultiplied composite preserved from the hand-written shader: outColor is already folded over the
  // background (RGB premultiplied), NOT straight-alpha — do not normalize by outA. The epilogue applies
  // fillColor.a + page-clip + clip-mask and premultiplies on top, reproducing the original output.
  return vec4<f32>(clamp(outColor, vec3<f32>(0.0), vec3<f32>(1.0)), outA);
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
