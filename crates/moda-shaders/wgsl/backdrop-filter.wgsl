// backdrop-filter — (native-only; hand-authored, no TS-renderer twin).
//
// The backdrop filter kinds of live-nodes stage 7b: grain, vignette, glaze,
// displace and grade, one per pass over a backdrop split's copied region
// (moda-engine backdrop/exec.rs, `BackdropOp::Filter`). Their parameters are
// defined once in moda-scene's `backdrop_kinds.json` (moving to moda-code);
// the planner (backdrop/plan.rs `filter_op`) resolves them to the uniforms
// below in target pixels. The result goes back through the ordinary backdrop
// composite (coverage / SDF, the frame's clip), like pixelate's.
//
// Colour: the layer stack is 8-bit gamma sRGB, premultiplied (motion-model
// §12.7). The kinds whose working space is linear (grain, vignette, grade)
// un-premultiply, decode sRGB to linear light, compute, and encode back at
// their edges; glaze and displace work on the stored values (glaze is the W3C
// blend formula, as CSS mix-blend-mode, so SVG / PDF can write it natively).
//
// Coordinates: `nodeRect` is the node's rotated AABB in sample-texture px and
// `shape` its local frame (rotation, local half-size, mirrored-y sign), the
// convention of backdrop-pixelate / backdrop-refraction. Grain cells and the
// displace lattice live in that local frame, so they move with the node.
//
// Randomness is the integer-exact lowbias32 hash of layer-filter-noise /
// layer-filter-texture (`noise_cell_hash` in layer_blur/exec.rs is the CPU
// twin), so every backend draws the same grain and displacement.

struct FilterUniforms {
  // Rotated AABB of the node in sample-texture px.
  nodeRect: vec4<f32>,
  // x: rotation (rad), y: local half-W (px), z: local half-H (px), w: mirrored-frame Y sign.
  shape: vec4<f32>,
  // x: kind (0 grain, 1 vignette, 2 glaze, 3 displace, 4 grade), y: glaze blend mode
  // (`BlendMode::exact_split_id`), z: glaze stop count, w: glaze paint (0 solid, 1 linear, 2 radial).
  header: vec4<f32>,
  // Per kind:
  //   grain     p0 = (amount, cell px, colour 0|1, seed)
  //   vignette  p0 = (amount, midpoint, roundness, feather), colors[0] = the colour
  //   glaze     p0 = (amount, -, -, -), p1 = (dir.x, dir.y, 1 / gradient length px, -), colors / stops
  //   displace  p0 = (amount px, feature px, seed, -)
  //   grade     p0 = (slope r, slope g, slope b, offset), p1 = (1 / gamma, saturation, -, -)
  p0: vec4<f32>,
  p1: vec4<f32>,
  // The clip box in sample px (the page, the target and every clipping frame above the node): displace samples
  // only inside it, so it never reads what lies outside a clipping frame.
  p2: vec4<f32>,
  // A clipping frame's rounded box (xywh, sample px; w = 0 without one) and its radii [tl, tr, br, bl].
  p3: vec4<f32>,
  p4: vec4<f32>,
  // Straight sRGB colours (glaze stops, the vignette colour).
  colors: array<vec4<f32>, 8>,
  // Glaze stop offsets 0..1, four per vec4.
  stops: array<vec4<f32>, 2>,
};

@group(0) @binding(0) var<uniform> u: FilterUniforms;
@group(0) @binding(1) var uSampler: sampler;
@group(0) @binding(2) var uTexture: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

// ---- randomness (layer-filter-noise / layer-filter-texture) ------------------

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
  let w = f * f * (3.0 - 2.0 * f);
  let c = vec2<i32>(i);
  let a = cellHash(c, salt);
  let b = cellHash(c + vec2<i32>(1, 0), salt);
  let d = cellHash(c + vec2<i32>(0, 1), salt);
  let e = cellHash(c + vec2<i32>(1, 1), salt);
  return mix(mix(a, b, w.x), mix(d, e, w.x), w.y);
}

// ---- colour -------------------------------------------------------------------

fn srgbToLinear(c: vec3<f32>) -> vec3<f32> {
  return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}

// Premultiplied gamma pixel -> straight linear colour.
fn toLinear(p: vec4<f32>) -> vec3<f32> {
  if (p.a <= 0.0) {
    return vec3<f32>(0.0);
  }
  return srgbToLinear(clamp(p.rgb / p.a, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// Straight linear colour at alpha `a` -> premultiplied gamma pixel.
fn fromLinear(c: vec3<f32>, a: f32) -> vec4<f32> {
  return vec4<f32>(linearToSrgb(c) * a, a);
}

// ---- W3C blend modes (Compositing and Blending Level 1 §10; the blend split's, backdrop/blend.rs) ----

fn hardLight(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
  let lo = 2.0 * cb * cs;
  let hi = vec3<f32>(1.0) - 2.0 * (vec3<f32>(1.0) - cb) * (vec3<f32>(1.0) - cs);
  return select(hi, lo, cs <= vec3<f32>(0.5));
}

fn colorDodge1(cb: f32, cs: f32) -> f32 {
  if (cb <= 0.0) { return 0.0; }
  if (cs >= 1.0) { return 1.0; }
  return min(1.0, cb / (1.0 - cs));
}

fn colorBurn1(cb: f32, cs: f32) -> f32 {
  if (cb >= 1.0) { return 1.0; }
  if (cs <= 0.0) { return 0.0; }
  return 1.0 - min(1.0, (1.0 - cb) / cs);
}

fn softLight(cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
  let d = select(sqrt(cb), ((16.0 * cb - 12.0) * cb + 4.0) * cb, cb <= vec3<f32>(0.25));
  let lo = cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb);
  let hi = cb + (2.0 * cs - 1.0) * (d - cb);
  return select(hi, lo, cs <= vec3<f32>(0.5));
}

fn lum(c: vec3<f32>) -> f32 {
  return dot(c, vec3<f32>(0.3, 0.59, 0.11));
}

fn clipColor(cIn: vec3<f32>) -> vec3<f32> {
  var c = cIn;
  let l = lum(c);
  let n = min(c.r, min(c.g, c.b));
  let x = max(c.r, max(c.g, c.b));
  if (n < 0.0) {
    c = l + (c - l) * l / max(l - n, 1e-6);
  }
  if (x > 1.0) {
    c = l + (c - l) * (1.0 - l) / max(x - l, 1e-6);
  }
  return c;
}

fn setLum(c: vec3<f32>, l: f32) -> vec3<f32> {
  return clipColor(c + (l - lum(c)));
}

fn satOf(c: vec3<f32>) -> f32 {
  return max(c.r, max(c.g, c.b)) - min(c.r, min(c.g, c.b));
}

fn setSat(c: vec3<f32>, s: f32) -> vec3<f32> {
  let n = min(c.r, min(c.g, c.b));
  let range = max(c.r, max(c.g, c.b)) - n;
  if (range <= 0.0) {
    return vec3<f32>(0.0);
  }
  return (c - n) * s / range;
}

// Mode ids: `BlendMode::exact_split_id` (seams.rs): 0 normal, 1 multiply, 2 screen, 3.. overlay … luminosity.
fn blendMode(mode: u32, cb: vec3<f32>, cs: vec3<f32>) -> vec3<f32> {
  switch (mode) {
    case 1u: { return cb * cs; }
    case 2u: { return cb + cs - cb * cs; }
    case 3u: { return hardLight(cs, cb); }
    case 4u: { return min(cb, cs); }
    case 5u: { return max(cb, cs); }
    case 6u: { return vec3<f32>(colorDodge1(cb.r, cs.r), colorDodge1(cb.g, cs.g), colorDodge1(cb.b, cs.b)); }
    case 7u: { return vec3<f32>(colorBurn1(cb.r, cs.r), colorBurn1(cb.g, cs.g), colorBurn1(cb.b, cs.b)); }
    case 8u: { return hardLight(cb, cs); }
    case 9u: { return softLight(cb, cs); }
    case 10u: { return abs(cb - cs); }
    case 11u: { return cb + cs - 2.0 * cb * cs; }
    case 12u: { return setLum(setSat(cs, satOf(cb)), lum(cb)); }
    case 13u: { return setLum(setSat(cb, satOf(cs)), lum(cb)); }
    case 14u: { return setLum(cs, lum(cb)); }
    case 15u: { return setLum(cb, lum(cs)); }
    default: { return cs; }
  }
}

// ---- the node's local frame ----------------------------------------------------

fn aabbCenter() -> vec2<f32> {
  return u.nodeRect.xy + u.nodeRect.zw * 0.5;
}

// Sample px -> node-local px (origin at the node's centre, x/y along its own axes).
fn toLocal(p: vec2<f32>) -> vec2<f32> {
  let centered = p - aabbCenter();
  let c = cos(u.shape.x);
  let s = sin(u.shape.x);
  return vec2<f32>(centered.x * c + centered.y * s, (-centered.x * s + centered.y * c) * u.shape.w);
}

// A node-local vector -> a sample-px vector.
fn fromLocalVector(v: vec2<f32>) -> vec2<f32> {
  let c = cos(u.shape.x);
  let s = sin(u.shape.x);
  let y = v.y * u.shape.w;
  return vec2<f32>(v.x * c - y * s, v.x * s + y * c);
}

fn halfSize() -> vec2<f32> {
  return max(u.shape.yz, vec2<f32>(1e-3));
}

// ---- the kinds --------------------------------------------------------------------

// Grain: each cell moves its pixels by ±amount stops (plus a small additive part, so blacks grain too).
fn grain(p: vec2<f32>, src: vec4<f32>) -> vec4<f32> {
  if (src.a <= 0.0) {
    return src;
  }
  let amount = u.p0.x;
  let cell = vec2<i32>(floor(toLocal(p) / max(u.p0.y, 1e-3)));
  let salt = u32(u.p0.w) * 4u;
  var g = vec3<f32>(cellHash(cell, salt));
  if (u.p0.z > 0.5) {
    g = vec3<f32>(g.x, cellHash(cell, salt + 1u), cellHash(cell, salt + 2u));
  }
  g = g * 2.0 - 1.0;
  let c = toLinear(src);
  return fromLinear(c * exp2(amount * g) + amount * 0.02 * g, src.a);
}

// Vignette: a falloff by the (super)elliptical distance from the centre of the node's box.
fn vignette(p: vec2<f32>, src: vec4<f32>) -> vec4<f32> {
  if (src.a <= 0.0) {
    return src;
  }
  let h = halfSize();
  let roundness = u.p0.z;
  var d: f32;
  if (roundness >= 0.0) {
    // 0: an ellipse inscribed in the box; 1: a circle on the shorter side.
    let r = mix(h, vec2<f32>(min(h.x, h.y)), roundness);
    d = length(toLocal(p) / r);
  } else {
    // Squarer: a superellipse with exponent 2 .. 8.
    let q = abs(toLocal(p) / h);
    let n = 2.0 - roundness * 6.0;
    d = pow(pow(q.x, n) + pow(q.y, n), 1.0 / n);
  }
  let mid = u.p0.y;
  let end = max(mix(mid, 1.41421356, u.p0.w), mid + 1e-3);
  let col = u.colors[0];
  let k = u.p0.x * smoothstep(mid, end, d) * col.a;
  let c = mix(toLinear(src), srgbToLinear(col.rgb), k);
  return fromLinear(c, src.a);
}

// The glaze paint at `p`: straight sRGB with alpha.
fn glazePaint(p: vec2<f32>) -> vec4<f32> {
  let count = i32(u.header.z + 0.5);
  let kind = i32(u.header.w + 0.5);
  if (kind == 0 || count < 2) {
    return u.colors[0];
  }
  let local = toLocal(p);
  var t: f32;
  if (kind == 1) {
    t = dot(local, u.p1.xy) * u.p1.z + 0.5;
  } else {
    // CSS radial-gradient's default: an ellipse reaching the farthest corner (half-size × √2).
    t = length(local / (halfSize() * 1.41421356));
  }
  let first = u.stops[0].x;
  if (t <= first) {
    return u.colors[0];
  }
  var prevT = first;
  var prev = u.colors[0];
  for (var i = 1; i < 8; i = i + 1) {
    if (i >= count) {
      break;
    }
    let st = u.stops[i / 4][i % 4];
    let c = u.colors[i];
    if (t <= st) {
      let f = select(0.0, (t - prevT) / (st - prevT), st > prevT);
      return mix(prev, c, f);
    }
    prevT = st;
    prev = c;
  }
  return prev;
}

// Glaze: the paint as the source over the backdrop at opacity `amount`, with the W3C general formula.
fn glaze(p: vec2<f32>, b: vec4<f32>) -> vec4<f32> {
  let paint = glazePaint(p);
  let sa = clamp(u.p0.x * paint.a, 0.0, 1.0);
  let cs = clamp(paint.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
  let cb = select(vec3<f32>(0.0), clamp(b.rgb / b.a, vec3<f32>(0.0), vec3<f32>(1.0)), b.a > 0.0);
  let mixed = blendMode(u32(u.header.y + 0.5), cb, cs);
  let co = cs * sa * (1.0 - b.a) + b.rgb * (1.0 - sa) + sa * b.a * mixed;
  return vec4<f32>(clamp(co, vec3<f32>(0.0), vec3<f32>(1.0)), sa + b.a * (1.0 - sa));
}

// Signed distance to a rounded box with per-corner radii [tl, tr, br, bl] (the composite's sdRoundedBoxNonUniform).
fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
  var r = select(radii.xw, radii.yz, p.x > 0.0);
  let rr = select(r.x, r.y, p.y > 0.0);
  let q = abs(p) - halfSize + vec2<f32>(rr);
  return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - rr;
}

// Whether a sample point lies inside the frame's rounded clip (always, without one).
fn insideRoundedClip(s: vec2<f32>) -> bool {
  if (u.p3.z <= 0.0) {
    return true;
  }
  let h = u.p3.zw * 0.5;
  return sdRoundedBox(s - (u.p3.xy + h), h, u.p4) <= -0.5;
}

// Displace: sample the backdrop offset by two octaves of smooth value noise (one pair of channels per axis),
// kept inside the clip: a sample that would land outside a clipping frame's rounded corner takes the pixel's own.
fn displace(p: vec2<f32>, src: vec4<f32>) -> vec4<f32> {
  let dims = vec2<f32>(textureDimensions(uTexture));
  let q = toLocal(p) / max(u.p0.y, 1e-3);
  let salt = u32(u.p0.z) * 4u;
  let nx = 0.65 * valueNoise(q, salt) + 0.35 * valueNoise(q * 2.0 + 17.0, salt + 2u);
  let ny = 0.65 * valueNoise(q, salt + 1u) + 0.35 * valueNoise(q * 2.0 + 17.0, salt + 3u);
  let offset = fromLocalVector((vec2<f32>(nx, ny) * 2.0 - 1.0) * u.p0.x);
  let lo = u.p2.xy + vec2<f32>(0.5);
  let hi = max(u.p2.xy + u.p2.zw - vec2<f32>(0.5), lo);
  let s = clamp(p + offset, lo, hi);
  let sampled = textureSampleLevel(uTexture, uSampler, clamp(s / dims, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
  return select(src, sampled, insideRoundedClip(s));
}

// Grade: white balance, exposure, lift and gain as one slope and offset, then gamma, then saturation (Rec. 709
// luminance, linear light). The same chain as an SVG feComponentTransfer (linear, gamma) + feColorMatrix saturate.
fn grade(src: vec4<f32>) -> vec4<f32> {
  if (src.a <= 0.0) {
    return src;
  }
  let lifted = clamp(toLinear(src) * u.p0.rgb + vec3<f32>(u.p0.w), vec3<f32>(0.0), vec3<f32>(1.0));
  let curved = pow(lifted, vec3<f32>(u.p1.x));
  let y = dot(curved, vec3<f32>(0.2126, 0.7152, 0.0722));
  let graded = clamp(vec3<f32>(y) + (curved - vec3<f32>(y)) * u.p1.y, vec3<f32>(0.0), vec3<f32>(1.0));
  return fromLinear(graded, src.a);
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
  let p = frag.xy;
  let src = textureLoad(uTexture, vec2<i32>(p), 0);
  switch (i32(u.header.x + 0.5)) {
    case 0: { return grain(p, src); }
    case 1: { return vignette(p, src); }
    case 2: { return glaze(p, src); }
    case 3: { return displace(p, src); }
    case 4: { return grade(src); }
    default: { return src; }
  }
}
