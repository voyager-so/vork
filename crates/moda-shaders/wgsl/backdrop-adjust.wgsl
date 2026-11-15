// backdrop-adjust — (native-only; hand-authored, no TS-renderer twin).
//
// The adjustment backdrop kinds (docs/design/adjustment-layers.md): Photoshop's adjustment layers as a backdrop
// effect on a fill-less layer. One pass over a backdrop split's copied region (moda-engine backdrop/exec.rs,
// `BackdropOp::Adjust`), then the ordinary backdrop composite (the layer's outline, opacity, blend and mask).
//
// It mirrors moda-paint's point kernels (native/crates/moda-paint/src/filter/point.rs) on the same 8-bit gamma
// values, so a live adjustment equals the destructive one (`paint.adjust`):
//   - the separable kinds (levels, curves, brightness / contrast, invert) read the lookup tables moda-color builds
//     for both (row 0 of `uLut`), indexed by the exact 8-bit value: byte-identical;
//   - desaturate, gradient map (row 1 of `uLut`) and the channel mixer are integer maths: byte-identical;
//   - hue / saturation, colour balance and vibrance are the same float formulas in f32: within ±1.
// Colour is straight: the premultiplied sample is divided by alpha (exact for opaque pixels), adjusted, and
// premultiplied again; alpha is kept and transparent pixels are left alone.

struct AdjustUniforms {
  // x: kind (0 table, 1 hue / saturation, 2 colour balance, 3 vibrance, 4 desaturate, 5 gradient map, 6 mixer),
  // y: a flag (colorize, preserve luminosity), z: desaturate method (0 luminosity, 1 average, 2 lightness).
  header: vec4<f32>,
  // hue / saturation: (hue turns, saturation, lightness, -); vibrance: (vibrance, saturation, -, -).
  p0: vec4<f32>,
  // Colour balance: shadows, midtones, highlights (r, g, b shifts, −1..1).
  b0: vec4<f32>,
  b1: vec4<f32>,
  b2: vec4<f32>,
  // Channel mixer rows, ×65536 integers (the constant ×255·65536): (r, g, b, constant).
  m0: vec4<f32>,
  m1: vec4<f32>,
  m2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: AdjustUniforms;
@group(0) @binding(1) var uTexture: texture_2d<f32>;
// 256 × 2 RGBA8 (uint): row 0 the per-channel tables, row 1 the gradient map's colours.
@group(0) @binding(2) var uLut: texture_2d<u32>;

@vertex
fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  return vec4<f32>(pos[vid], 0.0, 1.0);
}

fn luma(c: vec3<u32>) -> u32 {
  return (2126u * c.r + 7152u * c.g + 722u * c.b + 5000u) / 10000u;
}

fn rgbToHsl(c: vec3<f32>) -> vec3<f32> {
  let mx = max(max(c.r, c.g), c.b);
  let mn = min(min(c.r, c.g), c.b);
  let l = (mx + mn) / 2.0;
  if (mx == mn) {
    return vec3<f32>(0.0, 0.0, l);
  }
  let d = mx - mn;
  var s = d / (mx + mn);
  if (l > 0.5) {
    s = d / (2.0 - mx - mn);
  }
  var h: f32;
  if (mx == c.r) {
    h = (c.g - c.b) / d;
    if (c.g < c.b) {
      h = h + 6.0;
    }
  } else if (mx == c.g) {
    h = (c.b - c.r) / d + 2.0;
  } else {
    h = (c.r - c.g) / d + 4.0;
  }
  return vec3<f32>(h / 6.0, s, l);
}

fn hueChannel(p: f32, q: f32, tIn: f32) -> f32 {
  var t = tIn;
  if (t < 0.0) {
    t = t + 1.0;
  } else if (t > 1.0) {
    t = t - 1.0;
  }
  if (t < 1.0 / 6.0) {
    return p + (q - p) * 6.0 * t;
  }
  if (t < 0.5) {
    return q;
  }
  if (t < 2.0 / 3.0) {
    return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
  }
  return p;
}

fn hslToRgb(h: f32, s: f32, l: f32) -> vec3<f32> {
  if (s == 0.0) {
    return vec3<f32>(l, l, l);
  }
  var q = l + s - l * s;
  if (l < 0.5) {
    q = l * (1.0 + s);
  }
  let p = 2.0 * l - q;
  return vec3<f32>(hueChannel(p, q, h + 1.0 / 3.0), hueChannel(p, q, h), hueChannel(p, q, h - 1.0 / 3.0));
}

// The 8-bit value moda-paint would store for a 0..1 float: floor(v · 255 + 0.5), clamped.
fn store8(v: vec3<f32>) -> vec3<u32> {
  return vec3<u32>(clamp(floor(v * 255.0 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
}

fn balance(v: f32, l: f32, sh: f32, mid: f32, hi: f32) -> f32 {
  let ws = (1.0 - l) * (1.0 - l);
  let wm = 4.0 * l * (1.0 - l);
  let wh = l * l;
  return clamp(v + 0.4 * (sh * ws + mid * wm + hi * wh), 0.0, 1.0);
}

fn adjust(c: vec3<u32>) -> vec3<u32> {
  let kind = u32(u.header.x);
  let f = vec3<f32>(c) / 255.0;
  switch (kind) {
    case 0u: {
      return vec3<u32>(
        textureLoad(uLut, vec2<i32>(i32(c.r), 0), 0).r,
        textureLoad(uLut, vec2<i32>(i32(c.g), 0), 0).g,
        textureLoad(uLut, vec2<i32>(i32(c.b), 0), 0).b
      );
    }
    case 1u: {
      var hsl: vec3<f32>;
      if (u.header.y > 0.5) {
        hsl = vec3<f32>(u.p0.x, u.p0.y, f32(luma(c)) / 255.0);
      } else {
        let h0 = rgbToHsl(f);
        let h = h0.x + u.p0.x;
        hsl = vec3<f32>(h - floor(h), clamp(h0.y * (1.0 + u.p0.y), 0.0, 1.0), h0.z);
      }
      var o = hslToRgb(hsl.x, hsl.y, hsl.z);
      let light = u.p0.z;
      if (light > 0.0) {
        o = o * (1.0 - light) + vec3<f32>(light);
      } else {
        o = o * (1.0 + light);
      }
      return store8(o);
    }
    case 2u: {
      let l = rgbToHsl(f).z;
      var o = vec3<f32>(
        balance(f.r, l, u.b0.x, u.b1.x, u.b2.x),
        balance(f.g, l, u.b0.y, u.b1.y, u.b2.y),
        balance(f.b, l, u.b0.z, u.b1.z, u.b2.z)
      );
      if (u.header.y > 0.5) {
        let hs = rgbToHsl(o);
        o = hslToRgb(hs.x, hs.y, l);
      }
      return store8(o);
    }
    case 3u: {
      let y = f32(luma(c)) / 255.0;
      let mx = max(max(f.r, f.g), f.b);
      let mn = min(min(f.r, f.g), f.b);
      var s = 0.0;
      if (mx > 0.0) {
        s = (mx - mn) / mx;
      }
      let k = (1.0 + u.p0.y) * (1.0 + u.p0.x * (1.0 - s));
      return store8(vec3<f32>(y) + (f - vec3<f32>(y)) * k);
    }
    case 4u: {
      let method = u32(u.header.z);
      var v: u32;
      if (method == 1u) {
        v = (c.r + c.g + c.b + 1u) / 3u;
      } else if (method == 2u) {
        v = (max(max(c.r, c.g), c.b) + min(min(c.r, c.g), c.b) + 1u) / 2u;
      } else {
        v = luma(c);
      }
      return vec3<u32>(v, v, v);
    }
    case 5u: {
      return textureLoad(uLut, vec2<i32>(i32(luma(c)), 1), 0).rgb;
    }
    case 6u: {
      let ci = vec3<i32>(c);
      let rows = array<vec4<f32>, 3>(u.m0, u.m1, u.m2);
      var o: vec3<u32>;
      for (var k = 0; k < 3; k = k + 1) {
        let r = vec4<i32>(rows[k]);
        let v = (r.x * ci.r + r.y * ci.g + r.z * ci.b + r.w + 32768) >> 16u;
        o[k] = u32(clamp(v, 0, 255));
      }
      return o;
    }
    default: {
      return c;
    }
  }
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let src = textureLoad(uTexture, vec2<i32>(pos.xy), 0);
  if (src.a <= 0.0) {
    return vec4<f32>(0.0);
  }
  // Straight 8-bit colour, as moda-paint's kernels see it.
  let c = vec3<u32>(clamp(floor(src.rgb / src.a * 255.0 + 0.5), vec3<f32>(0.0), vec3<f32>(255.0)));
  let o = adjust(c);
  return vec4<f32>(vec3<f32>(o) / 255.0 * src.a, src.a);
}
