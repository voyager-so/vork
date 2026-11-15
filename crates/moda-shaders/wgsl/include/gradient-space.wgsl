// Gradient interpolation spaces and dither (Illustrator audit §11), spliced into path-gradient, shape-gradient and
// the vector text gradient by build.rs.
//
// `flags` = space + 8 × dither + 16 × repeat. Space 0 is sRGB; 1 linear-light
// sRGB, 2 OKLab, 3 OKLCh (shorter hue arc). All spaces interpolate premultiplied, like CSS Color 4 (OKLCh
// premultiplies L and C, never the hue), and clamp the result to the sRGB gamut. Dither adds triangular noise of
// ±1 LSB before the 8-bit target quantizes, so long, low-contrast ramps don't band.

fn gsToLinear(c: vec3<f32>) -> vec3<f32> {
  let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(pow((x + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)), x / 12.92, x <= vec3<f32>(0.04045));
}

fn gsToSrgb(c: vec3<f32>) -> vec3<f32> {
  let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
  return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), x * 12.92, x <= vec3<f32>(0.0031308));
}

fn gsCbrt(v: vec3<f32>) -> vec3<f32> {
  return sign(v) * pow(abs(v), vec3<f32>(1.0 / 3.0));
}

// Björn Ottosson's OKLab, from linear sRGB.
fn gsLinearToOklab(c: vec3<f32>) -> vec3<f32> {
  let lms = vec3<f32>(
    0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
    0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
    0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b
  );
  let q = gsCbrt(lms);
  return vec3<f32>(
    0.2104542553 * q.x + 0.7936177850 * q.y - 0.0040720468 * q.z,
    1.9779984951 * q.x - 2.4285922050 * q.y + 0.4505937099 * q.z,
    0.0259040371 * q.x + 0.7827717662 * q.y - 0.8086757660 * q.z
  );
}

fn gsOklabToLinear(lab: vec3<f32>) -> vec3<f32> {
  let q = vec3<f32>(
    lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
    lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
    lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z
  );
  let lms = q * q * q;
  return vec3<f32>(
    4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
    -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
    -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z
  );
}

const GS_TWO_PI: f32 = 6.283185307179586;
const GS_PI: f32 = 3.141592653589793;
// Below this OKLCh chroma a colour has no hue (CSS's "powerless" hue): it takes the other end's.
const GS_ACHROMATIC: f32 = 0.0001;

// Straight RGBA `a` → `b` at `t` in the space `flags` names.
fn gsMix(a: vec4<f32>, b: vec4<f32>, t: f32, flags: f32) -> vec4<f32> {
  let space = u32(flags + 0.5) % 8u;
  if (space == 0u) {
    let alpha = mix(a.a, b.a, t);
    var color = mix(a.rgb * a.a, b.rgb * b.a, t);
    if (alpha > 0.0) { color = color / alpha; }
    return vec4<f32>(color, alpha);
  }
  var pa = gsToLinear(a.rgb);
  var pb = gsToLinear(b.rgb);
  if (space >= 2u) {
    pa = gsLinearToOklab(pa);
    pb = gsLinearToOklab(pb);
  }
  let alpha = mix(a.a, b.a, t);
  var m: vec3<f32>;
  if (space == 3u) {
    // OKLCh: L and C premultiplied, the hue along the shorter arc.
    let ca = length(pa.yz);
    let cb = length(pb.yz);
    var ha = atan2(pa.z, pa.y);
    var hb = atan2(pb.z, pb.y);
    if (ca < GS_ACHROMATIC) { ha = hb; }
    if (cb < GS_ACHROMATIC) { hb = ha; }
    var dh = hb - ha;
    if (dh > GS_PI) { dh = dh - GS_TWO_PI; }
    if (dh < -GS_PI) { dh = dh + GS_TWO_PI; }
    var lc = mix(vec2<f32>(pa.x, ca) * a.a, vec2<f32>(pb.x, cb) * b.a, t);
    if (alpha > 0.0) { lc = lc / alpha; } else { lc = mix(vec2<f32>(pa.x, ca), vec2<f32>(pb.x, cb), t); }
    let h = ha + dh * t;
    m = vec3<f32>(lc.x, lc.y * cos(h), lc.y * sin(h));
  } else {
    m = mix(pa * a.a, pb * b.a, t);
    if (alpha > 0.0) { m = m / alpha; } else { m = mix(pa, pb, t); }
  }
  if (space >= 2u) {
    m = gsOklabToLinear(m);
  }
  return vec4<f32>(gsToSrgb(m), alpha);
}

fn gsHash(p: vec2<f32>) -> f32 {
  let q = vec2<u32>(vec2<i32>(floor(p)));
  var h = q.x * 1664525u + q.y * 1013904223u;
  h = h ^ (h >> 16u);
  h = h * 2246822519u;
  h = h ^ (h >> 13u);
  h = h * 3266489917u;
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}

// `c` (straight RGBA) with ±1 LSB triangular noise at device pixel `pixel` when `flags` asks for dither.
fn gsDither(c: vec4<f32>, pixel: vec2<f32>, flags: f32) -> vec4<f32> {
  if ((u32(flags + 0.5) & 8u) == 0u) {
    return c;
  }
  let n = gsHash(pixel) + gsHash(pixel + vec2<f32>(113.0, 57.0)) - 1.0;
  return vec4<f32>(clamp(c.rgb + vec3<f32>(n / 255.0), vec3<f32>(0.0), vec3<f32>(1.0)), c.a);
}

fn gsRepeat(t: f32, first: f32, last: f32, flags: f32) -> f32 {
  if ((u32(flags + 0.5) & 16u) == 0u) { return t; }
  let period = last - first;
  if (period <= 0.0) { return last; }
  return first + fract((t - first) / period) * period;
}
