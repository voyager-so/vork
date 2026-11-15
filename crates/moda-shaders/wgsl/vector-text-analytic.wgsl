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
// Quadratic outline segments: 3 control points per curve, em units (y-down, pen origin).
@group(0) @binding(2) var<storage, read> curvePoints: array<vec2<f32>>;

struct VSOut {
  @builtin(position) pos: vec4<f32>,
  // True world position: affine in screen space, so linear (non-perspective)
  // interpolation is exact on warped quads; em/local are affine in LOCAL
  // space and need the default perspective-correct interpolation.
  @location(0) @interpolate(linear) world: vec2<f32>,
  @location(1) em: vec2<f32>,
  @location(2) @interpolate(flat) curveRange: vec2<u32>, // firstCurve, curveCount
  @location(3) @interpolate(flat) fontSizePx: f32,
  @location(4) @interpolate(flat) glyphOpacity: f32, // KineticText per-glyph opacity
};


fn kineticGlyphLocal(
  pen: vec2<f32>, em: vec2<f32>, emBox: vec4<f32>, fontSizePx: f32,
  glyphLin: vec4<f32>, glyphOff: vec2<f32>,
) -> vec2<f32> {
  let baseLocal = pen + em * fontSizePx;
  if (glyphLin.x == 1.0 && glyphLin.y == 0.0 && glyphLin.z == 0.0 && glyphLin.w == 1.0 &&
      glyphOff.x == 0.0 && glyphOff.y == 0.0) {
    return baseLocal;
  }
  let center = pen + (emBox.xy + emBox.zw) * 0.5 * fontSizePx;
  let rel = baseLocal - center;
  return vec2<f32>(
    glyphLin.x * rel.x + glyphLin.y * rel.y + glyphOff.x,
    glyphLin.z * rel.x + glyphLin.w * rel.y + glyphOff.y,
  ) + center;
}

@vertex fn vs(
  @location(0) corner: vec2<f32>,      // unit quad corner (0..1)
  @location(1) pen: vec2<f32>,         // local-space pen position (x, baselineY)
  @location(2) fontSizePx: f32,
  @location(3) emBox: vec4<f32>,       // padded glyph bounds in em units
  @location(4) curveRange: vec2<u32>,
  @location(5) glyphLin: vec4<f32>,    // KineticText per-glyph 2×2 [a,c,b,d]
  @location(6) glyphOff: vec2<f32>,    // KineticText per-glyph px offset
  @location(7) glyphOpacity: f32,      // KineticText per-glyph opacity
) -> VSOut {
  let em = mix(emBox.xy, emBox.zw, corner);
  // KineticText: per-glyph transform about the glyph em-box
  // center; identity instances fast-path to the original math (byte-exact static).
  let local = kineticGlyphLocal(pen, em, emBox, fontSizePx, glyphLin, glyphOff);
  let h = vectorWarpLocal(local);
  let world = draw.model * vec4<f32>(h.xy, 0.0, h.z);
  var o: VSOut;
  o.pos = frame * world;
  o.pos.z = draw.params.x * h.z; // hardware divide restores params.x
  o.world = world.xy / h.z;
  o.em = em;
  o.curveRange = curveRange;
  o.fontSizePx = fontSizePx;
  o.glyphOpacity = glyphOpacity;
  return o;
}

// #include "vector-text-coverage.wgsl"

// Squared distance from p to the segment a-b, plus the parameter of the closest point
// (degenerate-safe: a == b → distance to a at h = 0).
fn vectorSegmentNearest(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  let pa = p - a;
  let ba = b - a;
  let bb = dot(ba, ba);
  var h = 0.0;
  if (bb > 1e-12) { h = clamp(dot(pa, ba) / bb, 0.0, 1.0); }
  let d = pa - ba * h;
  return vec2<f32>(dot(d, d), h);
}

// Min squared distance from p to the quadratic Bézier (p0, p1, p2), plus the curve
// parameter of the closest point: closed-form cubic root classification (iq's
// sdBezier). The extractor emits straight edges as degenerate quadratics (control =
// midpoint, so the quadratic term vanishes); those must route to the segment case — the
// cubic solve divides by dot(b, b). The routing threshold is generous (1e-5 em², i.e.
// |b| ≲ 3.2e-3 em) because kk = 1/dot(b, b) conditions the solve: near-degenerate
// curves catastrophically cancel the kx³-scale terms in f32, while linearizing them
// errs by at most |b|/4 < 1e-3 em — sub-pixel at any plausible rendered size, and only
// within the effect feather.
fn vectorQuadBezierNearest(p: vec2<f32>, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>) -> vec2<f32> {
  let b = p0 - 2.0 * p1 + p2;
  let bb = dot(b, b);
  if (bb < 1e-5) {
    return vectorSegmentNearest(p, p0, p2);
  }
  let a = p1 - p0;
  let d = p0 - p;
  let kk = 1.0 / bb;
  let kx = kk * dot(a, b);
  let ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
  let kz = kk * dot(d, a);
  let pp = ky - kx * kx;
  let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
  let h = q * q + 4.0 * pp * pp * pp;
  if (h >= 0.0) {
    // One real root.
    let sh = sqrt(h);
    let x = (vec2<f32>(sh, -sh) - q) / 2.0;
    let uv = sign(x) * pow(abs(x), vec2<f32>(1.0 / 3.0));
    let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
    let v = d + (2.0 * a + b * t) * t;
    return vec2<f32>(dot(v, v), t);
  }
  // Three real roots (the third can never be the closest).
  let z = sqrt(-pp);
  let theta = acos(clamp(q / (pp * z * 2.0), -1.0, 1.0)) / 3.0;
  let m = cos(theta);
  let n = sin(theta) * 1.7320508075688772;
  let t = clamp(vec2<f32>(m + m, -n - m) * z - kx, vec2<f32>(0.0), vec2<f32>(1.0));
  let v0 = d + (2.0 * a + b * t.x) * t.x;
  let v1 = d + (2.0 * a + b * t.y) * t.y;
  let d0 = dot(v0, v0);
  let d1 = dot(v1, v1);
  return select(vec2<f32>(d1, t.y), vec2<f32>(d0, t.x), d0 <= d1);
}

// Nearest outline point of one glyph from `em`: unsigned distance (em units) plus
// which curve won and where on it, so the effect can classify whether that edge is on
// the resolved boundary. Pair with the winding coverage for the sign.
struct VectorGlyphNearest {
  distEm: f32,
  curve: u32, // glyph-relative index of the winning curve
  t: f32,     // its parameter at the closest point (segment h for degenerate curves)
};

fn vectorGlyphNearestEm(first: u32, count: u32, em: vec2<f32>) -> VectorGlyphNearest {
  var best = vec2<f32>(1e30, 0.0);
  var bestCurve = 0u;
  for (var i = 0u; i < count; i = i + 1u) {
    let base = (first + i) * 3u;
    let cand = vectorQuadBezierNearest(em, curvePoints[base], curvePoints[base + 1u], curvePoints[base + 2u]);
    let closer = cand.x < best.x;
    best = select(best, cand, closer);
    bestCurve = select(bestCurve, i, closer);
  }
  var nearest: VectorGlyphNearest;
  nearest.distEm = sqrt(best.x);
  nearest.curve = bestCurve;
  nearest.t = best.y;
  return nearest;
}

// True when the glyph's nearest outline edge is an interior edge of overlapping
// contours (DM Sans A/H/k): both sides of the outline at the closest point
// sit inside the resolved non-zero-winding silhouette, so the edge is not on the glyph
// boundary and must not shape the effect band. Two probes straddle the outline at the
// closest point and a sharp winding number is taken at both in one pass over the
// curves (the 1e8 inverse diameter collapses the coverage helper's pixel filter to a
// step, so the crossing count is exact); a true boundary always leaves one probe
// outside. Mid-edge closest points straddle along the curve normal. A clamped t means
// the closest point is a contour corner shared with an adjacent edge — there the
// winning edge is an FP-noise tie and its normal would straddle the edge's tangent
// line extended past the corner, misreading sharp concave silhouette corners (whose
// inside Voronoi wedge resolves to the corner) as interior — so corner probes
// straddle along the fragment→corner direction instead: the fragment side is
// connected to the fragment (the segment to the closest point crosses no curves) and
// the far side alone decides, independent of which tied edge won. The 1.13e-3 em
// probe offset is deliberately off the 1/1000 and 1/2048 font-unit grids so probes
// (and their +x winding rays) can't sit exactly on outline coordinates of
// em-normalized glyphs and graze a vertex.
fn vectorNearestEdgeIsInterior(first: u32, count: u32, em: vec2<f32>, nearest: VectorGlyphNearest) -> bool {
  let base = (first + nearest.curve) * 3u;
  let p0 = curvePoints[base];
  let p1 = curvePoints[base + 1u];
  let p2 = curvePoints[base + 2u];
  let b = p0 - 2.0 * p1 + p2;
  var q: vec2<f32>;
  var tangent: vec2<f32>;
  if (dot(b, b) < 1e-5) {
    // Same degenerate routing as vectorQuadBezierNearest: t is the segment parameter.
    q = mix(p0, p2, nearest.t);
    tangent = p2 - p0;
  } else {
    let a = p1 - p0;
    q = p0 + (2.0 * a + b * nearest.t) * nearest.t;
    tangent = a + b * nearest.t; // B'(t) / 2
  }
  var axis: vec2<f32>;
  if (nearest.t <= 0.0 || nearest.t >= 1.0) {
    let away = q - em;
    let awayLen = length(away);
    if (awayLen < 1e-6) { return false; } // fragment on the corner: keep the band
    axis = (away / awayLen) * 1.13e-3;
  } else {
    let len = length(tangent);
    if (len < 1e-9) { return false; } // cusp: keep the raw distance
    axis = (vec2<f32>(-tangent.y, tangent.x) / len) * 1.13e-3;
  }
  let qa = q + axis;
  let qb = q - axis;
  var wa = 0.0;
  var wb = 0.0;
  for (var i = 0u; i < count; i = i + 1u) {
    let cbase = (first + i) * 3u;
    let c0 = curvePoints[cbase];
    let c1 = curvePoints[cbase + 1u];
    let c2 = curvePoints[cbase + 2u];
    wa = wa + vectorWindingAlongRay(c0 - qa, c1 - qa, c2 - qa, 1e8);
    wb = wb + vectorWindingAlongRay(c0 - qb, c1 - qb, c2 - qb, 1e8);
  }
  return abs(wa) > 0.5 && abs(wb) > 0.5;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4<f32> {
  // Pixel footprint in em units (derivatives must be computed before any discard).
  let footprint = max(fwidth(in.em), vec2<f32>(1e-6));
  let invDiameter = 1.0 / footprint;

  let clip = draw.pageClip;
  if (in.world.x < clip.x || in.world.x > clip.x + clip.z ||
      in.world.y < clip.y || in.world.y > clip.y + clip.w) {
    discard;
  }

  let coverage = vectorGlyphCoverage(in.curveRange.x, in.curveRange.y, in.em, invDiameter);

  // Stroke / drop-shadow effects (Phase 1): params.z = blur px, params.w = stroke px,
  // both in local (canvas) px so the effect scales with the node transform. Zero on
  // plain fill draws, so the branch is uniform per draw.
  let blurPx = draw.params.z;
  let strokePx = draw.params.w;
  var alpha = coverage;
  if (blurPx > 0.0 || strokePx > 0.0) {
    let fontPx = max(in.fontSizePx, 1e-3);
    // One device pixel in em units (isotropic approximation of the footprint).
    let aaEm = max(0.5 * (footprint.x + footprint.y), 1e-9);
    // Signed distance to the outline, positive inside: the winding coverage resolves
    // inside/outside, so counters and overlapping contours sign correctly.
    let nearest = vectorGlyphNearestEm(in.curveRange.x, in.curveRange.y, in.em);
    var sdEm = select(-nearest.distEm, nearest.distEm, coverage > 0.5);
    let widthEm = max(blurPx / fontPx, aaEm);
    let strokeRadiusEm = 0.5 * strokePx / fontPx;
    let interiorReachEm = strokeRadiusEm + widthEm + aaEm;
    // Interior edges of overlapping contours are not on the resolved boundary and must
    // not seam the band: force them deep inside so both band shapes below
    // saturate. Only inside fragments can be nearest to an interior edge (from outside,
    // some resolved-boundary point is always closer), and past widthEm + aaEm the bands
    // saturate anyway, so the probe windings only run for inside fragments near an edge.
    if (coverage > 0.5 && sdEm < interiorReachEm &&
        vectorNearestEdgeIsInterior(in.curveRange.x, in.curveRange.y, in.em, nearest)) {
      sdEm = interiorReachEm;
    }
    if (strokePx > 0.0) {
      // CSS/SVG stroke is centred on the outline: half inside, half outside.
      let outer = smoothstep(-widthEm, widthEm, sdEm + strokeRadiusEm);
      let inner = smoothstep(-widthEm, widthEm, sdEm - strokeRadiusEm);
      alpha = clamp(outer - inner, 0.0, 1.0);
    } else {
      // Drop-shadow blur: soften the silhouette edge over ±blur px (the same SDF-style
      // approximation as the MSDF shadow pass).
      alpha = smoothstep(-widthEm, widthEm, sdEm);
    }
  }

  if (alpha < 0.004) {
    discard;
  }
  // KineticText: per-glyph opacity multiplies coverage on the fill AND
  // every effect pass (shadow/stroke/glow/echo re-emit the glyph carrying it), so
  // a glyph at opacity 0 contributes zero everywhere. Exactly 1.0 for static text.
  let finalAlpha = draw.color.a * draw.params.y * in.glyphOpacity * alpha;
  return vec4<f32>(workingRgb(draw.color.rgb), finalAlpha);
}