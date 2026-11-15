// #include "working-color.wgsl"
// Common constants

// Mathematical constants
const INVERSE_SQRT_2: f32 = 0.7071067811865476;

// Shader constants
const AA_PIXEL_EXPANSION: f32 = 0.5;
const MIN_BLUR_RADIUS: f32 = 0.01;
const MIN_GRADIENT_STOP_DIFF: f32 = 0.0001;


// Main shader code

      @group(0) @binding(0) var<uniform> frameUniforms: mat4x4<f32>; // viewProjection

      struct RectData {
        model: mat4x4<f32>,     // 16 floats
        color: vec4<f32>,       // 4 floats
        dimensions: vec4<f32>,  // width, height, depth, shapeType
        cornerRadii: vec4<f32>, // tl, tr, br, bl (for rect) or radiusX, radiusY, 0, 0 (for ellipse)
        pageClipBounds: vec4<f32>, // x, y, width, height
        innerShadowColor: vec4<f32>,  // rgb + combined alpha (0 => no inner shadow)
        innerShadowParams: vec4<f32>, // offsetX, offsetY, blur, spread (shape-local px)
        strokeColor: vec4<f32>,       // rgb (straight) + combined alpha (0 => no stroke)
        strokeParams: vec4<f32>,      // width, alignCode (0=center,1=inside,2=outside), 0, 0
      }

      @group(1) @binding(0) var<storage, read> rects: array<RectData>;

      struct VSOut {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
        @location(1) worldPos: vec2<f32>,
        @location(2) @interpolate(flat) clip: vec4<f32>,
        @location(3) localPos: vec2<f32>,
        @location(4) @interpolate(flat) size: vec2<f32>,
        @location(5) @interpolate(flat) radii: vec4<f32>,
        @location(6) @interpolate(flat) shapeType: f32,
        @location(7) @interpolate(flat) innerShadowColor: vec4<f32>,
        @location(8) @interpolate(flat) innerShadowParams: vec4<f32>,
        @location(9) @interpolate(flat) strokeColor: vec4<f32>,
        @location(10) @interpolate(flat) strokeParams: vec4<f32>,
      }

      // SDF for rounded rectangle with per-corner radii
      // p: point in local space (centered at rectangle center)
      // size: half-size of rectangle (width/2, height/2)
      // radii: corner radii (tl, tr, br, bl)
      fn sdRoundedBox(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> f32 {
        // Select the appropriate corner radius based on which quadrant we're in
        // Top-left (x<0, y<0): radii.x
        // Top-right (x>0, y<0): radii.y
        // Bottom-right (x>0, y>0): radii.z
        // Bottom-left (x<0, y>0): radii.w
        let r = select(
          select(radii.w, radii.z, p.x > 0.0),  // bottom half
          select(radii.x, radii.y, p.x > 0.0),  // top half
          p.y < 0.0
        );

        // Standard rounded box SDF
        let q = abs(p) - size + r;
        return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - r;
      }

      // Unit gradient of sdRoundedBox at p, in local units. The field is only piecewise smooth: inside the box it is
      // max(q.x, q.y), whose gradient flips between the two axes along each corner's diagonal. fwidth(dist) over a
      // 2x2 pixel quad that straddles that diagonal adds BOTH axes' differences, twice a pixel at a square corner, so
      // a corner pixel whose quad reached outside the shape came out 3/4 covered (alpha 191).
      fn sdRoundedBoxGradient(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>) -> vec2<f32> {
        let r = select(
          select(radii.w, radii.z, p.x > 0.0),
          select(radii.x, radii.y, p.x > 0.0),
          p.y < 0.0
        );
        let q = abs(p) - size + r;
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
        if (q.x > 0.0 || q.y > 0.0) {
          return s * normalize(max(q, vec2<f32>(0.0)));
        }
        return s * select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), q.x > q.y);
      }

      // One device pixel in the SDF's units, as fwidth(dist) measures it where the field is smooth
      // (|d·dx| + |d·dy| for the gradient d), but from the analytic gradient for a rectangle, so the
      // flip at its corners does not double it. Ellipses keep fwidth(dist) (their field is smooth).
      fn sdfEdgeWidth(fw: f32, p: vec2<f32>, dpx: vec2<f32>, dpy: vec2<f32>, size: vec2<f32>, radii: vec4<f32>, shapeType: f32) -> f32 {
        let g = sdRoundedBoxGradient(p, size, radii);
        return select(fw, abs(dot(g, dpx)) + abs(dot(g, dpy)), shapeType < 0.5);
      }

      // SDF for ellipse
      // p: point in local space (centered at ellipse center)
      // r: radii (radiusX, radiusY)
      fn sdEllipse(p: vec2<f32>, r: vec2<f32>) -> f32 {
        if (r.x < 0.0001 || r.y < 0.0001) { return 1.0; }
        let k0 = length(p / r);
        let k1 = length(p / (r * r));
        if (k1 < 0.0001) { return -min(r.x, r.y); }
        return k0 * (k0 - 1.0) / k1;
      }

      // Unified shape SDF
      fn sdShape(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>, shapeType: f32) -> f32 {
        if (shapeType < 0.5) {
          // Rectangle with rounded corners
          return sdRoundedBox(p, size, radii);
        } else {
          // Ellipse (radii.x = radiusX, radii.y = radiusY)
          return sdEllipse(p, vec2<f32>(radii.x, radii.y));
        }
      }

      // How far the stroke band reaches OUTSIDE the shape geometry, so
      // the vertex shader can grow the quad to contain it. center => width/2,
      // outside => width, inside => 0; 0 when there is no stroke (strokeColor.a
      // == 0), leaving the fill quad unchanged.
      fn strokeOutsetExtent(strokeColor: vec4<f32>, strokeParams: vec4<f32>) -> f32 {
        if (strokeColor.a <= 0.0 || strokeParams.x <= 0.0) {
          return 0.0;
        }
        let alignCode = strokeParams.y;
        if (alignCode > 1.5) {
          return strokeParams.x;        // outside
        } else if (alignCode > 0.5) {
          return 0.0;                   // inside
        }
        return strokeParams.x * 0.5;    // center
      }

      // Signed distance to the stroke RING (negative inside the band).
      // d = signed distance to the shape; h = strokeWidth/2. center => abs(d) - h,
      // inside => abs(d + h) - h, outside => abs(d - h) - h. For ellipses the band
      // rides sdEllipse's near-boundary approximation (same field the fill/inner
      // shadow use), so a wide stroke on a very eccentric ellipse can vary slightly
      // in width — acceptable per the golden gate.
      fn strokeRingDistance(d: f32, strokeParams: vec4<f32>) -> f32 {
        let h = strokeParams.x * 0.5;
        let alignCode = strokeParams.y;
        var alignOffset = 0.0;          // center
        if (alignCode > 1.5) {
          alignOffset = -h;             // outside
        } else if (alignCode > 0.5) {
          alignOffset = h;              // inside
        }
        return abs(d + alignOffset) - h;
      }

      
// Box-filter edge coverage in [0,1]: 1 at half a pixel inside the silhouette,
// 0.5 on the edge, 0 at half a pixel outside. edgeWidth is fwidth(dist) —
// one device pixel expressed in the SDF's units.
fn sdfCoverage(dist: f32, edgeWidth: f32) -> f32 {
  return clamp(0.5 - dist / max(edgeWidth, 1e-6), 0.0, 1.0);
}


      
// Inner (inset) shadow, CSS / Figma semantics: the complement of the silhouette shrunk
// by the spread and moved by the offset, blurred with a Gaussian of sigma = blur / 2, over
// the fill. The blur is evaluated exactly per silhouette row (a difference of erfs) and
// summed over the rows within 4 sigma (Evan Wallace, "Fast rounded rectangle shadows") —
// the same evaluation as the path-shadow lane, so both lanes agree.
const INSET_SHADOW_ROWS: i32 = 12;

fn insetErf2(x: vec2<f32>) -> vec2<f32> {
  let s = sign(x);
  let a = abs(x);
  var y = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
  y = y * y;
  return s - s / (y * y);
}

// The shape (half size `size`, radii tl/tr/br/bl — or radii.xy = (rx, ry) for an ellipse)
// convolved with a Gaussian of `sigma`, at `p`.
fn insetGaussianShape(p: vec2<f32>, size: vec2<f32>, radii: vec4<f32>, isEllipse: bool, sigma: f32) -> f32 {
  let halfW = select(size.x, radii.x, isEllipse);
  let halfH = select(size.y, radii.y, isEllipse);
  if (halfW <= 0.0 || halfH <= 0.0) {
    return 0.0;
  }
  let lo = clamp(-4.0 * sigma, p.y - halfH, p.y + halfH);
  let hi = clamp(4.0 * sigma, p.y - halfH, p.y + halfH);
  let stepK = (hi - lo) / f32(INSET_SHADOW_ROWS);
  let invS = INVERSE_SQRT_2 / sigma;
  let norm = 1.0 / (2.5066282746 * sigma);
  var total = 0.0;
  var k = lo + 0.5 * stepK;
  for (var i = 0; i < INSET_SHADOW_ROWS; i++) {
    let rowY = p.y - k;
    var e: vec2<f32>;
    if (isEllipse) {
      let t = rowY / halfH;
      let w = halfW * sqrt(max(0.0, 1.0 - t * t));
      e = vec2<f32>(w, w);
    } else {
      let rl = select(radii.w, radii.x, rowY < 0.0);
      let rr = select(radii.z, radii.y, rowY < 0.0);
      let dl = max(abs(rowY) - (size.y - rl), 0.0);
      let dr = max(abs(rowY) - (size.y - rr), 0.0);
      e = vec2<f32>(size.x - rl + sqrt(max(0.0, rl * rl - dl * dl)), size.x - rr + sqrt(max(0.0, rr * rr - dr * dr)));
    }
    let span = 0.5 * insetErf2(vec2<f32>(p.x + e.x, p.x - e.y) * invS);
    total += (span.x - span.y) * exp(-0.5 * k * k / (sigma * sigma)) * norm * stepK;
    k += stepK;
  }
  return clamp(total, 0.0, 1.0);
}

// Composite the inner shadow over a straight-alpha fill rgb and return the new rgb.
// innerColor: rgb = shadow color, a = innerShadowOpacity (NOT multiplied by node
//             opacity — node opacity is applied once by the host via the body/final
//             alpha). When a <= 0 there is no inner shadow and the fill is returned unchanged.
// innerParams: (offsetX, offsetY, blur, spread) in shape-local pixels.
// pixelLocal: one device pixel in shape-local units (fwidth of the SDF), which widens sigma
//             so a hard (blur 0) inner shadow keeps an anti-aliased edge.
fn applyInnerShadow(
  fillRgb: vec3<f32>,
  innerColor: vec4<f32>,
  innerParams: vec4<f32>,
  localPos: vec2<f32>,
  size: vec2<f32>,
  radii: vec4<f32>,
  shapeType: f32,
  pixelLocal: f32
) -> vec3<f32> {
  if (innerColor.a <= 0.0) {
    return fillRgb;
  }
  let isEllipse = shapeType >= 0.5;
  let spread = innerParams.w;
  // The caster: the silhouette shrunk by the spread (a rounded corner r → max(0, r − spread),
  // a square corner stays square; ellipse radii − spread).
  var casterSize = size - vec2<f32>(spread);
  var casterRadii: vec4<f32>;
  if (isEllipse) {
    casterRadii = vec4<f32>(radii.xy - vec2<f32>(spread), 0.0, 0.0);
  } else {
    let shrunk = select(vec4<f32>(0.0), max(radii - vec4<f32>(spread), vec4<f32>(0.0)), radii > vec4<f32>(0.0));
    casterRadii = min(shrunk, vec4<f32>(max(min(casterSize.x, casterSize.y), 0.0)));
  }
  let sigma = sqrt(0.25 * innerParams.z * innerParams.z + 0.09 * pixelLocal * pixelLocal) + 1e-6;
  let caster = insetGaussianShape(localPos - innerParams.xy, casterSize, casterRadii, isEllipse, sigma);
  let strength = clamp(innerColor.a * (1.0 - caster), 0.0, 1.0);
  return mix(fillRgb, innerColor.rgb, strength);
}


      @vertex
      fn vertexMain(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> VSOut {
        var out: VSOut;
        let r = rects[iid];

        // Two triangles forming a quad in [-1,1]
        var pos: vec2<f32>;
        switch(vid) {
          case 0u: { pos = vec2<f32>(-1.0, -1.0); }
          case 1u: { pos = vec2<f32>( 1.0, -1.0); }
          case 2u: { pos = vec2<f32>(-1.0,  1.0); }
          case 3u: { pos = vec2<f32>(-1.0,  1.0); }
          case 4u: { pos = vec2<f32>( 1.0, -1.0); }
          default: { pos = vec2<f32>( 1.0,  1.0); }
        }

        let halfSize = 0.5 * r.dimensions.xy;
        let shapeType = r.dimensions.w;

        // Unified positioning: model matrix translation is top-left for BOTH rectangles and ellipses.
        // Grow the quad outward by the stroke's outset so an outer/
        // center stroke band is not clipped. Inside-aligned strokes and unstroked
        // bodies grow by 0, so the fill quad (and its rasterized fragments) is
        // unchanged. Map [-1,1] -> [-outset, width+outset] x [-outset, height+outset].
        let strokeExtend = strokeOutsetExtent(r.strokeColor, r.strokeParams);
        let grownHalf = halfSize + vec2<f32>(strokeExtend);
        let anchored = pos * grownHalf + halfSize;
        // Local position centered at shape center for SDF evaluation (rect + ellipse share this).
        let localPos = anchored - halfSize;

        let local = vec4<f32>(anchored, 0.0, 1.0);
        let world = r.model * local;
        var clip = frameUniforms * world;
        clip.z = r.dimensions.z; // use precomputed linear depth
        out.position = clip;
        out.color = r.color;
        out.worldPos = world.xy;
        out.clip = r.pageClipBounds;

        // Pass local position for SDF calculation
        out.localPos = localPos;
        out.size = halfSize;
        out.radii = r.cornerRadii;
        out.shapeType = shapeType;
        out.innerShadowColor = r.innerShadowColor;
        out.innerShadowParams = r.innerShadowParams;
        out.strokeColor = r.strokeColor;
        out.strokeParams = r.strokeParams;

        return out;
      }

      // Fragment for premultiplied blending mode (transparent path + MSAA=1 fallback).
      //
      // inp.color carries STRAIGHT-alpha color: rgb = fill color, a = body alpha
      // (fill alpha × node opacity). The pipeline blend is ('one', 'one-minus-src-alpha'),
      // which expects PREMULTIPLIED output. We therefore combine body alpha with the
      // SDF coverage into a single combined alpha and premultiply rgb by it.
      @fragment
      fn fragmentMainPremul(inp: VSOut) -> @location(0) vec4<f32> {
        // Check page clip bounds first
        let bounds = inp.clip;
        if (inp.worldPos.x < bounds.x ||
            inp.worldPos.x > bounds.x + bounds.z ||
            inp.worldPos.y < bounds.y ||
            inp.worldPos.y > bounds.y + bounds.w) {
          discard;
        }

        // Calculate signed distance to shape (rectangle or ellipse)
        let dist = sdShape(inp.localPos, inp.size, inp.radii, inp.shapeType);
        // fwidth() must run in uniform control flow, so derive the edge width and
        // both coverages before compositing (no per-fragment branch on them).
        let aa = sdfEdgeWidth(fwidth(dist), inp.localPos, dpdx(inp.localPos), dpdy(inp.localPos), inp.size, inp.radii, inp.shapeType);
        let coverage = sdfCoverage(dist, aa);

        // Composite the inner (inset) shadow over the fill. No-op when the inner
        // shadow color alpha is 0. Edge coverage below keeps it inside the shape.
        let shadedRgb = applyInnerShadow(
          workingRgb(inp.color.rgb), vec4<f32>(workingRgb(inp.innerShadowColor.rgb), inp.innerShadowColor.a), inp.innerShadowParams,
          inp.localPos, inp.size, inp.radii, inp.shapeType, aa
        );
        // Combined fill alpha = body alpha × SDF edge coverage.
        let fillAlpha = inp.color.a * coverage;

        // Solid stroke ring composited OVER the inner-shadowed fill.
        // strokeColor.a == 0 ⇒ no stroke ⇒ strokeAlpha 0, so the result reduces
        // EXACTLY to the pre-stroke premultiplied fill (+0 / ×1 only).
        let strokeDist = strokeRingDistance(dist, inp.strokeParams);
        let strokeCoverage = sdfCoverage(strokeDist, aa);
        let strokeAlpha = inp.strokeColor.a * strokeCoverage;
        let outAlpha = strokeAlpha + fillAlpha * (1.0 - strokeAlpha);

        // Discard pixels outside the shape AND the outer half of the AA edge band
        // (fill or stroke). Discarding on dist > 0 alone would throw away coverage
        // in (0, 0.5] where the edge feathers to zero — visibly aliasing the outer
        // edge, especially with MSAA disabled.
        if (outAlpha <= 0.001) {
          discard;
        }

        // Output premultiplied rgba so the ('one','one-minus-src-alpha') blend
        // produces src_rgb * outAlpha + dst * (1 - outAlpha). "Stroke over fill"
        // in premultiplied space; for the unstroked opaque case (strokeAlpha 0,
        // inp.color.a == 1) this reduces to the previous behaviour (rgb*cov, cov).
        let premulRgb = workingRgb(inp.strokeColor.rgb) * strokeAlpha + shadedRgb * fillAlpha * (1.0 - strokeAlpha);
        return vec4<f32>(premulRgb, outAlpha);
      }

      // Fragment for alpha-to-coverage mode (opaque path with MSAA>=2)
      @fragment
      fn fragmentMainA2C(inp: VSOut) -> @location(0) vec4<f32> {
        // Check page clip bounds first
        let bounds = inp.clip;
        if (inp.worldPos.x < bounds.x ||
            inp.worldPos.x > bounds.x + bounds.z ||
            inp.worldPos.y < bounds.y ||
            inp.worldPos.y > bounds.y + bounds.w) {
          discard;
        }

        // Calculate signed distance to shape (rectangle or ellipse)
        let dist = sdShape(inp.localPos, inp.size, inp.radii, inp.shapeType);
        // fwidth() must run in uniform control flow (see fragmentMainPremul).
        let aa = sdfEdgeWidth(fwidth(dist), inp.localPos, dpdx(inp.localPos), dpdy(inp.localPos), inp.size, inp.radii, inp.shapeType);
        let coverage = sdfCoverage(dist, aa);

        // Composite the inner (inset) shadow over the fill (rgb only; coverage/alpha
        // unchanged). No-op when the inner shadow color alpha is 0.
        let shadedRgb = applyInnerShadow(
          workingRgb(inp.color.rgb), vec4<f32>(workingRgb(inp.innerShadowColor.rgb), inp.innerShadowColor.a), inp.innerShadowParams,
          inp.localPos, inp.size, inp.radii, inp.shapeType, aa
        );
        let fillAlpha = inp.color.a * coverage;

        // Stroke ring over the fill (see fragmentMainPremul). A2C outputs
        // STRAIGHT rgb (blend disabled, alpha => coverage), so un-premultiply the
        // composite. The strokeAlpha == 0 select returns the fill rgb directly (no
        // division), so an unstroked body is byte-identical to the pre-stroke path
        // -- A2C runs only in the opaque phase, where inp.color.a == 1, so outAlpha
        // there reduces to the SDF coverage exactly (matching the old alpha).
        let strokeDist = strokeRingDistance(dist, inp.strokeParams);
        let strokeCoverage = sdfCoverage(strokeDist, aa);
        let strokeAlpha = inp.strokeColor.a * strokeCoverage;
        let outAlpha = strokeAlpha + fillAlpha * (1.0 - strokeAlpha);

        // Discard pixels outside the shape AND the outer half of the AA edge band.
        if (outAlpha <= 0.001) {
          discard;
        }

        // IMPORTANT: In A2C mode, output UNPREMULTIPLIED color with alpha.
        // Blending is disabled; alpha is converted to coverage by the pipeline.
        let premulRgb = workingRgb(inp.strokeColor.rgb) * strokeAlpha + shadedRgb * fillAlpha * (1.0 - strokeAlpha);
        let outRgb = select(premulRgb / outAlpha, shadedRgb, strokeAlpha <= 0.0);
        return vec4<f32>(outRgb, outAlpha);
      }
      