
        struct VSOut {
          @builtin(position) position: vec4<f32>,
        };

        struct RefractionUniforms {
          nodeRect: vec4<f32>,
          cornerRadii: vec4<f32>,
          params: vec4<f32>, // x: refraction px, y: depth px, z: mode (0=barrel, 1=lens, 2=swirl), w: unused
          shapeParams: vec4<u32>, // x: shape type (0=rounded rect, 1=ellipse); y/z/w reserved
          // x: rotation radians (canvas-space, Y-down); y: local half-width px;
          // z: local half-height px; w: mirrored-frame Y sign (+1 or -1).
          // The local half-extents are
          // the shape's unrotated half-size in pixel space — distinct from
          // nodeRect.zw which is the rotated AABB extent in pixel space.
          shapeTransform: vec4<f32>,
        };

        @group(0) @binding(0) var<uniform> refraction: RefractionUniforms;
        @group(0) @binding(1) var uSampler: sampler;
        @group(0) @binding(2) var uTexture: texture_2d<f32>;
        // Arbitrary-shape coverage (shapeType == 2). 1x1 fallback for SDF shapes.
        @group(0) @binding(3) var sharpCoverageTex: texture_2d<f32>;
        @group(0) @binding(4) var fieldCoverageTex: texture_2d<f32>;

        @vertex
        fn vs(@builtin(vertex_index) vid: u32) -> VSOut {
          var pos = array<vec2<f32>, 3>(
            vec2<f32>(-1.0, -1.0),
            vec2<f32>( 3.0, -1.0),
            vec2<f32>(-1.0,  3.0)
          );
          var out: VSOut;
          out.position = vec4<f32>(pos[vid], 0.0, 1.0);
          return out;
        }

        
// SDF for a rounded rectangle with different corner radii
fn sdRoundedBoxNonUniform(p: vec2<f32>, halfSize: vec2<f32>, radii: vec4<f32>) -> f32 {
  // radii: x=top-left, y=top-right, z=bottom-right, w=bottom-left
  
  // Determine which quadrant we're in
  let signX = sign(p.x);
  let signY = sign(p.y);
  
  // Select the appropriate corner radius based on quadrant
  var radius: f32;
  if (signX < 0.0 && signY < 0.0) {
    radius = radii.x; // top-left
  } else if (signX > 0.0 && signY < 0.0) {
    radius = radii.y; // top-right
  } else if (signX > 0.0 && signY > 0.0) {
    radius = radii.z; // bottom-right
  } else {
    radius = radii.w; // bottom-left
  }
  
  // Calculate distance using the selected radius
  let q = abs(p) - halfSize + radius;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// SDF for a rounded rectangle with uniform corner radius
fn sdRoundedBox(p: vec2<f32>, halfSize: vec2<f32>, radius: f32) -> f32 {
  let q = abs(p) - halfSize + radius;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// SDF for a regular rectangle (no rounded corners)
fn sdBox(p: vec2<f32>, halfSize: vec2<f32>) -> f32 {
  let q = abs(p) - halfSize;
  return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
}

// SDF for a circle
fn sdCircle(p: vec2<f32>, radius: f32) -> f32 {
  return length(p) - radius;
}

// SDF for an ellipse.
// Guards against degenerate radii and a zero-length p (which would otherwise
// produce NaN from the k1 divide) so callers can safely sample every pixel
// inside the bounding box, including the exact ellipse center.
fn sdEllipse(p: vec2<f32>, radii: vec2<f32>) -> f32 {
  if (radii.x < 1e-4 || radii.y < 1e-4) { return 1.0; }
  let k0 = length(p / radii);
  let k1 = length(p / (radii * radii));
  if (k1 < 1e-4) { return -min(radii.x, radii.y); }
  return k0 * (k0 - 1.0) / k1;
}

        
  // Shape-type dispatch. Keep the branch values in sync with the TS
  // SHAPE_TYPE_* constants. Adding a new shape is "one more branch" here
  // (and a matching tag on the TS side) — the rest of the pipeline is
  // shape-agnostic and reads the SDF through this entry point.
  //   - shapeType = 0 -> rounded rectangle (uses cornerRadii)
  //   - shapeType = 1 -> axis-aligned ellipse (cornerRadii ignored; halfSize is the ellipse radii)
  //
  // Rotation is handled here as an orthogonal transform on the sample point:
  // callers pass the AABB-centered sample point plus the node local X-axis
  // angle in radians; we rotate back into the shape's local (unrotated) frame
  // and then apply a mirrored-frame Y sign so reflected orthogonal transforms
  // are reconstructed correctly before per-shape dispatch.
  // halfSize is the *local* (unrotated) half-extent of the shape, not the
  // AABB half-extent. Future shapes slot in as another branch here.
  //
  // The rotation is inverted (R(-theta) in Y-down canvas space, which matches
  // affineFromTRS): canvas-space rotation maps local to canvas, and we need
  // the inverse to go the other way.
  fn sdShape(
    shapeType: u32,
    centered: vec2<f32>,
    halfSize: vec2<f32>,
    cornerRadii: vec4<f32>,
    rotationRad: f32,
    localFrameYSign: f32
  ) -> f32 {
    let cosR = cos(rotationRad);
    let sinR = sin(rotationRad);
    let local = vec2<f32>(
      centered.x * cosR + centered.y * sinR,
      (-centered.x * sinR + centered.y * cosR) * localFrameYSign,
    );
    if (shapeType == 1u) {
      return sdEllipse(local, halfSize);
    }
    return sdRoundedBoxNonUniform(local, halfSize, cornerRadii);
  }

  // Finite-difference the SDF in the AABB-centered frame. Because sdShape
  // internally rotates, the gradient we recover is naturally in the AABB
  // frame — which is what downstream refraction/edge-lighting math wants
  // (UVs and the light direction are both expressed in screen/AABB space).
  fn estimateShapeNormal(
    shapeType: u32,
    centered: vec2<f32>,
    halfSize: vec2<f32>,
    cornerRadii: vec4<f32>,
    rotationRad: f32,
    localFrameYSign: f32
  ) -> vec2<f32> {
    let eps = 1.0;
    let dx =
      sdShape(shapeType, centered + vec2<f32>(eps, 0.0), halfSize, cornerRadii, rotationRad, localFrameYSign) -
      sdShape(shapeType, centered - vec2<f32>(eps, 0.0), halfSize, cornerRadii, rotationRad, localFrameYSign);
    let dy =
      sdShape(shapeType, centered + vec2<f32>(0.0, eps), halfSize, cornerRadii, rotationRad, localFrameYSign) -
      sdShape(shapeType, centered - vec2<f32>(0.0, eps), halfSize, cornerRadii, rotationRad, localFrameYSign);
    let gradient = vec2<f32>(dx, dy);
    let gradientLength = length(gradient);
    if (gradientLength > 1e-4) {
      return gradient / gradientLength;
    }
    let fallback = centered + vec2<f32>(1e-4, 0.0);
    return fallback / max(length(fallback), 1e-4);
  }

  fn edgeFactor(distance: f32, depthPx: f32, alpha: f32) -> f32 {
    if (depthPx <= 1e-4) {
      return 0.0;
    }
    let insideDistance = max(-distance, 0.0);
    return alpha * (1.0 - smoothstep(0.0, depthPx, insideDistance));
  }

  // Arbitrary-shape coverage path (shapeType == 2). Star / polygon / path
  // silhouettes can't be expressed as an analytic SDF, so the renderer
  // tessellates the fill into a coverage texture and the shaders read the
  // shape's (alpha, normal, distance) from it instead of sdShape:
  //   - sharpTex  : 1-bit-ish silhouette; .r is coverage, used directly as alpha
  //                 (and for masking in the composite). Confines every effect to
  //                 the outline + corner radius.
  //   - fieldTex  : a Gaussian-blurred copy of the silhouette (blur radius ~ glass
  //                 depth). Its smooth ramp yields the edge normal (gradient) and
  //                 an inside-distance proxy for the glass refraction / rim /
  //                 dispersion bands. A blurred binary mask has gradient magnitude
  //                 ~1/pixel near the edge, so (coverage - 0.5) is ≈ the signed
  //                 distance there — an approximation, but monotonic and
  //                 edge-anchored, which is what the band math needs.
  // needsField is false for blur / pixelate / dither (alpha-only); the gradient
  // taps are skipped so the common case stays cheap.
  struct ShapeFieldSample {
    alpha: f32,
    normal: vec2<f32>,
    distance: f32,
  };

  fn sampleShapeCoverage(
    sharpTex: texture_2d<f32>,
    fieldTex: texture_2d<f32>,
    samp: sampler,
    uv: vec2<f32>,
    depthPx: f32,
    needsField: bool
  ) -> ShapeFieldSample {
    var out: ShapeFieldSample;
    out.alpha = clamp(textureSampleLevel(sharpTex, samp, uv, 0.0).r, 0.0, 1.0);
    out.normal = vec2<f32>(0.0, 0.0);
    out.distance = 0.0;
    if (!needsField) {
      return out;
    }
    let dims = max(vec2<f32>(textureDimensions(fieldTex)), vec2<f32>(1.0));
    let texel = vec2<f32>(1.0) / dims;
    let c = textureSampleLevel(fieldTex, samp, uv, 0.0).r;
    let cx1 = textureSampleLevel(fieldTex, samp, uv + vec2<f32>(texel.x, 0.0), 0.0).r;
    let cx0 = textureSampleLevel(fieldTex, samp, uv - vec2<f32>(texel.x, 0.0), 0.0).r;
    let cy1 = textureSampleLevel(fieldTex, samp, uv + vec2<f32>(0.0, texel.y), 0.0).r;
    let cy0 = textureSampleLevel(fieldTex, samp, uv - vec2<f32>(0.0, texel.y), 0.0).r;
    // Gradient of the blurred coverage points inward (toward higher coverage);
    // sdShape's normal points outward, so negate to match that convention.
    let grad = vec2<f32>(cx1 - cx0, cy1 - cy0);
    let gradLen = length(grad);
    if (gradLen > 1e-5) {
      out.normal = -grad / gradLen;
    }
    // Inside-distance proxy: blurred coverage crosses ~0.5 at the true edge and
    // ramps to 1 deep inside. Encode as a negative distance so edgeFactor()'s
    // max(-distance, 0) recovers the inside depth unchanged.
    let insideDist = clamp((c - 0.5) * 2.0, 0.0, 1.0) * max(depthPx, 1.0);
    out.distance = -insideDist;
    return out;
  }


        @fragment
        fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
          let dims = vec2<f32>(textureDimensions(uTexture));
          let uv = clamp(fragCoord.xy / dims, vec2<f32>(0.0), vec2<f32>(1.0));
          // nodeRect is the rotated AABB in pixel space. The shape's rotation
          // center is the AABB center (rotation of a symmetric local rect about
          // its own center produces an AABB symmetric about that same center).
          let aabbHalf = refraction.nodeRect.zw * 0.5;
          let centered = fragCoord.xy - (refraction.nodeRect.xy + aabbHalf);
          // Use the *local* (unrotated) half-size for the SDF dispatch — the
          // rotation is applied inside sdShape, so the per-shape SDFs always
          // see canonical axis-aligned coordinates.
          let halfSize = refraction.shapeTransform.yz;
          let rotationRad = refraction.shapeTransform.x;
          let localFrameYSign = refraction.shapeTransform.w;
          let shapeType = refraction.shapeParams.x;
          let depthPx = refraction.params.y;

          // Shape coverage vs analytic SDF. The refraction pass only runs for
          // glass, so the coverage field (normal + inside distance) is always
          // needed here.
          var alpha: f32;
          var normal: vec2<f32>;
          var distance: f32;
          if (shapeType == 2u) {
            let s = sampleShapeCoverage(sharpCoverageTex, fieldCoverageTex, uSampler, uv, depthPx, true);
            alpha = s.alpha;
            normal = s.normal;
            distance = s.distance;
          } else {
            distance = sdShape(shapeType, centered, halfSize, refraction.cornerRadii, rotationRad, localFrameYSign);
            alpha = 1.0 - smoothstep(-0.5, 0.5, distance);
            normal = estimateShapeNormal(
              shapeType,
              centered,
              halfSize,
              refraction.cornerRadii,
              rotationRad,
              localFrameYSign
            );
          }
          let factor = edgeFactor(distance, refraction.params.y, alpha);
          let tangent = vec2<f32>(normal.y, -normal.x);
          let mode = refraction.params.z;

          // Swirl: tangential S-curve. Pixels slide sideways along the edge,
          // reversing direction at the midpoint of the depth band. Lines
          // perpendicular to the edge are unaffected (tangent is along the line);
          // angled lines get the characteristic S-distortion.
          let insideDist = max(-distance, 0.0);
          var swirlFactor = 0.0;
          if (depthPx > 1e-4) {
            let t = clamp(insideDist / depthPx, 0.0, 1.0);
            swirlFactor = sin(t * 6.28318) * alpha;
          }

          let swirlDisp = tangent * (refraction.params.x * swirlFactor) / dims;
          let lensDisp = normal * (refraction.params.x * factor) / dims;
          let barrelDisp = -normal * (refraction.params.x * factor) / dims;

          let displacement = select(select(barrelDisp, lensDisp, mode > 0.5), swirlDisp, mode > 1.5);
          let displacedUv = clamp(uv + displacement, vec2<f32>(0.0), vec2<f32>(1.0));
          return textureSample(uTexture, uSampler, displacedUv);
        }
      