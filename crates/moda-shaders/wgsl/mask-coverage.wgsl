// #include "working-color.wgsl"

        // One coverage draw of a mask (E7.18): the mask's placement and source, then the draw's paint. The
        // fragment resolves the paint's straight RGBA, turns it into coverage by the mask type (alpha: a;
        // luminance: luma(rgb) * a; a vector mask draws opaque white) and writes (coverage, 0, 0, a), which the
        // pipeline composites source-over, so a mask's fill and stroke layers stack as they draw.
        struct Globals {
          model: mat4x4<f32>,
          origin: vec2<f32>,
          invSize: vec2<f32>,
          source: u32,  // 0 = paint, 1 = text raster, 2 = image (sampled, times color)
          hasAncestorMask: u32,
          textOrigin: vec2<f32>,
          textInvSize: vec2<f32>,
          ancestorOrigin: vec2<f32>,
          ancestorInvSize: vec2<f32>,
          // Image-mode pattern placement (matches the visible pattern fill so the mask
          // silhouette lines up with the rendered image). Shape-local space.
          imagePatternOffset: vec2<f32>,
          imageTextureDims: vec2<f32>,
          imagePatternScale: vec2<f32>,
          imagePatternRotation: f32,
          luminance: u32,  // 1 = luminance mask
          // The draw's paint: a straight colour multiplier (a = layer x node opacity) and, for gradType > 0,
          // the path-gradient model (1 linear, 2 radial, 3 conic, 4 diamond; points in the 0-100 space of
          // gradBounds, node-local [x, y, w, h]; stops packed as vec4(offset, r, g, floor(b*255) + a*0.99999)).
          color: vec4<f32>,
          gradType: f32,
          gradRotation: f32,
          gradStart: vec2<f32>,
          gradEnd: vec2<f32>,
          numStops: u32,
          _pad: u32,
          gradBounds: vec4<f32>,
          stops: array<vec4<f32>, 8>,
        }
        @group(0) @binding(0) var<uniform> globals: Globals;
        @group(1) @binding(0) var textSampler: sampler;
        @group(1) @binding(1) var textTexture: texture_2d<f32>;
        @group(2) @binding(0) var ancestorSampler: sampler;
        @group(2) @binding(1) var ancestorMaskTexture: texture_2d<f32>;
        struct VSIn { @location(0) pos: vec2<f32> };
        struct VSOut {
          @builtin(position) position: vec4<f32>,
          @location(0) worldPos: vec2<f32>,
          @location(1) localPos: vec2<f32>,
        };
        @vertex fn vs_main(input: VSIn) -> VSOut {
          var out: VSOut;
          let world = globals.model * vec4<f32>(input.pos, 0.0, 1.0);
          out.worldPos = world.xy;
          out.localPos = input.pos;
          let uv = (world.xy - globals.origin) * globals.invSize;
          let clipX = uv.x * 2.0 - 1.0;
          let clipY = (1.0 - uv.y) * 2.0 - 1.0;
          out.position = vec4<f32>(clipX, clipY, 0.0, 1.0);
          return out;
        }

        fn unpackStop(stop: vec4<f32>) -> vec4<f32> {
          let blue = clamp(floor(stop.w) / 255.0, 0.0, 1.0);
          let alpha = clamp(fract(stop.w) / 0.99999, 0.0, 1.0);
          return vec4<f32>(clamp(stop.y, 0.0, 1.0), clamp(stop.z, 0.0, 1.0), blue, alpha);
        }

        // The path-gradient WGSL's interpolateGradient: pad the end stops, mix straight colours.
        fn gradientColor(t: f32) -> vec4<f32> {
          let n = min(globals.numStops, 8u);
          if (n == 0u) { return vec4<f32>(1.0, 1.0, 1.0, 1.0); }
          let first = globals.stops[0];
          if (n == 1u || t <= first.x) { return unpackStop(first); }
          let last = globals.stops[n - 1u];
          if (t >= last.x) { return unpackStop(last); }
          for (var i = 1u; i < n; i++) {
            let curr = globals.stops[i];
            let prev = globals.stops[i - 1u];
            if (t <= curr.x) {
              let span = curr.x - prev.x;
              var localT = 0.0;
              if (span > 0.001) {
                localT = (t - prev.x) / span;
              } else {
                localT = select(0.0, 1.0, t > (prev.x + curr.x) * 0.5);
              }
              return mix(unpackStop(prev), unpackStop(curr), localT);
            }
          }
          return unpackStop(last);
        }

        // The gradient's t at a node-local point (the path-gradient WGSL's calculate*T, with the box origin at
        // gradBounds.xy).
        fn gradientT(localPos: vec2<f32>) -> f32 {
          let size = max(globals.gradBounds.zw, vec2<f32>(1.0, 1.0));
          let p = localPos - globals.gradBounds.xy;
          let start = clamp(globals.gradStart, vec2<f32>(-100000.0, -100000.0), vec2<f32>(100000.0, 100000.0));
          let startLocal = start * 0.01 * size;
          if (globals.gradType == 1.0) {
            let endLocal = clamp(globals.gradEnd, vec2<f32>(-100000.0, -100000.0), vec2<f32>(100000.0, 100000.0)) * 0.01 * size;
            let dir = endLocal - startLocal;
            let lenSq = dot(dir, dir);
            if (lenSq <= 0.01) { return 0.0; }
            return dot(p - startLocal, dir) / lenSq;
          }
          if (globals.gradType == 3.0) {
            let rel = p - startLocal;
            let angle = atan2(rel.x, -rel.y) - globals.gradRotation;
            return fract(angle / 6.28318530717958647692);
          }
          // Radial (2) and diamond (4): placed by center, radii and rotation.
          let rel = p - startLocal;
          let c = cos(-globals.gradRotation);
          let s = sin(-globals.gradRotation);
          let r = vec2<f32>(rel.x * c - rel.y * s, rel.x * s + rel.y * c);
          let radii = max(vec2<f32>(1.0, 1.0), clamp(globals.gradEnd, vec2<f32>(1.0, 1.0), vec2<f32>(100000.0, 100000.0)) * 0.01 * size);
          let n = r / radii;
          if (globals.gradType == 4.0) { return abs(n.x) + abs(n.y); }
          return length(n);
        }

        @fragment fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
          var paint = globals.color;

          if (globals.source == 1u) {
            // Text: the raster (straight RGBA) at world-UV against its logical size.
            let uvText = (input.worldPos - globals.textOrigin) * globals.textInvSize;
            if (uvText.x < 0.0 || uvText.x > 1.0 || uvText.y < 0.0 || uvText.y > 1.0) { discard; }
            paint = paint * textureSample(textTexture, textSampler, uvText);
          } else if (globals.source == 2u || globals.source == 3u) {
            // Image mode: UVs are computed exactly like the visible pattern fill (WebGPUPathShaderFactory)
            // from the shape-local position, so the mask silhouette matches the rendered image under
            // offset / scale / rotation: offset in shape space, normalize by the texture dimensions, rotate,
            // then scale. No-repeat clips outside [0,1]. Fragments come from the tessellated fill geometry,
            // so the image is also clipped to the shape (and its corner radius).
            var uvImg = (input.localPos - globals.imagePatternOffset) / globals.imageTextureDims;
            let rot = globals.imagePatternRotation;
            if (abs(rot) > 0.001) {
              let cosr = cos(rot);
              let sinr = sin(rot);
              uvImg = vec2<f32>(uvImg.x * cosr - uvImg.y * sinr, uvImg.x * sinr + uvImg.y * cosr);
            }
            uvImg = uvImg / globals.imagePatternScale;
            // Sample unconditionally (textureSample needs uniform control flow), then clip
            // no-repeat by zeroing outside [0,1].
            var texel = textureSample(textTexture, textSampler, uvImg);
            if (globals.source == 3u && texel.a > 0.0) {
              // A rendered mask is the engine's own picture, in its working encoding: luminance is of sRGB.
              texel = vec4<f32>(workingToSrgb(texel.rgb / texel.a), texel.a);
            }
            let inBounds = uvImg.x >= 0.0 && uvImg.x <= 1.0 && uvImg.y >= 0.0 && uvImg.y <= 1.0;
            paint = paint * select(vec4<f32>(0.0), texel, inBounds);
          } else if (globals.gradType > 0.0) {
            paint = paint * gradientColor(gradientT(input.localPos));
          }

          var coverage = paint.a;
          if (globals.luminance == 1u) {
            // CSS Masking's luminance coefficients, on the sRGB-encoded colour, times alpha.
            coverage = dot(clamp(paint.rgb, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.2125, 0.7154, 0.0721)) * paint.a;
          }

          // If we have an ancestor mask, sample and multiply by its coverage
          // This ensures nested containers properly compose their clip regions
          if (globals.hasAncestorMask == 1u) {
            let uvAncestor = (input.worldPos - globals.ancestorOrigin) * globals.ancestorInvSize;
            // If outside ancestor bounds, discard (fully clipped)
            if (uvAncestor.x < 0.0 || uvAncestor.x > 1.0 || uvAncestor.y < 0.0 || uvAncestor.y > 1.0) {
              discard;
            }
            let ancestorCoverage = textureSample(ancestorMaskTexture, ancestorSampler, uvAncestor).r;
            coverage = coverage * ancestorCoverage;
          }

          // Nothing to composite.
          if (paint.a < 0.001) { discard; }

          return vec4<f32>(coverage, 0.0, 0.0, paint.a);
        }
