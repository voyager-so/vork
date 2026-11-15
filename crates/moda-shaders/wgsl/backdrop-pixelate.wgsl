
        struct VSOut {
          @builtin(position) position: vec4<f32>,
        };

        struct PixelateUniforms {
          // Rotated AABB of the node in sample-texture pixel coords (matches
          // refraction.nodeRect / composite.nodeRect convention).
          nodeRect: vec4<f32>,
          // x: rotation (rad, canvas-space, Y-down — matches affineFromTRS);
          // y: local half-W (px); z: local half-H (px); w: mirrored-frame Y sign (+1 / -1).
          // Reserved for future linear-pixelate gradient direction projection.
          shapeTransform: vec4<f32>,
          // x: pixelSizePx. y/z/w reserved for future gradient stops
          // (start offset, end offset, stop count).
          params: vec4<f32>,
        };

        @group(0) @binding(0) var<uniform> u: PixelateUniforms;
        @group(0) @binding(1) var uSampler: sampler;
        @group(0) @binding(2) var uTexture: texture_2d<f32>;

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

        // Per-fragment pixel size. Today it's a constant; tomorrow this is
        // where the linear-pixelate gradient lookup will plug in. Keep it as
        // a single helper so adding gradient is a one-function change without
        // restructuring the rest of the shader.
        fn pixelSizeAt(localPos: vec2<f32>) -> f32 {
          return max(u.params.x, 1.0);
        }

        @fragment
        fn fs(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
          let dims = vec2<f32>(textureDimensions(uTexture));
          let aabbCenter = u.nodeRect.xy + u.nodeRect.zw * 0.5;
          let centered = fragCoord.xy - aabbCenter;

          // Inverse-rotate into the shape's local frame so the grid stays
          // aligned with the node, not the screen. Mirrors the convention in
          // sdShape (see GLASS_HELPERS_WGSL above).
          let rotationRad = u.shapeTransform.x;
          let yFrameSign = u.shapeTransform.w;
          let cosR = cos(rotationRad);
          let sinR = sin(rotationRad);
          let local = vec2<f32>(
            centered.x * cosR + centered.y * sinR,
            (-centered.x * sinR + centered.y * cosR) * yFrameSign,
          );

          let p = pixelSizeAt(local);
          let snappedLocal = (floor(local / p) + 0.5) * p;

          // Rotate the snapped point back to AABB-pixel space, then to UV.
          let snappedCentered = vec2<f32>(
            snappedLocal.x * cosR - snappedLocal.y * yFrameSign * sinR,
            snappedLocal.x * sinR + snappedLocal.y * yFrameSign * cosR,
          );
          let snappedFrag = aabbCenter + snappedCentered;
          let uv = clamp(snappedFrag / dims, vec2<f32>(0.0), vec2<f32>(1.0));

          return textureSample(uTexture, uSampler, uv);
        }
      