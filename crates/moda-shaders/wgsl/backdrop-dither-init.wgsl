
        struct DitherUniforms {
          params: vec4<f32>, // x: pixelSizePx; y: brightness offset; z: contrast; w: pattern
          dims: vec4<u32>,   // x: texture width; y: texture height; z: grid width; w: grid height
        };

        @group(0) @binding(0) var<uniform> u: DitherUniforms;
        @group(0) @binding(1) var inputTexture: texture_2d<f32>;
        @group(0) @binding(2) var<storage, read_write> gray: array<f32>;

        fn adjustedLuma(color: vec4<f32>) -> f32 {
          var luma = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
          luma = (luma - 0.5) * u.params.z + 0.5 + u.params.y;
          return clamp(luma, 0.0, 1.0);
        }

        @compute @workgroup_size(8, 8)
        fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
          let gx = gid.x;
          let gy = gid.y;
          if (gx >= u.dims.z || gy >= u.dims.w) {
            return;
          }

          let pixelSize = max(u.params.x, 1.0);
          let startX = min(u32(ceil(f32(gx) * pixelSize)), u.dims.x);
          let startY = min(u32(ceil(f32(gy) * pixelSize)), u.dims.y);
          let endX = min(u32(ceil(f32(gx + 1u) * pixelSize)), u.dims.x);
          let endY = min(u32(ceil(f32(gy + 1u) * pixelSize)), u.dims.y);

          var sum = 0.0;
          var count = 0u;
          for (var y = startY; y < endY; y++) {
            for (var x = startX; x < endX; x++) {
              sum += adjustedLuma(textureLoad(inputTexture, vec2<i32>(i32(x), i32(y)), 0));
              count += 1u;
            }
          }
          gray[gy * u.dims.z + gx] = sum / max(f32(count), 1.0);
        }
      