
        struct DitherUniforms {
          params: vec4<f32>, // x: pixelSizePx; y: brightness offset; z: contrast; w: pattern
          dims: vec4<u32>,   // x: texture width; y: texture height; z: grid width; w: grid height
        };

        @group(0) @binding(0) var<uniform> u: DitherUniforms;
        @group(0) @binding(1) var<storage, read_write> gray: array<f32>;

        fn addError(x: i32, y: i32, err: f32) {
          if (x < 0 || y < 0 || x >= i32(u.dims.z) || y >= i32(u.dims.w)) {
            return;
          }
          let i = u32(y) * u.dims.z + u32(x);
          gray[i] = gray[i] + err;
        }

        @compute @workgroup_size(1)
        fn cs() {
          let pattern = u32(round(u.params.w));
          let useAtkinson = pattern == 3u;

          for (var y = 0u; y < u.dims.w; y++) {
            for (var x = 0u; x < u.dims.z; x++) {
              let i = y * u.dims.z + x;
              let old = gray[i];
              let quantized = select(0.0, 1.0, old > 0.5);
              gray[i] = quantized;
              let err = old - quantized;

              if (useAtkinson) {
                let share = err / 8.0;
                addError(i32(x) + 1, i32(y), share);
                addError(i32(x) + 2, i32(y), share);
                addError(i32(x) - 1, i32(y) + 1, share);
                addError(i32(x), i32(y) + 1, share);
                addError(i32(x) + 1, i32(y) + 1, share);
                addError(i32(x), i32(y) + 2, share);
              } else {
                addError(i32(x) + 1, i32(y), err * 7.0 / 16.0);
                addError(i32(x) - 1, i32(y) + 1, err * 3.0 / 16.0);
                addError(i32(x), i32(y) + 1, err * 5.0 / 16.0);
                addError(i32(x) + 1, i32(y) + 1, err * 1.0 / 16.0);
              }
            }
          }
        }
      