
        const BAYER4_MATRIX: array<f32, 16> = array<f32, 16>(
          0.0, 128.0, 32.0, 160.0,
          192.0, 64.0, 224.0, 96.0,
          48.0, 176.0, 16.0, 144.0,
          240.0, 112.0, 208.0, 80.0
        );

        struct DitherUniforms {
          params: vec4<f32>, // x: pixelSizePx; y: brightness offset; z: contrast; w: pattern
          dims: vec4<u32>,   // x: texture width; y: texture height; z: grid width; w: grid height
        };

        @group(0) @binding(0) var<uniform> u: DitherUniforms;
        @group(0) @binding(1) var<storage, read_write> gray: array<f32>;

        fn bayer4Threshold(x: u32, y: u32) -> f32 {
          let index = (y % 4u) * 4u + (x % 4u);
          return (BAYER4_MATRIX[index] + 0.5) / 255.0;
        }

        @compute @workgroup_size(8, 8)
        fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
          let gx = gid.x;
          let gy = gid.y;
          if (gx >= u.dims.z || gy >= u.dims.w) {
            return;
          }

          let i = gy * u.dims.z + gx;
          let threshold = select(0.5, bayer4Threshold(gx, gy), u32(round(u.params.w)) == 1u);
          gray[i] = select(0.0, 1.0, gray[i] > threshold);
        }
      