//! Compiler-certified complete-expression extinction bounds. No GPU probes.
use super::*;

pub(crate) fn certify(
    program: &MediumProgram,
    words: &[Option<u32>],
    bounds: (Vec3, Vec3),
    times: [f32; 2],
) -> Result<f32> {
    program.extinction_majorant(&programs::abi(),words,
        std::array::from_fn(|i|[bounds.0[i],bounds.1[i]]),times)
}

/// Include every render view's shutter, including captured render images.
pub(crate) fn certification_times(scene:&v2::Scene)->[f32;2] {
    std::iter::once(scene.render.clone())
        .chain(scene.images.values().filter_map(|image|image.resolved_render(&scene.render)))
        .map(|render| {
            let shutter=render.motion_blur.as_ref().map_or(0.,|m|m.shutter/render.fps.as_f32());
            let offset=render.motion_blur.as_ref().map_or(0.,|m|m.position.offset());
            [(offset-0.5)*shutter,render.duration.as_f32()+(offset+0.5)*shutter]
        }).fold([0.,0.],|[a,b],[c,d]|[a.min(c),b.max(d)])
}
