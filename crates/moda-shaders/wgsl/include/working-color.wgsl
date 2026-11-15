// The engine's working colour (moda-shaders `working_color`), spliced by build.rs into every module that puts
// an authored colour into the frame. Authored colours are sRGB code values: `workingRgb` takes one,
// unassociated, to the encoding the engine blends in, and `workingToSrgb` takes a blended colour back.
// Textures are in that encoding already. An sRGB engine keeps the two definitions below, which return their
// argument. An engine in another encoding replaces the lines between the markers before it compiles the
// module.
// working-color begin
fn workingRgb(c: vec3<f32>) -> vec3<f32> { return c; }
fn workingToSrgb(c: vec3<f32>) -> vec3<f32> { return c; }
// working-color end
