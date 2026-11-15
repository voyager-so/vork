//! An engine's working colour: the encoding it blends in, when that is not sRGB.
//!
//! Everything a document authors is sRGB code values, and each is converted where it enters the engine, before
//! anything filters or blends it:
//!
//! - A colour (solid and gradient paints, text, decorations, shadows, the page fill) enters in a shader. The
//!   modules that put one into the frame splice `wgsl/include/working-color.wgsl` and pass the unassociated
//!   colour through `workingRgb` before they associate and blend it. In an sRGB engine it returns its
//!   argument. An engine in another encoding replaces the include's definitions, once, before it compiles a
//!   module ([`specialize`]): the conversion is part of the engine and never changes, so it costs no uniform
//!   and no branch.
//! - A decoded bitmap enters at its upload ([`Conversion::apply`]). An image's texels are unassociated,
//!   converted, associated again and written as half floats, and its mip chain is built from them, so the
//!   sampler and the mips filter what the engine blends. A text raster's or an emoji's texels, which are
//!   straight alpha and have one level as the text shaders take them, are converted the same way and stay
//!   straight. The working encoding starts at the decoded bitmap, so a host hands the engine an image no
//!   one has resampled: the asset store's sizing of a still resamples its sRGB bytes, and
//!   `moda_assets::decode::prepare_raster_natural` prepares one without it. A picture a host hands over on
//!   the GPU (a capture) is the host's to supply in the working encoding.
//!
//! Pictures the engine draws itself (layer captures, blurs, rendered masks) are already in the working
//! encoding and are composited as they are. One module reads such a picture back as an authored colour: a
//! luminance mask is the luminance of the mask's picture as sRGB, so `mask-coverage` passes a rendered mask
//! through `workingToSrgb`, and the planner renders a luminance mask of a working-colour plan unless it is one
//! layer of one solid colour, whose luminance needs no picture. Procedural
//! shader fills, backdrop effects, image adjustments and blend modes other than source-over are not converted:
//! a host that sets a working colour does not author them.
use std::borrow::Cow;

/// One colour conversion: the sRGB transfer decoded, a change of primaries, the sRGB transfer encoded. Each
/// step is there or not. The transfer is the extended one (linear below zero, the power curve above one), so
/// signed and extended-range colour passes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conversion {
    pub decode: bool,
    /// The columns of the matrix between the two encodings' linear light; `None` between the same primaries.
    pub matrix: Option<[[f32; 3]; 3]>,
    pub encode: bool,
}
impl Conversion {
    /// One unassociated colour converted.
    pub fn apply(&self, mut rgb: [f32; 3]) -> [f32; 3] {
        if self.decode {
            rgb = rgb.map(|v| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) });
        }
        if let Some(m) = self.matrix {
            let v = rgb;
            rgb = std::array::from_fn(|row| m[0][row] * v[0] + m[1][row] * v[1] + m[2][row] * v[2]);
        }
        if self.encode {
            rgb = rgb.map(|v| if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1. / 2.4) - 0.055 });
        }
        rgb
    }
    /// The conversion as the WGSL function `fn <name>(c: vec3<f32>) -> vec3<f32>`, of unassociated colour.
    pub fn wgsl(&self, name: &str) -> String {
        let mut body = String::new();
        if self.decode {
            body.push_str("  v = select(pow(max((v + vec3<f32>(0.055)) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4)), v / 12.92, v <= vec3<f32>(0.04045));\n");
        }
        if let Some(m) = self.matrix {
            let column = |c: [f32; 3]| format!("vec3<f32>({:?}, {:?}, {:?})", c[0], c[1], c[2]);
            body.push_str(&format!("  v = mat3x3<f32>({}, {}, {}) * v;\n", column(m[0]), column(m[1]), column(m[2])));
        }
        if self.encode {
            body.push_str("  v = select(1.055 * pow(max(v, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055), v * 12.92, v <= vec3<f32>(0.0031308));\n");
        }
        format!("fn {name}(c: vec3<f32>) -> vec3<f32> {{\n  var v = c;\n{body}  return v;\n}}\n")
    }
}

/// The encoding an engine blends in, as its two conversions with sRGB.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkingColor {
    /// An authored sRGB colour as the working encoding.
    pub from_srgb: Conversion,
    /// A working-encoding colour as sRGB code values.
    pub to_srgb: Conversion,
}
impl WorkingColor {
    /// A straight-alpha sRGB colour as the working encoding, on the CPU: a pass's clear colour.
    pub fn rgba(&self, [r, g, b, a]: [f32; 4]) -> [f32; 4] {
        let [r, g, b] = self.from_srgb.apply([r, g, b]);
        [r, g, b, a]
    }
}

/// The line that opens the definitions of `wgsl/include/working-color.wgsl`, and the line that closes them.
const BEGIN: &str = "// working-color begin";
const END: &str = "// working-color end";

/// `source` for an engine that blends in `working`: the module as it is for sRGB (`None`), or for a module
/// that splices no working colour (it blends no authored colour). Otherwise the two definitions between the
/// include's markers are replaced by `working`'s conversions. A module holds the block once.
pub fn specialize(working: Option<&WorkingColor>, source: Cow<'static, str>) -> Cow<'static, str> {
    let Some(working) = working else { return source };
    let Some(begin) = source.find(BEGIN) else { return source };
    let body = begin + source[begin..].find('\n').expect("working-color: the opening line ends") + 1;
    let end = body + source[body..].find(END).expect("working-color: the definitions are closed");
    assert!(!source[end..].contains(BEGIN), "working-color: a module splices the include once");
    format!("{}{}{}{}", &source[..body], working.from_srgb.wgsl("workingRgb"), working.to_srgb.wgsl("workingToSrgb"), &source[end..]).into()
}
