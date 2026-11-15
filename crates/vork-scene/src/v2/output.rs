//! Render-call destinations; these are not scene fields.
use super::*;
use std::path::PathBuf;

pub(super) fn optional_present<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where D: serde::Deserializer<'de>, T: serde::Deserialize<'de> {
    T::deserialize(deserializer).map(Some)
}

/// Display encoding owned by a destination, resolved together with composition.view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum OutputDisplay {
    #[default]
    #[serde(rename = "sRGB")]
    Srgb,
    #[serde(rename = "Display P3")]
    DisplayP3,
    #[serde(rename = "Rec.2100-PQ")]
    Rec2100Pq,
    #[serde(rename = "Rec.2100-HLG")]
    Rec2100Hlg,
}
/// RGB encodings with pinned primaries and white points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum WorkingColorSpace {
    #[serde(rename = "Linear Rec.709")]
    LinearRec709,
    #[serde(rename = "Linear Rec.2020")]
    LinearRec2020,
    ACEScg,
    #[serde(rename = "sRGB")]
    Srgb,
    #[serde(rename = "Display P3")]
    DisplayP3,
}
impl WorkingColorSpace {
    /// The authored spelling.
    pub fn name(self) -> &'static str {
        match self { Self::LinearRec709 => "Linear Rec.709", Self::LinearRec2020 => "Linear Rec.2020", Self::ACEScg => "ACEScg", Self::Srgb => "sRGB", Self::DisplayP3 => "Display P3" }
    }
    /// Whether values are linear light; sRGB and Display P3 carry the sRGB transfer.
    pub fn is_linear(self) -> bool { !matches!(self, Self::Srgb | Self::DisplayP3) }
    /// The display whose code values this encoding is: sRGB's and Display P3's. A linear space is no
    /// display's encoding.
    pub fn display(self) -> Option<OutputDisplay> {
        match self { Self::Srgb => Some(OutputDisplay::Srgb), Self::DisplayP3 => Some(OutputDisplay::DisplayP3), _ => None }
    }
    /// CIE xy chromaticities: red, green, blue, adopted white.
    pub fn chromaticities(self) -> [[f32; 2]; 4] {
        match self {
            Self::LinearRec709 | Self::Srgb => [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06], [0.3127, 0.3290]],
            Self::LinearRec2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046], [0.3127, 0.3290]],
            Self::ACEScg => [[0.713, 0.293], [0.165, 0.830], [0.128, 0.044], [0.32168, 0.33767]],
            Self::DisplayP3 => [[0.68, 0.32], [0.265, 0.69], [0.15, 0.06], [0.3127, 0.3290]],
        }
    }
    /// This space's linear light as linear Rec.709: the pinned OCIO 2.6.0 configuration's matrices (Blender
    /// 5.2.1), with ACEScg's D60 adapted to D65.
    fn to_rec709(self) -> glam::DMat3 {
        let rows = match self {
            Self::LinearRec709 | Self::Srgb => return glam::DMat3::IDENTITY,
            Self::LinearRec2020 => [[1.66049087, -0.58764118, -0.07284992], [-0.12455040, 1.13289988, -0.00834931], [-0.01815077, -0.10057894, 1.11872971]],
            Self::ACEScg => [[1.70505095, -0.62179214, -0.08325893], [-0.13025640, 1.14080477, -0.01054827], [-0.02400335, -0.12896900, 1.15297234]],
            Self::DisplayP3 => [[1.22494018, -0.22494021, 0.], [-0.04205693, 1.04205692, 0.], [-0.01963755, -0.07863605, 1.09827363]],
        };
        glam::DMat3::from_cols_array_2d(&rows).transpose()
    }
    /// The matrix from this space's linear light to `destination`'s: a colourimetric change of primaries that
    /// keeps signed and extended-range values. `None` between spaces with the same primaries, whose values
    /// pass as they are.
    pub fn linear_matrix(self, destination: Self) -> Option<glam::Mat3> {
        let (from, to) = (self.to_rec709(), destination.to_rec709());
        (from != to).then(|| (to.inverse() * from).as_mat3())
    }
    /// The colourimetric conversion of one colour from this space to `destination`, prepared once for many
    /// colours: this space's transfer decoded, the change of primaries, `destination`'s transfer encoded.
    /// Signed and extended-range values are kept (the sRGB transfer is linear below zero and its power curve
    /// continues above one). Alpha is the caller's.
    pub fn conversion(self, destination: Self) -> ColorConversion {
        let same = self == destination;
        ColorConversion { decode: !same && !self.is_linear(), matrix: self.linear_matrix(destination), encode: !same && !destination.is_linear() }
    }
    /// One colour converted to `destination` (`conversion`).
    pub fn convert_rgb(self, destination: Self, rgb: [f32; 3]) -> [f32; 3] { self.conversion(destination).apply(rgb) }
}
/// `WorkingColorSpace::conversion`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorConversion {
    decode: bool,
    matrix: Option<glam::Mat3>,
    encode: bool,
}
impl ColorConversion {
    /// Whether colours pass as they are, bit for bit.
    pub fn is_identity(&self) -> bool { !self.decode && self.matrix.is_none() && !self.encode }
    /// The conversion's steps in order, for a caller that runs it elsewhere (a shader): whether the sRGB
    /// transfer is decoded, the change of primaries, whether the sRGB transfer is encoded.
    pub fn steps(&self) -> (bool, Option<glam::Mat3>, bool) { (self.decode, self.matrix, self.encode) }
    /// Whether the conversion is linear (a matrix or nothing): it then commutes with alpha association. One
    /// that decodes or encodes a transfer curve does not, and takes unassociated colour.
    pub fn is_linear(&self) -> bool { !self.decode && !self.encode }
    pub fn apply(&self, mut rgb: [f32; 3]) -> [f32; 3] {
        if self.decode { rgb = rgb.map(|v| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }); }
        if let Some(matrix) = self.matrix { rgb = (matrix * glam::Vec3::from_array(rgb)).to_array(); }
        if self.encode { rgb = rgb.map(|v| if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1. / 2.4) - 0.055 }); }
        rgb
    }
}
impl OutputDisplay {
    pub fn is_hdr(self) -> bool { matches!(self, Self::Rec2100Pq | Self::Rec2100Hlg) }
    /// The authored spelling.
    pub fn name(self) -> &'static str {
        match self { Self::Srgb => "sRGB", Self::DisplayP3 => "Display P3", Self::Rec2100Pq => "Rec.2100-PQ", Self::Rec2100Hlg => "Rec.2100-HLG" }
    }
}
/// A render-call destination. Raw destinations bypass the view and have no default display.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OutputDestination {
    #[schemars(extend("x-source" = "vork render-call destination", "x-help-path" = "output.file", "x-help" = "path; format follows extension"))]
    pub file: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "optional_present")]
    #[schemars(with = "OutputDisplay", extend("x-source" = "vork output target", "x-help-path" = "output.display", "x-help" = "sRGB|Display P3|Rec.2100-PQ|Rec.2100-HLG; omitted: sRGB for viewed outputs; raw outputs bypass the view"))]
    pub display: Option<OutputDisplay>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "optional_present")]
    #[schemars(with = "f32", extend("x-source" = "vork colour contract (docs/design/colour.md)", "x-help-path" = "output.peak_luminance", "x-help" = "positive cd/m²; omitted: SDR 100, HDR 1000"))]
    pub peak_luminance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "optional_present")]
    #[schemars(with = "f32", extend("x-source" = "vork colour contract (docs/design/colour.md)", "x-help-path" = "output.reference_white", "x-help" = "positive cd/m², at most peak_luminance; omitted: SDR 100, HDR 203"))]
    pub reference_white: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "optional_present")]
    #[schemars(with = "WorkingColorSpace", extend("x-source" = "vork colour contract (docs/design/colour.md)", "x-help-path" = "output.color_space", "x-help" = "EXR storage space; omitted: the composition's working space; a world takes the linear spaces, a canvas any; excludes display and luminance fields"))]
    pub color_space: Option<WorkingColorSpace>,
}
impl OutputDestination {
    pub fn display_target(&self) -> OutputDisplay { self.display.unwrap_or_default() }
    pub fn peak(&self) -> f32 { self.peak_luminance.unwrap_or(if self.display_target().is_hdr() { 1000. } else { 100. }) }
    pub fn white(&self) -> f32 { self.reference_white.unwrap_or(if self.display_target().is_hdr() { 203. } else { 100. }) }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.color_space.is_none() || (self.display.is_none() && self.peak_luminance.is_none() && self.reference_white.is_none()),
            "output.color_space excludes output.display, peak_luminance and reference_white");
        ensure!(self.peak().is_finite() && self.peak() > 0., "output.peak_luminance must be positive and finite");
        ensure!(self.white().is_finite() && self.white() > 0. && self.white() <= self.peak(), "output.reference_white must be positive, finite and at most peak_luminance");
        ensure!(!self.display_target().is_hdr() || self.peak() <= 10000., "output.peak_luminance must not exceed 10000 cd/m² for Rec.2100");
        Ok(())
    }
}
impl From<PathBuf> for OutputDestination {
    fn from(file: PathBuf) -> Self {
        Self {
            file,
            display: None, peak_luminance: None, reference_white: None, color_space: None,
        }
    }
}
impl From<&str> for OutputDestination {
    fn from(file: &str) -> Self {
        PathBuf::from(file).into()
    }
}
impl std::ops::Deref for OutputDestination {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.file
    }
}
impl std::str::FromStr for OutputDestination {
    type Err = String;
    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value.trim_start().starts_with('{') {
            serde_json::from_str(value).map_err(|e| format!("output descriptor: {e}"))
        } else {
            Ok(PathBuf::from(value).into())
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn primaries_conversion_matches_the_pinned_ocio_configuration() {
        use WorkingColorSpace as S;
        let value = glam::Vec3::new(2., -0.25, 0.5);
        // Independent OCIO 2.6.0 processors on the pinned Blender 5.2.1 configuration: signed and above one.
        for (space, expected) in [(S::LinearRec2020, [1.19414353, -0.08600947, 0.45857722]), (S::ACEScg, [1.16500378, -0.08197488, 0.44874606])] {
            let converted = S::LinearRec709.linear_matrix(space).unwrap() * value;
            for (a, b) in converted.to_array().into_iter().zip(expected) { assert!((a - b).abs() < 2e-6, "{space:?}: {a} against {b}"); }
            let back = space.linear_matrix(S::LinearRec709).unwrap() * converted;
            for (a, b) in back.to_array().into_iter().zip(value.to_array()) { assert!((a - b).abs() < 2e-6, "{space:?}: {a} against {b}"); }
        }
        // The same primaries have no matrix: values pass bit for bit.
        assert!(S::LinearRec709.linear_matrix(S::LinearRec709).is_none() && S::Srgb.linear_matrix(S::LinearRec709).is_none());
        assert!(S::ACEScg.linear_matrix(S::ACEScg).is_none() && S::DisplayP3.linear_matrix(S::Srgb).is_some());
        assert!([S::LinearRec709, S::LinearRec2020, S::ACEScg].into_iter().all(S::is_linear) && !S::Srgb.is_linear() && !S::DisplayP3.is_linear());
        // The encoded spaces are each a display's code values; a linear space is none's.
        assert_eq!((S::Srgb.display(), S::DisplayP3.display()), (Some(OutputDisplay::Srgb), Some(OutputDisplay::DisplayP3)));
        assert!([S::LinearRec709, S::LinearRec2020, S::ACEScg].into_iter().all(|space| space.display().is_none()));
        // A conversion names its steps: sRGB to ACEScg decodes and changes primaries; the way back encodes.
        let (decode, matrix, encode) = S::Srgb.conversion(S::ACEScg).steps();
        assert!(decode && !encode && matrix == S::LinearRec709.linear_matrix(S::ACEScg));
        assert_eq!(S::LinearRec709.conversion(S::Srgb).steps(), (false, None, true));
        assert_eq!(S::ACEScg.chromaticities()[3], [0.32168, 0.33767]);
    }
    #[test]
    fn conversion_keeps_signed_and_extended_colour_and_identity_bits() {
        use WorkingColorSpace as S;
        let value = [2., -0.25, 0.5];
        for (space, expected) in [(S::LinearRec2020, [1.19414353, -0.08600947, 0.45857722]), (S::ACEScg, [1.16500378, -0.08197488, 0.44874606])] {
            let actual = S::LinearRec709.convert_rgb(space, value);
            for (a, b) in actual.into_iter().zip(expected) { assert!((a - b).abs() < 2e-6, "{space:?}: {a} against {b}"); }
        }
        let signed = [-0., -0.2, 1000.];
        for space in [S::LinearRec709, S::Srgb, S::ACEScg] {
            assert!(space.conversion(space).is_identity());
            assert_eq!(space.convert_rgb(space, signed).map(f32::to_bits), signed.map(f32::to_bits));
        }
        // The encoded spaces keep values below zero and above one, and return them.
        for space in [S::Srgb, S::DisplayP3] {
            let encoded = S::LinearRec709.convert_rgb(space, value);
            assert!(encoded.iter().any(|x| *x < 0.) && encoded.iter().any(|x| *x > 1.), "{space:?}: {encoded:?}");
            let back = space.convert_rgb(S::LinearRec709, encoded);
            for (a, b) in back.into_iter().zip(value) { assert!((a - b).abs() < 4e-6, "{space:?}: {a} against {b}"); }
        }
        // sRGB and linear Rec.709 share primaries: only the transfer runs.
        assert!((S::Srgb.convert_rgb(S::LinearRec709, [0.5, 0.5, 0.5])[0] - 0.21404114).abs() < 1e-6);
    }
    #[test]
    fn target_defaults_and_invalid_luminance_contract() {
        let sdr: OutputDestination = "still.png".parse().unwrap();
        assert_eq!((sdr.peak(), sdr.white()), (100., 100.));
        let hdr: OutputDestination = r#"{"file":"still.png","display":"Rec.2100-HLG"}"#.parse().unwrap();
        assert_eq!((hdr.peak(), hdr.white()), (1000., 203.));
        for value in [
            serde_json::json!({"file":"a.png","display":null}),
            serde_json::json!({"file":"a.png","reference_white":null}),
            serde_json::json!({"file":"a.png","peak_luminance":0}),
            serde_json::json!({"file":"a.png","reference_white":101}),
            serde_json::json!({"file":"a.png","display":"Rec.2100-PQ","peak_luminance":10001}),
            serde_json::json!({"file":"a.exr","display":"sRGB","color_space":"Linear Rec.709"}),
        ] {
            assert!(serde_json::from_value::<OutputDestination>(value).map_err(anyhow::Error::from).and_then(|d| d.validate()).is_err());
        }
    }
    #[test]
    fn descriptors_are_strict_and_paths_keep_srgb() {
        let short: OutputDestination = "still.png".parse().unwrap();
        assert_eq!(short.display_target(), OutputDisplay::Srgb);
        let full: OutputDestination =
            r#"{"file":"still.png","display":"Rec.2100-PQ"}"#.parse().unwrap();
        assert_eq!(full.display_target(), OutputDisplay::Rec2100Pq);
        for bad in [
            r#"{"file":"a.png","display":"pq"}"#,
            r#"{"file":"a.png","peak_luminance":"1000"}"#,
            r#"{"file":"a.png","colorSpace":"sRGB"}"#,
        ] {
            assert!(bad.parse::<OutputDestination>().is_err());
        }
    }
}
