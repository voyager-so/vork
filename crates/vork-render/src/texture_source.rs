//! Install-time image sources. Render names resolve to slots in the material table.
use std::path::PathBuf;

/// A file's own colour description that cannot be read. 0.5.1 read none, and an explicit `color_space`
/// replaces the description, so the one fix is said once here, for every such cause, stills and footage alike.
#[derive(Debug)]
pub(crate) struct DeclaredColour { cause: String, image: Option<String> }
impl std::fmt::Display for DeclaredColour {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The decoder knows a file, not who reads it: each owner of a scene image's pictures names it (`for_image`).
        let field = self.image.as_ref().map_or_else(|| "the image's color_space".to_owned(), |name| vork_scene::v2::diagnostics::member(&vork_scene::v2::diagnostics::member("images", name), "color_space"));
        write!(f, "{}; 0.5.1 read no colour description: set {field} to \"sRGB\" (\"Linear Rec.709\" for an EXR or HDR) \
            for the 0.5.1 picture, or to the space the file is in; see vork help upgrade", self.cause)
    }
}
impl std::error::Error for DeclaredColour {}
pub(crate) fn declared_colour(cause: impl std::fmt::Display) -> anyhow::Error {
    anyhow::Error::new(DeclaredColour { cause: cause.to_string(), image: None })
}
/// Name the scene image a colour-description refusal belongs to. Its owners call this where they decode: the
/// installer of a program image, a resident media image on every frame, an environment's images. The nearest
/// owner's name stands. A hoisted inline image (`~…`) has no authored name.
pub(crate) fn for_image(mut error: anyhow::Error, name: &str) -> anyhow::Error {
    if !name.is_empty() && !name.starts_with('~') && let Some(declared) = error.downcast_mut::<DeclaredColour>()
        && declared.image.is_none() { declared.image = Some(name.to_owned()); }
    error
}
/// Where a file failed to decode. An unreadable colour description names its own fix; anything else is a damaged file.
pub(crate) fn decode_context(error: anyhow::Error, what: &str, path: &std::path::Path) -> anyhow::Error {
    let damaged = !error.is::<DeclaredColour>();
    error.context(format!("{what} {}{}", path.display(), if damaged {"; restore a valid image or correct its path"} else {""}))
}
pub(crate) fn missing_image(path: &std::path::Path) -> image::DynamicImage {
    crate::warnings::dropped("missing_image", serde_json::json!({"file":path}),
        "using a 1x1 magenta texture; restore the image or correct its path.".into());
    image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(1, 1,
        image::Rgba(vork_slang::registry::DIAGNOSTIC_MAGENTA_RGBA.to_le_bytes())))
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) enum TextureSource {
    File(PathBuf),
    Render(String),
    Media { name: String, file: PathBuf, time: Option<vork_time::Clock> },
    /// A picture of a canvas node, drawn by the 2D engine every frame.
    Canvas(String),
}
impl TextureSource {
    /// Render images are rendered again every frame; file images never change.
    pub(crate) fn changes_between_frames(&self) -> bool { matches!(self, Self::Render(_) | Self::Media {..} | Self::Canvas(_)) }
    pub(crate) fn scene(name: &str, image: &vork_scene::v2::Image) -> Self {
        if image.picture_of.is_some() { return Self::Canvas(name.into()); }
        match &image.render {
            Some(_) => Self::Render(name.into()),
            None => {
                let file = PathBuf::from(image.file.as_ref().expect("validated file image"));
                if crate::footage::is_media(&file) {Self::Media {name:name.into(),file,time:image.time.clone()}}
                else {Self::File(file)}
            },
        }
    }
}

/// Decode an image file at its native precision: 8/16-bit integer or 32-bit float
/// channels (PNG, JPEG, TIFF, Radiance HDR, OpenEXR), or IES photometry. Material,
/// program and environment images share this one decoder.
pub(crate) fn decode(
    path: &std::path::Path,
    bytes: &[u8],
    max_side: u32,
) -> anyhow::Result<image::DynamicImage> {
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ies")) {
        return crate::ies::decode(bytes);
    }
    decode_format(image::ImageFormat::from_path(path)?, bytes, max_side)
}

/// Decode colour into `destination`, the encoding the caller uploads: any of the working spaces. Explicit
/// declarations bypass embedded descriptions, including Non-Color data. Untagged identity paths retain their
/// native integer precision and exact bytes.
pub(crate) fn decode_color(
    path: &std::path::Path,
    bytes: &[u8],
    max_side: u32,
    explicit: Option<vork_scene::v2::ColorSpace>,
    destination: vork_scene::v2::WorkingColorSpace,
) -> anyhow::Result<image::DynamicImage> {
    crate::failure::contain("the colour decoder", || decode_color_impl(path,bytes,max_side,explicit,destination,false)).map(|(image,_)|image)
}
/// `decode_color`, with the colour type the file stores: its own precision, whatever conversion makes of it.
pub(crate) fn decode_color_stored(
    path: &std::path::Path,
    bytes: &[u8],
    max_side: u32,
    explicit: Option<vork_scene::v2::ColorSpace>,
    destination: vork_scene::v2::WorkingColorSpace,
) -> anyhow::Result<(image::DynamicImage, image::ColorType)> {
    crate::failure::contain("the colour decoder", || decode_color_impl(path,bytes,max_side,explicit,destination,false))
}
/// `decode_color` for an image that is light (an environment), and light is not negative. The file's own
/// negative samples refuse, judged on the stored samples before any conversion. A colour that conversion then
/// carries outside the upload's gamut (a wide-gamut image read into a narrower space has negative
/// coordinates) is limited to that gamut: its negative coordinates become zero. Data (Non-Color) is not colour.
pub(crate) fn decode_light(
    path: &std::path::Path,
    bytes: &[u8],
    max_side: u32,
    explicit: Option<vork_scene::v2::ColorSpace>,
    destination: vork_scene::v2::WorkingColorSpace,
) -> anyhow::Result<image::DynamicImage> {
    crate::failure::contain("the colour decoder", || decode_color_impl(path,bytes,max_side,explicit,destination,true)).map(|(image,_)|image)
}
/// Whether a float image has a colour sample that is not light: negative, or not finite.
fn not_light(image: &image::DynamicImage) -> bool {
    let bad = |v: &f32| !(*v >= 0. && v.is_finite());
    match image {
        image::DynamicImage::ImageRgb32F(pixels) => pixels.pixels().any(|p| p.0.iter().any(bad)),
        image::DynamicImage::ImageRgba32F(pixels) => pixels.pixels().any(|p| p.0[..3].iter().any(bad)),
        _ => false,
    }
}
/// Negative colour coordinates limited to zero (`decode_light`). Only those: a value that is not finite stays
/// as it is, for the consumer to refuse.
pub(crate) fn limit_to_gamut(image: image::DynamicImage) -> image::DynamicImage {
    let negative = |v: &f32| *v < 0.;
    let any = match &image {
        image::DynamicImage::ImageRgb32F(pixels) => pixels.pixels().any(|p| p.0.iter().any(negative)),
        image::DynamicImage::ImageRgba32F(pixels) => pixels.pixels().any(|p| p.0[..3].iter().any(negative)),
        _ => false,
    };
    if !any {return image;}
    let mut pixels = image.into_rgba32f();
    for pixel in pixels.pixels_mut() {for value in &mut pixel.0[..3] {if *value < 0. {*value = 0.;}}}
    image::DynamicImage::ImageRgba32F(pixels)
}
/// What a file says about its own colour, as its format holds it.
#[derive(Default)]
struct Description {
    /// A named space: PNG cICP, or an EXR's `vork:color_space`.
    named: Option<vork_scene::v2::WorkingColorSpace>,
    /// An embedded ICC profile.
    icc: Option<Vec<u8>>,
    /// The PNG sRGB chunk.
    srgb: bool,
    /// CIE xy of red, green, blue and white: PNG cHRM, EXR chromaticities.
    chromaticities: Option<[[f32; 2]; 4]>,
    /// The file gamma: PNG gAMA; one for EXR.
    gamma: Option<f32>,
}
/// One preparation of an image for the colour decoder: its bounds, its own description (read only when
/// `describe`: a declaration makes it irrelevant) and its stored pixels. One image decoder supplies the size,
/// the ICC profile and the pixels. The image crate's decoders do not expose a PNG's colour chunks or an EXR's
/// attributes, so those come from one read of the format's header, which also bounds an EXR's data windows;
/// it stands where the dimensions-only decoder of `bounded` stands for other callers.
fn prepare(path: &std::path::Path, bytes: &[u8], max_side: u32, describe: bool) -> anyhow::Result<(image::DynamicImage, Description)> {
    use image::ImageDecoder;
    use vork_scene::v2::WorkingColorSpace as S;
    let mut description = Description::default();
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("ies")) {return Ok((crate::ies::decode(bytes)?, description));}
    let format = image::ImageFormat::from_path(path)?;
    if format == image::ImageFormat::OpenExr {
        // The EXR decoder allocates each part's data window, which a header may place far outside the image
        // it displays.
        let metadata = exr::meta::MetaData::read_from_buffered(bytes, false)?;
        for header in &metadata.headers {
            side(max_side, "data window", header.layer_size.width(), header.layer_size.height())?;
        }
        if describe {
            // The part the pixels come from: the image decoder reads the first non-deep part with R, G and B.
            let part = metadata.headers.iter().find(|header| !header.deep && ["R","G","B"].iter()
                .all(|&channel|header.channels.find_index_of_channel(&exr::meta::attribute::Text::from(channel)).is_some()));
            description.chromaticities = part.and_then(|h|h.shared_attributes.chromaticities)
                .map(|c|[c.red,c.green,c.blue,c.white].map(|xy|[xy.0,xy.1]));
            description.gamma = Some(1.);
            if let Some(exr::meta::attribute::AttributeValue::Text(space)) = part
                .and_then(|h|h.own_attributes.other.get(&exr::meta::attribute::Text::from("vork:color_space"))) {
                description.named = Some(serde_json::from_value(serde_json::Value::String(space.to_string()))
                    .map_err(|_|declared_colour(format!("the EXR names its colour encoding {space}, which is not read")))?);
            }
        }
    }
    if describe && format == image::ImageFormat::Png {
        let reader = png::Decoder::new(std::io::Cursor::new(bytes)).read_info()?;
        let info = reader.info();
        description.srgb = info.srgb.is_some();
        description.gamma = info.gama_chunk.map(|g| g.into_value());
        description.chromaticities = info.chrm_chunk.map(|c| [c.red,c.green,c.blue,c.white].map(|(x,y)|[x.into_value(),y.into_value()]));
        // PNG Third Edition: cICP takes precedence over ICC and legacy chunks.
        if let Some(cicp) = info.coding_independent_code_points {
            if cicp.matrix_coefficients != 0 || !cicp.is_video_full_range_image {
                return Err(declared_colour("the image's cICP tag is not full-range RGB, which is not read"));
            }
            description.named = Some(match (cicp.color_primaries, cicp.transfer_function) {
                (1,13) => S::Srgb,
                (12,13) => S::DisplayP3,
                (1,8) => S::LinearRec709,
                (9,8) => S::LinearRec2020,
                (_,16|18) => return Err(declared_colour("the image is tagged PQ or HLG, which is not decoded yet")),
                _ => return Err(declared_colour(format!("the image's cICP tag (primaries {}, transfer {}) is not read yet",cicp.color_primaries,cicp.transfer_function))),
            });
        }
    }
    let mut reader = image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    // As `bounded`: sides are checked on the decoder's header below, before any pixel is decoded.
    reader.no_limits();
    let mut decoder = reader.into_decoder()?;
    let (width, height) = decoder.dimensions();
    side(max_side, "image", width as usize, height as usize)?;
    if describe && description.named.is_none() {description.icc = decoder.icc_profile()?;}
    Ok((image::DynamicImage::from_decoder(decoder)?, description))
}
fn decode_color_impl(path:&std::path::Path,bytes:&[u8],max_side:u32,explicit:Option<vork_scene::v2::ColorSpace>,destination:vork_scene::v2::WorkingColorSpace,light:bool) -> anyhow::Result<(image::DynamicImage, image::ColorType)> {
    let (decoded, description) = prepare(path, bytes, max_side, explicit.is_none())?;
    let stored = decoded.color();
    // Light is finite and not negative. The file's own samples are judged here, as stored, before a conversion
    // can hide them; what conversion then carries outside the gamut is limited to it.
    let light = light && explicit != Some(vork_scene::v2::ColorSpace::NonColor);
    anyhow::ensure!(!(light && not_light(&decoded)), "the image has negative or non-finite colour samples; environment radiance must be finite and nonnegative");
    // Without a declaration the pixels are decoded by now: what fails here is the reading of the file's description.
    let image = convert_described(decoded, description, path, explicit, destination)
        .map_err(|error| if explicit.is_some() || error.is::<DeclaredColour>() {error} else {declared_colour(format!("the image's colour description is not usable ({error:#})"))})?;
    Ok((if light {limit_to_gamut(image)} else {image}, stored))
}

/// Whether an ICC profile describes sRGB itself, read from what the profile holds: a matrix profile (no
/// lookup tables, which a transform would use instead) whose colourants are sRGB's and whose curves are the
/// sRGB curve; a grey profile needs the curve alone. Each is compared in the precision a profile stores it:
/// colourants within 0.002 in XYZ. That is the equivalence policy: makers of sRGB profiles adapt the same
/// primaries to D50 with differently rounded white points and matrices, and their profiles (LittleCMS's
/// built-in one among them, measured on an L4 as identical pictures) sit within it, while the nearest other
/// space in use, Display P3, is 0.08 away;
/// a parametric curve by its parameters, a sampled curve at every one of its samples within two 16-bit steps.
fn icc_is_srgb(profile: &moxcms::ColorProfile) -> bool {
    use moxcms::{DataColorSpace, ToneReprCurve};
    let srgb = |curve: &Option<ToneReprCurve>| match curve {
        Some(ToneReprCurve::Parametric(p)) => matches!(p.len(), 5 | 7)
            && p[..5].iter().zip([2.4, 1. / 1.055, 0.055 / 1.055, 1. / 12.92, 0.04045]).all(|(a, b): (&f32, f32)| (a - b).abs() < 1e-4)
            && p[5..].iter().all(|v| v.abs() < 1e-6),
        Some(ToneReprCurve::Lut(samples)) => samples.len() >= 256 && samples.iter().enumerate()
            .all(|(i, sample)| (f32::from(*sample) - decode_srgb(i as f32 / (samples.len() - 1) as f32) * 65535.).abs() <= 2.),
        None => false,
    };
    if profile.lut_a_to_b_perceptual.is_some() || profile.lut_a_to_b_colorimetric.is_some() || profile.lut_a_to_b_saturation.is_some() {return false;}
    // A profile's own cICP tag has authority over its curves in the transform: only an sRGB one agrees with them.
    if profile.cicp.as_ref().is_some_and(|cicp| cicp.transfer_characteristics != moxcms::TransferCharacteristics::Srgb
        || cicp.color_primaries != moxcms::CicpColorPrimaries::Bt709) {return false;}
    if profile.color_space == DataColorSpace::Gray {return srgb(&profile.gray_trc);}
    let reference = moxcms::ColorProfile::new_srgb();
    let same = |a: moxcms::Xyzd, b: moxcms::Xyzd| (a.x - b.x).abs() < 2e-3 && (a.y - b.y).abs() < 2e-3 && (a.z - b.z).abs() < 2e-3;
    profile.color_space == DataColorSpace::Rgb
        && same(profile.red_colorant, reference.red_colorant) && same(profile.green_colorant, reference.green_colorant) && same(profile.blue_colorant, reference.blue_colorant)
        && srgb(&profile.red_trc) && srgb(&profile.green_trc) && srgb(&profile.blue_trc)
}
/// The stored image converted into the upload encoding by what describes it: the declaration, else the file.
fn convert_described(decoded:image::DynamicImage,description:Description,path:&std::path::Path,explicit:Option<vork_scene::v2::ColorSpace>,destination:vork_scene::v2::WorkingColorSpace) -> anyhow::Result<image::DynamicImage> {
    use vork_scene::v2::WorkingColorSpace as S;
    let extension = |name: &str| path.extension().is_some_and(|e|e.eq_ignore_ascii_case(name));
    // EXR pixels are stored associated, and stay so.
    let associated = extension("exr");
    let convert = |image,source,destination|convert_associated_image(image,source,destination,associated);
    if let Some(explicit) = explicit {
        use vork_scene::v2::ColorSpace as C;
        return match explicit {
            C::NonColor => Ok(decoded),
            C::Srgb => Ok(convert(decoded,S::Srgb,destination)),
            C::LinearRec709 => Ok(convert(decoded,S::LinearRec709,destination)),
            C::LinearRec2020 => Ok(convert(decoded,S::LinearRec2020,destination)),
            C::ACEScg => Ok(convert(decoded,S::ACEScg,destination)),
            C::DisplayP3 => Ok(convert(decoded,S::DisplayP3,destination)),
        };
    }
    if extension("ies") {return Ok(decoded);}
    let format = image::ImageFormat::from_path(path)?;
    let Description {named: source, icc: profile, srgb: tagged_srgb, chromaticities, gamma} = description;
    if let Some(source) = source {return Ok(convert(decoded,source,destination));}
    if let Some(icc) = profile {
        use moxcms::{ColorProfile, DataColorSpace, Layout, TransformOptions};
        let profile = ColorProfile::new_from_slice(&icc)?;
        if !matches!(profile.color_space,DataColorSpace::Rgb|DataColorSpace::Gray) {
            return Err(declared_colour("the image's ICC profile is not RGB or grey (CMYK, for one), which is not read"));
        }
        if profile.cicp.as_ref().is_some_and(|cicp|matches!(cicp.transfer_characteristics,moxcms::TransferCharacteristics::Smpte2084|moxcms::TransferCharacteristics::Hlg)) {
            return Err(declared_colour("the image is tagged PQ or HLG, which is not decoded yet"));
        }
        let gray = profile.color_space == DataColorSpace::Gray;
        // A profile that describes sRGB itself changes nothing: the image is an sRGB image, and keeps its samples
        // and their precision, as one with an sRGB chunk does.
        if icc_is_srgb(&profile) {return Ok(convert(decoded,S::Srgb,destination));}
        let transform = profile.create_transform_f32(if gray {Layout::GrayAlpha} else {Layout::Rgba}, &ColorProfile::new_srgb(), Layout::Rgba, TransformOptions {allow_extended_range_rgb_xyz:true,prefer_fixed_point:false,..Default::default()})?;
        let input = decoded.to_rgba32f();
        let mut output = input.clone();
        if gray {
            let channels:Vec<f32> = input.pixels().flat_map(|p|[p[0],p[3]]).collect();
            transform.transform(&channels,output.as_mut())?;
        } else {transform.transform(input.as_raw(), output.as_mut())?;}
        // The transform's result is sRGB code values.
        let to_destination = S::Srgb.conversion(destination);
        for (original,pixel) in input.pixels().zip(output.pixels_mut()) {
            pixel[3] = original[3];
            if !to_destination.is_identity() {
                let rgb = to_destination.apply([pixel[0],pixel[1],pixel[2]]);
                pixel.0[..3].copy_from_slice(&rgb);
            }
        }
        return Ok(image::DynamicImage::ImageRgba32F(output));
    }
    if tagged_srgb {return Ok(convert(decoded,S::Srgb,destination));}
    if chromaticities.is_some() || gamma.is_some() {
        let primaries = chromaticities.unwrap_or(S::LinearRec709.chromaticities());
        anyhow::ensure!(gamma.is_none_or(|g|g.is_finite() && g > 0.), "image gamma must be positive");
        // The description resolved: Rec.709 primaries with no gamma are sRGB, with gamma one linear Rec.709.
        // Those are named spaces, converted (or left as they are) like any other.
        let identity = primaries.iter().flatten().zip(S::LinearRec709.chromaticities().iter().flatten()).all(|(a,b)|(a-b).abs()<1e-5);
        if identity && gamma.is_none() {return Ok(convert(decoded,S::Srgb,destination));}
        if identity && gamma == Some(1.) {return Ok(convert(decoded,S::LinearRec709,destination));}
        let matrix = chromaticities_to_rec709(primaries)?;
        let encode = S::LinearRec709.conversion(destination);
        let mut pixels = decoded.to_rgba32f();
        for pixel in pixels.pixels_mut() {
            let alpha = if associated && (gamma != Some(1.) || !destination.is_linear()) {pixel[3]} else {1.};
            let rgb = [pixel[0],pixel[1],pixel[2]].map(|v|if alpha>0. {v/alpha} else {0.})
                .map(|v|gamma.map_or_else(||decode_srgb(v),|g|v.signum()*v.abs().powf(1./g)));
            let rgb = glam::Vec3::from_array(rgb);
            let rgb = (matrix*rgb).to_array();
            let rgb = encode.apply(rgb).map(|v|v*alpha);
            pixel.0[..3].copy_from_slice(&rgb);
        }
        return Ok(image::DynamicImage::ImageRgba32F(pixels));
    }
    let fallback = if matches!(format,image::ImageFormat::OpenExr|image::ImageFormat::Hdr) {S::LinearRec709} else {S::Srgb};
    Ok(convert(decoded,fallback,destination))
}

/// Convert an image's colour between two spaces. `associated`: its colour is multiplied by alpha (EXR); a
/// change of primaries alone commutes with that, a transfer curve does not, so the colour is unassociated
/// around it. The same space leaves the image as it is, in its own precision.
fn convert_associated_image(decoded:image::DynamicImage,source:vork_scene::v2::WorkingColorSpace,destination:vork_scene::v2::WorkingColorSpace,associated:bool)->image::DynamicImage {
    let conversion=source.conversion(destination);
    if conversion.is_identity() {return decoded;}
    let around_alpha=associated && !(source.is_linear() && destination.is_linear());
    let mut pixels=decoded.to_rgba32f();
    for pixel in pixels.pixels_mut() {
        let alpha=if around_alpha {pixel[3]} else {1.};
        let rgb=[pixel[0],pixel[1],pixel[2]].map(|v|if alpha>0. {v/alpha} else {0.});
        pixel.0[..3].copy_from_slice(&conversion.apply(rgb).map(|v|v*alpha));
    }
    image::DynamicImage::ImageRgba32F(pixels)
}

/// Convert an image of straight colour between two spaces.
pub(crate) fn convert_image(decoded: image::DynamicImage, source: vork_scene::v2::WorkingColorSpace, destination: vork_scene::v2::WorkingColorSpace) -> image::DynamicImage {
    convert_associated_image(decoded,source,destination,false)
}

/// CIE xy primaries to linear Rec.709, with Bradford white adaptation.
fn chromaticities_to_rec709(xy: [[f32;2];4]) -> anyhow::Result<glam::Mat3> {
    use glam::{Mat3,Vec3};
    fn basis(xy: [[f32;2];4]) -> anyhow::Result<(Mat3,Vec3)> {
        anyhow::ensure!(xy.iter().flatten().all(|v|v.is_finite()) && xy.iter().all(|v|v[1] != 0.), "invalid image chromaticities");
        let xyz = xy.map(|[x,y]|Vec3::new(x/y,1.,(1.-x-y)/y));
        let unscaled = Mat3::from_cols(xyz[0],xyz[1],xyz[2]);
        anyhow::ensure!(unscaled.determinant().abs()>1e-10,"singular image chromaticities");
        let scale = unscaled.inverse()*xyz[3];
        Ok((unscaled*Mat3::from_diagonal(scale),xyz[3]))
    }
    let (source,white) = basis(xy)?;
    let (destination,d65) = basis(vork_scene::v2::WorkingColorSpace::LinearRec709.chromaticities())?;
    let bradford = Mat3::from_cols_array(&[0.8951,-0.7502,0.0389,0.2664,1.7135,-0.0685,-0.1614,0.0367,1.0296]);
    let cone = bradford*white;
    anyhow::ensure!(cone.abs().min_element()>1e-10,"invalid image white point");
    let adaptation = bradford.inverse()*Mat3::from_diagonal((bradford*d65)/cone)*bradford;
    let matrix = destination.inverse()*adaptation*source;
    anyhow::ensure!(matrix.is_finite(),"invalid image colour conversion");
    Ok(matrix)
}

/// The raster image decoder for scene and glTF images. Each image uploads as one
/// texture, so a side beyond the device's `max_side` is refused from the header,
/// before any pixel is decoded. Upstream decoders panic on some malformed files;
/// that is returned as an error.
pub(crate) fn decode_format(
    format: image::ImageFormat,
    bytes: &[u8],
    max_side: u32,
) -> anyhow::Result<image::DynamicImage> {
    crate::failure::contain("the image decoder", || {
        let reader = bounded(format, bytes, max_side)?;
        Ok(reader.decode()?)
    })
}

/// Size and channels from the header alone (glTF images whose pixels are cached),
/// with the same format and bounds as `decode_format`.
pub(crate) fn header(
    format: image::ImageFormat,
    bytes: &[u8],
    max_side: u32,
) -> anyhow::Result<(u32, u32, image::ColorType)> {
    use image::ImageDecoder;
    crate::failure::contain("the image decoder", || {
        let decoder = bounded(format, bytes, max_side)?.into_decoder()?;
        let (width, height) = decoder.dimensions();
        Ok((width, height, decoder.color_type()))
    })
}

/// A side of an image, or of an EXR part's data window, within `max_side`.
fn side(max_side: u32, what: &str, width: usize, height: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        width.max(height) <= max_side as usize,
        crate::gpu_memory::device_limit(format_args!("the {what} is {width}x{height}"), format_args!("{width}x{height} texels"), format_args!("{max_side} texels per texture side"), "reduce the source image resolution")
    );
    Ok(())
}

/// A reader for an image whose sides, and for EXR each part's data window, fit
/// `max_side`.
fn bounded(
    format: image::ImageFormat,
    bytes: &[u8],
    max_side: u32,
) -> anyhow::Result<image::ImageReader<std::io::Cursor<&[u8]>>> {
    let side = |what: &str, width: usize, height: usize| side(max_side, what, width, height);
    if format == image::ImageFormat::OpenExr {
        // The EXR decoder allocates each part's data window, which a header may
        // place far outside the image it displays.
        for header in &exr::meta::MetaData::read_from_buffered(bytes, false)?.headers {
            side("data window", header.layer_size.width(), header.layer_size.height())?;
        }
    }
    let reader = || image::ImageReader::with_format(std::io::Cursor::new(bytes), format);
    let (width, height) = reader().into_dimensions()?;
    side("image", width as usize, height as usize)?;
    let mut reader = reader();
    // Device-sized float images can exceed image's default 512 MiB decode
    // budget. Dimensions (including EXR data windows) are bounded above;
    // actual GPU allocations retain their device capacity checks.
    reader.no_limits();
    Ok(reader)
}

/// The encoding a world's image is uploaded in: sRGB code values where the upload is sRGB-encoded (the
/// texture's format or its installation decodes them), else linear Rec.709 light.
pub(crate) fn upload(srgb: bool) -> vork_scene::v2::WorkingColorSpace {
    if srgb { vork_scene::v2::WorkingColorSpace::Srgb } else { vork_scene::v2::WorkingColorSpace::LinearRec709 }
}

/// IEC 61966-2-1 decoding, shared by raw and compressed image installation.
pub(crate) fn decode_srgb(value: f32) -> f32 {
    let curve = crate::constants::SRGB_DECODE_CURVE;
    if value <= curve[0] {
        value / curve[1]
    } else {
        ((value + curve[2]) / curve[3]).powf(curve[4])
    }
}

#[cfg(test)]
mod tests {
    fn tagged_png(primaries: u8, transfer: u8) -> Vec<u8> {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes,1,1);
            encoder.set_color(png::ColorType::Rgba);
            let mut writer = encoder.write_header().unwrap();
            writer.write_chunk(png::chunk::ChunkType(*b"cICP"),&[primaries,transfer,0,1]).unwrap();
            writer.write_image_data(&[255,0,0,128]).unwrap();
        }
        bytes
    }

    #[test]
    fn tagged_p3_preserves_extended_colour_and_explicit_override_wins() {
        use vork_scene::v2::ColorSpace;
        let bytes = tagged_png(12,13);
        let path = std::path::Path::new("picture.png");
        let image = super::decode_color(path,&bytes,16,None,super::upload(true)).unwrap().to_rgba32f();
        let pixel = image.get_pixel(0,0).0;
        assert!(pixel[0]>1. && pixel[1]<0. && pixel[2]<0., "{pixel:?}");
        assert_eq!(pixel[3],128./255.);
        for space in [ColorSpace::Srgb,ColorSpace::NonColor,ColorSpace::LinearRec709] {
            let image = super::decode_color(path,&bytes,16,Some(space),super::upload(space==ColorSpace::Srgb)).unwrap();
            assert_eq!(image.as_rgba8().unwrap().get_pixel(0,0).0,[255,0,0,128]);
        }
        let bytes = tagged_png(9,16);
        assert!(super::decode_color(path,&bytes,16,None,super::upload(true)).unwrap_err().to_string().contains("is tagged PQ or HLG"));
        assert!(super::decode_color(path,&bytes,16,Some(ColorSpace::NonColor),super::upload(false)).is_ok());
    }

    #[test]
    fn icc_rgb_gray_and_cicp_precedence_preserve_alpha() {
        fn png(profile:moxcms::ColorProfile,gray:bool,cicp:bool)->Vec<u8> {
            let mut info=png::Info::with_size(1,1);
            info.color_type=if gray {png::ColorType::GrayscaleAlpha} else {png::ColorType::Rgba};
            info.icc_profile=Some(std::borrow::Cow::Owned(profile.encode().unwrap()));
            let mut bytes=Vec::new();
            {
                let mut writer=png::Encoder::with_info(&mut bytes,info).unwrap().write_header().unwrap();
                if cicp {writer.write_chunk(png::chunk::ChunkType(*b"cICP"),&[1,13,0,1]).unwrap();}
                writer.write_image_data(if gray {&[128,64]} else {&[255,0,0,128]}).unwrap();
            }
            bytes
        }
        let path=std::path::Path::new("profile.png");
        let bytes=png(moxcms::ColorProfile::new_display_p3(),false,false);
        let pixel=super::decode_color(path,&bytes,16,None,super::upload(false)).unwrap().to_rgba32f().get_pixel(0,0).0;
        for (actual,expected) in pixel.into_iter().zip([1.22494018,-0.04205693,-0.01963755,128./255.]) {
            assert!((actual-expected).abs()<0.003,"{pixel:?}");
        }
        // What a profile holds decides whether it is sRGB: the sRGB curve as parameters or as samples, and
        // sRGB's colourants. Another curve, other colourants or one off sample is another profile.
        let srgb=moxcms::ColorProfile::new_srgb();
        assert!(super::icc_is_srgb(&srgb) && !super::icc_is_srgb(&moxcms::ColorProfile::new_display_p3()));
        let sampled=|edit:fn(&mut Vec<u16>)| {
            let mut samples:Vec<u16>=(0..1024).map(|i|(super::decode_srgb(i as f32/1023.)*65535.).round() as u16).collect();
            edit(&mut samples);
            let mut profile=moxcms::ColorProfile::new_srgb();
            let curve=Some(moxcms::ToneReprCurve::Lut(samples));
            (profile.red_trc,profile.green_trc,profile.blue_trc)=(curve.clone(),curve.clone(),curve);
            profile
        };
        assert!(super::icc_is_srgb(&sampled(|_|{})));
        assert!(!super::icc_is_srgb(&sampled(|samples|samples[128]+=64)), "one sample off the sRGB curve is another curve");
        // A profile's own cICP tag overrides its curves: sRGB curves under a linear tag are not sRGB.
        let mut linear=moxcms::ColorProfile::new_srgb();
        linear.cicp.as_mut().unwrap().transfer_characteristics=moxcms::TransferCharacteristics::Linear;
        assert!(!super::icc_is_srgb(&linear));
        let mut gamma=moxcms::ColorProfile::new_srgb();
        gamma.red_trc=Some(moxcms::ToneReprCurve::Parametric(vec![2.2]));
        assert!(!super::icc_is_srgb(&gamma));
        // A profile that describes sRGB leaves the image as it is, in its own 8 bits; linear uploads decode it.
        let bytes=png(moxcms::ColorProfile::new_srgb(),false,false);
        assert_eq!(super::decode_color(path,&bytes,16,None,super::upload(true)).unwrap().as_rgba8().unwrap().get_pixel(0,0).0,[255,0,0,128]);
        let pixel=super::decode_color(path,&bytes,16,None,super::upload(false)).unwrap().to_rgba32f().get_pixel(0,0).0;
        assert_eq!(pixel,[1.,0.,0.,128./255.]);
        let bytes=png(moxcms::ColorProfile::new_display_p3(),false,true);
        let decoded=super::decode_color(path,&bytes,16,None,super::upload(true)).unwrap();
        assert_eq!(decoded.as_rgba8().unwrap().get_pixel(0,0).0,[255,0,0,128]);
        let bytes=png(moxcms::ColorProfile::new_gray_with_gamma(2.2),true,false);
        let pixel=super::decode_color(path,&bytes,16,None,super::upload(false)).unwrap().to_rgba32f().get_pixel(0,0).0;
        for value in &pixel[..3] {assert!((*value-(128f32/255.).powf(2.2)).abs()<0.001,"{pixel:?}");}
        assert_eq!(pixel[3],64./255.);
    }

    #[test]
    fn light_is_limited_to_the_gamut_and_a_files_own_negatives_stay() {
        use exr::prelude::*;
        let write=|pixel:(f32,f32,f32,f32),space:Option<&str>| {
            let mut image=Image::from_channels((1,1),SpecificChannels::rgba(move |_:Vec2<usize>|pixel));
            if let Some(space)=space {image.layer_data.attributes.other.insert(Text::from("vork:color_space"),AttributeValue::Text(Text::from(space)));}
            let mut bytes=std::io::Cursor::new(Vec::new());
            image.write().to_buffered(&mut bytes).unwrap();
            bytes.into_inner()
        };
        let path=std::path::Path::new("sky.exr");
        let pixel=|bytes:&[u8],light:bool,explicit:Option<vork_scene::v2::ColorSpace>| {
            let decoded=if light {super::decode_light(path,bytes,16,explicit,super::upload(false))} else {super::decode_color(path,bytes,16,explicit,super::upload(false))};
            decoded.unwrap().to_rgba32f().get_pixel(0,0).0
        };
        // Saturated ACEScg green is outside linear Rec.709: a material keeps the signed coordinates, light is limited.
        let wide=write((0.,1.,0.,1.),Some("ACEScg"));
        let signed=pixel(&wide,false,None);
        assert!(signed[0]<0. && signed[2]<0. && signed[1]>1.,"{signed:?}");
        assert_eq!(pixel(&wide,true,None),[0.,signed[1],0.,1.]);
        // The file's own negative samples refuse as light, judged as stored, whatever a conversion would make
        // of them. A material reads them.
        for own in [write((-0.01,0.5,0.5,1.),None),write((-0.01,0.5,0.5,1.),Some("ACEScg"))] {
            let error=super::decode_light(path,&own,16,None,super::upload(false)).unwrap_err();
            assert!(error.to_string().contains("environment radiance must be finite and nonnegative"),"{error:#}");
            assert!(pixel(&own,false,None)[1]>0.);
        }
        // Data is not colour.
        assert_eq!(pixel(&wide,true,Some(vork_scene::v2::ColorSpace::NonColor)),[0.,1.,0.,1.]);
        // A sample that is not finite refuses as light too, and the gamut limit never turns one into black.
        let error=super::decode_light(path,&write((f32::NAN,1.,0.,1.),Some("ACEScg")),16,None,super::upload(false)).unwrap_err();
        assert!(error.to_string().contains("environment radiance must be finite and nonnegative"),"{error:#}");
        let limited=super::limit_to_gamut(image::DynamicImage::ImageRgba32F(image::ImageBuffer::from_raw(2,1,vec![f32::NAN,-1.,2.,1., -0.5,0.5,f32::INFINITY,1.]).unwrap())).to_rgba32f();
        assert!(limited.get_pixel(0,0)[0].is_nan() && limited.get_pixel(0,0)[1]==0. && limited.get_pixel(1,0)[0]==0. && limited.get_pixel(1,0)[2]==f32::INFINITY);
    }

    #[test]
    fn exr_colour_stays_associated_and_a_named_encoded_space_converts_around_alpha() {
        use exr::prelude::*;
        let encoded=1.055*4f32.powf(1./2.4)-0.055;
        let write=|red:f32,space:Option<&str>| {
            let mut image=Image::from_channels((1,1),SpecificChannels::rgba(move |_:Vec2<usize>|(red,0f32,0f32,0.5f32)));
            if let Some(space)=space {image.layer_data.attributes.other.insert(Text::from("vork:color_space"),AttributeValue::Text(Text::from(space)));}
            let mut bytes=std::io::Cursor::new(Vec::new());
            image.write().to_buffered(&mut bytes).unwrap();
            bytes.into_inner()
        };
        let path=std::path::Path::new("associated.exr");
        // Untagged EXR is linear Rec.709: its associated samples upload as they are.
        let plain=super::decode_color(path,&write(2.,None),16,None,super::upload(false)).unwrap().to_rgba32f();
        assert_eq!(plain.get_pixel(0,0).0,[2.,0.,0.,0.5]);
        // An EXR that names an encoded space stores encoded colour times alpha: straight 4 in linear light.
        let tagged=super::decode_color(path,&write(encoded*0.5,Some("sRGB")),16,None,super::upload(false)).unwrap().to_rgba32f();
        assert!((tagged.get_pixel(0,0)[0]-2.).abs()<1e-5,"{:?}",tagged.get_pixel(0,0));
        assert_eq!(tagged.get_pixel(0,0)[3],0.5);
    }

    #[test]
    fn chromaticity_conversion_matches_pinned_working_spaces() {
        use vork_scene::v2::WorkingColorSpace as S;
        let rgb = [2.,-0.25,0.5];
        for space in [S::LinearRec709,S::LinearRec2020,S::ACEScg] {
            let matrix = super::chromaticities_to_rec709(space.chromaticities()).unwrap();
            let actual = (matrix*glam::Vec3::from_array(rgb)).to_array();
            let expected = space.convert_rgb(S::LinearRec709,rgb);
            for (a,e) in actual.into_iter().zip(expected) {assert!((a-e).abs()<2e-5,"{space:?}: {actual:?} vs {expected:?}");}
        }
    }

    fn half_exr(display: (usize, usize), data: (usize, usize)) -> Vec<u8> {
        use exr::prelude::*;
        // A cropped half-float pixel keeps the fixture tiny even when its
        // display window requires a large decoded float32 image.
        let mut image = Image::from_channels(data, SpecificChannels::rgb(|_: Vec2<usize>| {
            (f16::from_f32(0.125), f16::from_f32(0.5), f16::from_f32(2.))
        }));
        image.attributes.display_window = IntegerBounds::from_dimensions(display);
        let mut bytes = std::io::Cursor::new(Vec::new());
        image.write().to_buffered(&mut bytes).unwrap();
        bytes.into_inner()
    }

    #[test]
    fn device_sized_half_exr_can_exceed_default_decoder_budget() {
        // RGB32F at this size is just over image's default 512 MiB budget.
        let bytes = half_exr((16384, 2731), (1, 1));
        let image = super::decode(std::path::Path::new("sky.exr"), &bytes, 16384).unwrap();
        let pixels = image.as_rgb32f().unwrap();
        assert_eq!(pixels.dimensions(), (16384, 2731));
        assert_eq!(pixels.get_pixel(0, 0).0, [0.125, 0.5, 2.]);
        assert_eq!(pixels.get_pixel(16383, 2730).0, [0.; 3]);
    }

    #[test]
    fn exr_windows_still_obey_device_texture_limit() {
        let bytes = half_exr((16384, 8192), (1, 1));
        let error = super::decode(std::path::Path::new("sky.exr"), &bytes, 8192).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("the image is 16384x8192"), "{message}");
        assert!(message.contains("8192 texels per texture side"), "{message}");
        assert!(message.contains("reduce the source image resolution"), "{message}");
        assert!(message.contains("vork help limits"), "{message}");

        // A data window outside a device-sized display must also be refused.
        let bytes = half_exr((1, 1), (2, 1));
        let error = super::decode(std::path::Path::new("sky.exr"), &bytes, 1).unwrap_err();
        assert!(format!("{error:#}").contains("the data window is 2x1"));
    }

    /// QA fuzz reproducers in crates/vork-fuzz/seeds/image (first byte: the fuzz
    /// harness's format selector). Each aborted, panicked, took gigabytes or ran for
    /// tens of seconds; each is now refused with its cause or returns its panic.
    #[test]
    #[cfg(vork_private_tests)]
    fn malformed_images_are_refused_before_decoding_or_contained() {
        let seeds = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-fuzz/seeds/image");
        let mut cases = vec![
            ("exr-dwa-ac-count-abort", "exr", "the image is 7x65537"),
            ("exr-dwa-capacity-overflow", "exr", "the image is 7x65537"),
            ("exr-dwaa-truncated-slow", "exr", "the data window is 83886082x66"),
            ("exr-data-window-5gb", "exr", "the data window is 352321538x2"),
            ("exr-dwa-ac-count-small-window", "exr", "DWA section sizes exceed the chunk"),
            // zune-jpeg 0.5.15 asserts in its AVX2 upsampler on this one.
            ("jpeg-avx2-upsampler-assert", "jpg", "the image is 4x16388"),
        ];
        // zune-jpeg 0.5.15 unwraps None in its AVX2 IDCT, the path it takes on
        // x86-64 hosts with AVX2 (every CI and VM host).
        if cfg!(target_arch = "x86_64") {
            cases.push(("jpeg-idct-unwrap", "jpg", "the image decoder failed on malformed input (upstream panic at zune-jpeg-0.5.15/src/idct/avx2.rs:"));
        }
        for (seed, extension, cause) in cases {
            let bytes = std::fs::read(seeds.join(seed)).unwrap();
            let path = std::path::PathBuf::from(format!("image.{extension}"));
            let error = crate::fuzz::image(&path, &bytes[1..]).expect_err(seed);
            assert!(format!("{error:#}").contains(cause), "{seed}: {error:#}");
        }
    }
}
