//! Render-call destinations: the display their viewed pictures are encoded for.
use anyhow::{Context, Result, ensure};
use vork_scene::v2::{self, OutputDestination, OutputDisplay};

/// The display one render call encodes for. The view names it in `ViewContext`, and the display stage, the
/// only place a picture is encoded, converts the view's result to it for every viewed destination: stills,
/// frames and GPU colour alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub display: OutputDisplay,
    /// The display's peak luminance in cd/m².
    pub peak: f32,
    /// The luminance of unit white in cd/m².
    pub white: f32,
}
impl Default for Target {
    fn default() -> Self { Self::for_display(OutputDisplay::Srgb) }
}
impl Target {
    /// `display` at its default luminances.
    pub fn for_display(display: OutputDisplay) -> Self {
        Self::of(&OutputDestination { display: Some(display), ..std::path::PathBuf::new().into() })
    }
    fn of(destination: &OutputDestination) -> Self {
        Self { display: destination.display_target(), peak: destination.peak(), white: destination.white() }
    }
    /// The primaries a video stream of this display's pictures names.
    pub(crate) fn video_primaries(self) -> vork_encode::Primaries {
        match self.display {
            OutputDisplay::DisplayP3 => vork_encode::Primaries::DisplayP3,
            // `target` refuses Rec.2100 video; sRGB pictures have BT.709 primaries.
            OutputDisplay::Srgb | OutputDisplay::Rec2100Pq | OutputDisplay::Rec2100Hlg => vork_encode::Primaries::Bt709,
        }
    }
    /// HDR code values need 16 bits; an SDR picture takes the depth asked for.
    pub(crate) fn png_depth(self, asked: crate::PngDepth) -> crate::PngDepth {
        if self.display.is_hdr() { crate::PngDepth::Sixteen } else { asked }
    }
}
/// The BT.2100 HLG system gamma for a display peak: Table 5, with note 5f's extended formula outside the
/// 400 to 2000 cd/m² range.
pub(crate) fn hlg_gamma(peak: f32) -> f32 {
    let relative = f64::from(peak) / 1000.;
    (if (400. ..=2000.).contains(&peak) { 1.2 + 0.42 * relative.log10() } else { 1.2 * 1.111f64.powf(relative.log2()) }) as f32
}
/// OpenColorIO bounds the HLG surround gain at a display luminance of 1e-4 in its E_MAX=3 intermediate; this
/// is that floor as a fraction of the peak.
pub(crate) fn hlg_floor(peak: f32) -> f32 { 1e-4 / 3f32.powf(hlg_gamma(peak)) }

/// What the destinations ask for, and the one display they share. All refusals precede device creation and
/// destination invalidation.
pub(crate) fn target(destinations: &[OutputDestination]) -> Result<Target> {
    let check = || -> Result<Target> {
        let mut shared = None;
        for destination in destinations {
            destination.validate()?;
            let raw = !crate::shot::directory_marked(&destination.file) && destination.file.extension().is_some_and(|e| e.eq_ignore_ascii_case("exr"));
            ensure!(!raw || (destination.display.is_none() && destination.peak_luminance.is_none() && destination.reference_white.is_none()),
                "EXR output is not encoded for a display; omit output.display and its luminance fields (output.color_space names its storage)");
            ensure!(raw || destination.color_space.is_none(), "output.color_space is the storage space of raw EXR output; a viewed output takes output.display");
            if raw {
                continue;
            }
            let asked = Target::of(destination);
            let display = asked.display;
            // An SDR display shows unit white at code value one; its luminance is the viewer's. The fields are
            // refused there whatever they say, 100 included: what is authored decides, not what it resolves to.
            ensure!(display.is_hdr() || (destination.peak_luminance.is_none() && destination.reference_white.is_none()),
                "output.peak_luminance and output.reference_white describe a Rec.2100 display; omit them for {}", display.name());
            // Below the reference surround's floor the HLG inverse OOTF is bounded and unit white no longer lands on the reference white.
            ensure!(display != OutputDisplay::Rec2100Hlg || asked.white >= asked.peak * hlg_floor(asked.peak),
                "output.reference_white is below the Rec.2100-HLG reference surround's floor for this peak_luminance; raise it");
            // Video frames are 8-bit: an SDR display's code values, tagged with its primaries.
            ensure!(!display.is_hdr() || !crate::shot::video_destination(&destination.file),
                "output.display {} for video is not built yet: video here is 8-bit; write PNG frames to a directory", display.name());
            // The display stage runs once per frame, for one display.
            let first: Target = *shared.get_or_insert(asked);
            ensure!(first.display == display, "one render call encodes for one display, and these destinations ask for {} and {}; render each display in its own call",
                first.display.name(), display.name());
            ensure!(first == asked, "one render call encodes for one display, and these {} destinations ask for different luminances; render each in its own call",
                display.name());
        }
        Ok(shared.unwrap_or_default())
    };
    check().context(crate::failure::FailureKind::Usage)
}
/// The storage of one EXR destination: its picture is converted to the space the destination asks for, which
/// the file then names, with the reference domain of the picture. A world's EXR is its raw picture,
/// scene-referred light in its working space, that bypasses the view. A canvas's EXR is its picture after its
/// view as a reader takes it, display-referred: sRGB code values, whatever encoding the canvas blends in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RawColor {
    /// The space the file's colour is stored in: `output.color_space`, or the working space.
    pub storage: v2::WorkingColorSpace,
    /// The picture's colour (a world's working space; a canvas's sRGB code values after its view) to storage
    /// colour; the identity stores the values as they are.
    conversion: v2::ColorConversion,
    /// Whether the picture is display-referred (a canvas's) or scene-referred (a world's).
    display_referred: bool,
}
impl Default for RawColor {
    /// A world's working space, stored as it is.
    fn default() -> Self {
        let storage = v2::WorkingColorSpace::LinearRec709;
        Self { storage, conversion: storage.conversion(storage), display_referred: false }
    }
}
impl RawColor {
    pub(crate) fn of(scene: &v2::Scene, destination: &OutputDestination) -> Self {
        let working = scene.composition.as_ref().map_or(v2::WorkingColorSpace::LinearRec709, v2::Composition::working_space);
        let storage = destination.color_space.unwrap_or(working);
        // A world's raw picture is its working space's light. A canvas's picture is the one a reader takes:
        // through its view, as sRGB code values, whatever encoding the canvas blends in
        // (`canvas::Pictures::reader_stage`).
        let picture = if scene.is_canvas() { v2::WorkingColorSpace::Srgb } else { working };
        Self { storage, conversion: picture.conversion(storage), display_referred: scene.is_canvas() }
    }
    /// The file's `vork:reference_domain`.
    pub(crate) fn reference_domain(self) -> &'static str {
        if self.display_referred { "display" } else { "scene" }
    }
    /// One unassociated colour in the storage space.
    pub(crate) fn store(self, rgb: [f32; 3]) -> [f32; 3] { self.conversion.apply(rgb) }
    /// One associated pixel in the storage space. Between linear spaces the conversion is a matrix: it commutes
    /// with association, and light without coverage (a bloom halo on transparent film) keeps its energy. A
    /// conversion through a transfer curve (a canvas's sRGB code values to another space) takes the pixel's
    /// unassociated colour and associates the result again; a pixel without coverage has no colour to
    /// unassociate, and its values are converted as they are.
    pub(crate) fn store_associated(self, [r, g, b, a]: [f32; 4]) -> [f32; 4] {
        let [r, g, b] = if self.conversion.is_linear() || a <= 0. { self.store([r, g, b]) } else { self.store([r / a, g / a, b / a]).map(|v| v * a) };
        [r, g, b, a]
    }
}

/// What the scene can encode for the target and store in its raw destinations. A usage error, raised when
/// the scene is admitted, before device work.
pub(crate) fn admit(scene: &v2::Scene, destinations: &[OutputDestination], target: Target) -> Result<()> {
    let check = || -> Result<()> {
        // A world's raw picture is scene-linear light, and its file says so: it is stored in a linear space.
        // A canvas's EXR is display-referred and takes any of the spaces: sRGB's code values as they are, Display
        // P3's, or display-linear light in a linear space.
        for space in destinations.iter().filter_map(|destination| destination.color_space) {
            ensure!(scene.is_canvas() || space.is_linear(),
                "output.color_space {} is not linear; a world's EXR takes Linear Rec.709, Linear Rec.2020 or ACEScg", space.name());
        }
        // A canvas's frame is 8-bit; its view stage encodes its picture for an SDR display.
        ensure!(!scene.is_canvas() || !target.display.is_hdr(),
            "output.display {} on a 2D canvas is not built yet: a canvas is 8-bit; use sRGB or Display P3", target.display.name());
        // A bundled view with its own bake for the display has that display's rendering only as it was baked.
        // Its other looks, or an authored table in place of the library's, are sRGB renderings: converting
        // one would hand over a gamut-limited picture labelled as the wider display, so they refuse.
        let bakes: Vec<_> = vork_scene::libraries::LIBRARIES.iter().flat_map(|library| library.images)
            .filter(|data| data.displays.contains(&target.display)).collect();
        if bakes.is_empty() {
            return Ok(());
        }
        let (name, view) = scene.view_material()?;
        // Whether a track on the named view's `input` can set it to anything but `token`. Keys held in
        // buffers are not read here and count as doing so.
        let keyed_away = |input: &str, token: &str| name.as_ref().is_some_and(|name| scene.animation.iter().flat_map(|layer| layer.tracks.iter().flatten())
            .filter(|track| track.path.iter().map(String::as_str).eq(["materials", name.as_str(), "inputs", input]))
            .any(|track| match &track.keys {
                vork_scene::v2::Keys::Inline(keys) => keys.iter().any(|key| key.get(1).and_then(|value| value.as_str()) != Some(token)),
                vork_scene::v2::Keys::Buffers { .. } => true,
            }));
        // An author's own table in the bake's input replaces the library's data, not the rule: the program
        // still reads it only under that look.
        let bakes: Vec<_> = bakes.into_iter().filter(|data| data.program == view.program).collect();
        for data in &bakes {
            let authored = data.instead_of.filter(|input| view.inputs.contains_key(*input));
            let other = data.read_when.filter(|(input, token)| keyed_away(input, token)
                || matches!(view.inputs.get(*input), Some(vork_scene::v2::Input::Token(value)) if value != token));
            if authored.is_none() && other.is_none() {
                continue;
            }
            let mut only = Vec::new();
            if let Some((input, token)) = data.read_when { only.push(format!("{input} {token}")); }
            if let Some(input) = data.instead_of { only.push(format!("the library's {input}")); }
            anyhow::bail!("{} renders {} only with {}; use that, or output.display sRGB", view.program, target.display.name(), only.join(" and "));
        }
        // The bakes render the luminances they were baked for. An author's own table in the bake's input is
        // read by the same program, which converts it for those luminances, so it is held to them too.
        if bakes.is_empty() || bakes.iter().any(|data| data.baked_for(target.peak, target.white)) {
            return Ok(());
        }
        let baked: Vec<_> = bakes.iter().flat_map(|data| data.luminances).map(|[peak, white]| format!("{peak}/{white}")).collect();
        anyhow::bail!("{} renders {} at output.peak_luminance/output.reference_white {}; other luminances are not built yet",
            view.program, target.display.name(), baked.join(" or "))
    };
    check().context(crate::failure::FailureKind::Usage)
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Admission with no raw storage asked for.
    fn admit(scene: &v2::Scene, target: Target) -> Result<()> { super::admit(scene, &[], target) }
    #[test]
    fn raw_storage_is_for_exr_and_a_world_stores_linear_light() {
        use v2::WorkingColorSpace as S;
        let raw = |space| OutputDestination { color_space: Some(space), .."frames.%06d.exr".into() };
        let world: v2::Scene = serde_json::from_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}]})).unwrap();
        // Raw storage beside a viewed destination leaves the display alone.
        assert_eq!(target(&[raw(S::ACEScg), "still.png".into()]).unwrap(), Target::default());
        let error = target(&[OutputDestination { color_space: Some(S::ACEScg), .."still.png".into() }]).unwrap_err();
        assert_eq!(error.root_cause().to_string(), "output.color_space is the storage space of raw EXR output; a viewed output takes output.display");
        assert_eq!(crate::failure::exit_code(&error), 2);
        assert!(target(&[OutputDestination { display: Some(OutputDisplay::Srgb), ..raw(S::ACEScg) }]).is_err());
        for space in [S::LinearRec709, S::LinearRec2020, S::ACEScg] {
            super::admit(&world, &[raw(space), "still.png".into()], Target::default()).unwrap();
            assert_eq!(RawColor::of(&world, &raw(space)).storage, space);
        }
        for space in [S::Srgb, S::DisplayP3] {
            let error = super::admit(&world, &[raw(space)], Target::default()).unwrap_err();
            assert_eq!(error.root_cause().to_string(),
                format!("output.color_space {} is not linear; a world's EXR takes Linear Rec.709, Linear Rec.2020 or ACEScg", space.name()));
            assert_eq!(crate::failure::exit_code(&error), 2);
        }
        // Omitted, the storage is the working space and the values are stored as they are, bit for bit.
        let default = RawColor::of(&world, &"frames.%06d.exr".into());
        assert_eq!(default, RawColor::default());
        let signed = [-0., -0.2, 1000.];
        assert_eq!(default.store(signed).map(f32::to_bits), signed.map(f32::to_bits));
        let stored = RawColor::of(&world, &raw(S::LinearRec2020)).store([2., -0.25, 0.5]);
        for (a, b) in stored.into_iter().zip([1.19414353, -0.08600947, 0.45857722]) { assert!((a - b).abs() < 2e-6, "{a} against {b}"); }
        assert_eq!(default.reference_domain(), "scene");
        // A world's conversion is linear: an associated pixel is converted as it is, coverage or none.
        assert_eq!(default.store_associated([-0., -0.2, 1000., 0.]).map(f32::to_bits), [-0., -0.2, 1000., 0.].map(f32::to_bits));
        // A canvas's EXR is display-referred and takes every space. Omitted, it stores the canvas's sRGB code
        // values as they are; a linear space takes display-linear light, converted on unassociated colour.
        let canvas: v2::Scene = serde_json::from_value(serde_json::json!({"version":2,"composition":{"space":"2d"},"nodes":[]})).unwrap();
        for space in [S::Srgb, S::DisplayP3, S::LinearRec709, S::LinearRec2020, S::ACEScg] {
            super::admit(&canvas, &[raw(space)], Target::default()).unwrap();
            assert_eq!(RawColor::of(&canvas, &raw(space)).storage, space);
        }
        let own = RawColor::of(&canvas, &"frames.%06d.exr".into());
        assert_eq!((own.storage, own.reference_domain()), (S::Srgb, "display"));
        assert_eq!(own.store_associated([0.25, 0.5, 0.75, 0.5]).map(f32::to_bits), [0.25, 0.5, 0.75, 0.5].map(f32::to_bits));
        let linear = RawColor::of(&canvas, &raw(S::LinearRec709)).store_associated([0.25, 0.25, 0.25, 0.5]);
        // Code 0.5 at half coverage: linear 0.21404114, associated again.
        assert!((linear[0] - 0.21404114 * 0.5).abs() < 1e-6 && linear[3] == 0.5, "{linear:?}");
        // A canvas that blends in another encoding hands a reader its view's sRGB code values all the same.
        // Omitted, the storage is its working space: a linear canvas's file holds display-linear light, a
        // Display P3 canvas's that display's code values.
        let in_space = |space: &str| -> v2::Scene {
            serde_json::from_value(serde_json::json!({"version":2,"composition":{"space":"2d","color_space":space},"nodes":[]})).unwrap()
        };
        let linear = RawColor::of(&in_space("Linear Rec.709"), &"frames.%06d.exr".into());
        assert_eq!((linear.storage, linear.reference_domain()), (S::LinearRec709, "display"));
        assert!((linear.store([0.5, 0.5, 0.5])[0] - 0.21404114).abs() < 1e-6);
        let wide = RawColor::of(&in_space("Display P3"), &"frames.%06d.exr".into());
        assert_eq!(wide.storage, S::DisplayP3);
        // sRGB red as Display P3 code values.
        for (a, b) in wide.store([1., 0., 0.]).into_iter().zip([0.9174876, 0.2002867, 0.1385606]) { assert!((a - b).abs() < 2e-5, "{a} against {b}"); }
        // Its sRGB storage is the reader's picture as it is, bit for bit.
        let own = RawColor::of(&in_space("ACEScg"), &raw(S::Srgb));
        assert_eq!(own.store_associated([0.25, 0.5, 0.75, 0.5]).map(f32::to_bits), [0.25, 0.5, 0.75, 0.5].map(f32::to_bits));
    }
    fn out(file: &str, display: OutputDisplay) -> OutputDestination {
        OutputDestination { display: Some(display), ..file.into() }
    }
    fn scene(view: serde_json::Value) -> v2::Scene {
        serde_json::from_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"composition":{"space":"3d","view":view}})).unwrap()
    }
    #[test]
    fn destinations_share_one_display_and_unbuilt_ones_refuse_in_one_line() {
        assert_eq!(target(&["still.png".into(), "frames.%06d.exr".into()]).unwrap(), Target::default());
        assert_eq!(target(&[out("still.png", OutputDisplay::Srgb), "movie.mp4".into()]).unwrap().display, OutputDisplay::Srgb);
        // Display P3 is built for PNG stills, frame directories and video; raw EXR beside it takes no display.
        let wide = target(&[out("still.png", OutputDisplay::DisplayP3), out("frames/", OutputDisplay::DisplayP3), out("movie.mp4", OutputDisplay::DisplayP3),
            out("movie.mov", OutputDisplay::DisplayP3), "raw.%06d.exr".into()]).unwrap();
        assert_eq!(wide.display, OutputDisplay::DisplayP3);
        assert_eq!((wide.video_primaries(), Target::default().video_primaries()), (vork_encode::Primaries::DisplayP3, vork_encode::Primaries::Bt709));
        // Video frames are 8-bit, so Rec.2100 video refuses.
        for (display, file) in [(OutputDisplay::Rec2100Pq, "movie.mp4"), (OutputDisplay::Rec2100Hlg, "movie.mov")] {
            let error = target(&[out(file, display)]).unwrap_err();
            assert_eq!(error.root_cause().to_string(), format!("output.display {} for video is not built yet: video here is 8-bit; write PNG frames to a directory", display.name()));
            assert_eq!(crate::failure::exit_code(&error), 2);
        }
        // An omitted display is sRGB, so it does not join a Display P3 destination.
        let error = target(&["still.png".into(), out("wide.png", OutputDisplay::DisplayP3)]).unwrap_err();
        assert!(error.root_cause().to_string().starts_with("one render call encodes for one display, and these destinations ask for sRGB and Display P3"), "{error:#}");
        assert_eq!(crate::failure::exit_code(&error), 2);
        let error = target(&[out("frames.%06d.exr", OutputDisplay::Srgb)]).unwrap_err();
        assert!(error.root_cause().to_string().contains("EXR output is not encoded for a display"));
        assert_eq!(crate::failure::exit_code(&error), 2);
        // A trailing separator marks a directory of viewed PNG frames, whatever its name ends in.
        assert_eq!(target(&[out("frames.exr/", OutputDisplay::Srgb)]).unwrap().display, OutputDisplay::Srgb);
    }
    #[test]
    fn a_canvas_root_encodes_sdr_displays() {
        let scene = |space: &str| -> v2::Scene {
            serde_json::from_value(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"composition":{"space":space}})).unwrap()
        };
        admit(&scene("2d"), Target::default()).unwrap();
        admit(&scene("2d"), Target::for_display(OutputDisplay::DisplayP3)).unwrap();
        admit(&scene("3d"), Target::for_display(OutputDisplay::Rec2100Pq)).unwrap();
        let error = admit(&scene("2d"), Target::for_display(OutputDisplay::Rec2100Pq)).unwrap_err();
        assert_eq!(error.root_cause().to_string(), "output.display Rec.2100-PQ on a 2D canvas is not built yet: a canvas is 8-bit; use sRGB or Display P3");
        assert_eq!(crate::failure::exit_code(&error), 2);
    }
    #[test]
    fn a_bundled_view_renders_a_wider_display_only_as_it_was_baked() {
        let p3 = Target::for_display(OutputDisplay::DisplayP3);
        let scene = |value: serde_json::Value| -> vork_scene::v2::Scene { serde_json::from_value(value).unwrap() };
        let world = |view: serde_json::Value| scene(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"composition":{"space":"3d","view":view}}));
        let keyed = |keys: serde_json::Value| scene(serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "materials":{"grade":{"program":"std/agx"}},"composition":{"space":"3d","view":"grade"},
            "animation":[{"tracks":[{"path":["materials","grade","inputs","look"],"keys":keys}]}]}));
        // The default look with the library's table is AgX's own Display P3 rendering; Standard and views that
        // return XYZ are encoded for any display; an author's own destination table is read under the same look.
        for view in [serde_json::json!({"program":"std/agx"}), serde_json::json!({"program":"std/agx","inputs":{"look":"none","gamma":2.}}),
            serde_json::json!({"program":"std/standard"}), serde_json::json!({"program":"blender/filmic"}),
            serde_json::json!({"program":"std/agx","inputs":{"destination_lut":{"image":"mine"}}})] {
            admit(&world(view), p3).unwrap();
        }
        admit(&keyed(serde_json::json!([[0,"none"],[1,"none"]])), p3).unwrap();
        // A look, a track that leaves the default look, and an authored sRGB table have only sRGB renderings.
        for refused in [world(serde_json::json!({"program":"std/agx","inputs":{"look":"punchy"}})),
            world(serde_json::json!({"program":"std/agx","inputs":{"look":"punchy","destination_lut":{"image":"mine"}}})),
            world(serde_json::json!({"program":"std/agx","inputs":{"lut":{"image":"mine"}}})),
            keyed(serde_json::json!([[0,"none"],[1,"punchy"]]))] {
            let error = admit(&refused, p3).unwrap_err();
            assert_eq!(error.root_cause().to_string(), "std/agx renders Display P3 only with look none and the library's lut; use that, or output.display sRGB");
            assert_eq!(crate::failure::exit_code(&error), 2);
            // Every look and an authored table render for sRGB.
            admit(&refused, Target::default()).unwrap();
        }
    }
    #[test]
    fn hdr_destinations_carry_their_luminances_and_take_sixteen_bits() {
        for display in [OutputDisplay::Rec2100Pq, OutputDisplay::Rec2100Hlg] {
            // Omitted: a 1000 cd/m² display with reference white 203 (BT.2408).
            let default = target(&[out("still.png", display)]).unwrap();
            assert_eq!(default, Target { display, peak: 1000., white: 203. });
            assert_eq!(default, Target::for_display(display));
            assert_eq!(default.png_depth(crate::PngDepth::Eight), crate::PngDepth::Sixteen);
            let authored = OutputDestination { peak_luminance: Some(600.), reference_white: Some(100.), ..out("still.png", display) };
            assert_eq!(target(&[authored.clone()]).unwrap(), Target { display, peak: 600., white: 100. });
            // Destinations of one call share the luminances too.
            let error = target(&[authored, out("other.png", display)]).unwrap_err();
            assert!(error.root_cause().to_string().contains("different luminances"), "{error:#}");
            for (peak, white) in [(Some(0.), None), (Some(100.), Some(101.)), (Some(10001.), None), (None, Some(f32::NAN))] {
                let error = target(&[OutputDestination { peak_luminance: peak, reference_white: white, ..out("still.png", display) }]).unwrap_err();
                assert_eq!(crate::failure::exit_code(&error), 2, "{error:#}");
            }
        }
        assert_eq!(Target::for_display(OutputDisplay::DisplayP3).png_depth(crate::PngDepth::Eight), crate::PngDepth::Eight);
        // Unit white below the HLG surround floor would not land on the reference white.
        let dim = OutputDestination { reference_white: Some(0.001), ..out("still.png", OutputDisplay::Rec2100Hlg) };
        assert!(target(&[dim.clone()]).unwrap_err().root_cause().to_string().contains("surround's floor"));
        assert!(target(&[OutputDestination { display: Some(OutputDisplay::Rec2100Pq), ..dim }]).is_ok());
        // SDR luminances are the viewer's; raw output takes none.
        for display in [OutputDisplay::Srgb, OutputDisplay::DisplayP3] {
            // Refused whatever the value, the SDR nominal 100 included.
            for (peak, white) in [(Some(200.), None), (Some(100.), None), (None, Some(100.)), (Some(100.), Some(100.))] {
                let error = target(&[OutputDestination { peak_luminance: peak, reference_white: white, ..out("still.png", display) }]).unwrap_err();
                assert_eq!(error.root_cause().to_string(),
                    format!("output.peak_luminance and output.reference_white describe a Rec.2100 display; omit them for {}", display.name()));
                assert_eq!(crate::failure::exit_code(&error), 2);
            }
            assert_eq!(target(&[out("still.png", display)]).unwrap(), Target::for_display(display));
        }
        assert!(target(&[OutputDestination { reference_white: Some(100.), .."raw.%06d.exr".into() }]).is_err());
    }
    #[test]
    fn hlg_system_gamma_follows_bt2100() {
        assert!((hlg_gamma(1000.) - 1.2).abs() < 1e-6);
        assert!((hlg_gamma(2000.) - 1.326432).abs() < 1e-5);
        assert!((hlg_gamma(400.) - 1.032866).abs() < 1e-5);
        // The extended formula meets the table's at 1000 cd/m² and continues beyond its range.
        assert!((hlg_gamma(4000.) - 1.2 * 1.111f32.powi(2)).abs() < 1e-5);
        assert!((hlg_floor(1000.) - 1e-4 / 3f32.powf(1.2)).abs() < 1e-9);
    }
    #[test]
    fn a_bundled_view_refuses_luminances_it_is_not_baked_for() {
        for display in [OutputDisplay::Rec2100Pq, OutputDisplay::Rec2100Hlg] {
            let default = Target::for_display(display);
            let agx = scene(serde_json::json!({"program":"std/agx"}));
            admit(&agx, default).unwrap();
            admit(&agx, Target { white: 100., ..default }).unwrap();
            for other in [Target { white: 150., ..default }, Target { peak: 600., ..default }, Target { peak: 4000., ..default }] {
                let error = admit(&agx, other).unwrap_err();
                assert_eq!(error.root_cause().to_string(), format!("std/agx renders {} at output.peak_luminance/output.reference_white 1000/203 or 1000/100; other luminances are not built yet", display.name()));
                assert_eq!(crate::failure::exit_code(&error), 2);
                // Views that return XYZ, or encode analytically, take any luminance.
                admit(&scene(serde_json::json!({"program":"std/standard"})), other).unwrap();
                admit(&scene(serde_json::json!({"program":"blender/filmic"})), other).unwrap();
            }
            // An author's own destination table is read by the same shader, for the same display.
            let own = scene(serde_json::json!({"program":"std/agx","inputs":{"destination_lut":{"image":"mine"}}}));
            admit(&own, default).unwrap();
            assert!(admit(&own, Target { peak: 2000., white: 100., ..default }).unwrap_err().root_cause().to_string().contains("other luminances are not built yet"));
            // A look and an authored table are sRGB renderings: no luminances make them an HDR one.
            for view in [serde_json::json!({"program":"std/agx","inputs":{"look":"punchy"}}), serde_json::json!({"program":"std/agx","inputs":{"lut":{"image":"mine"}}})] {
                let error = admit(&scene(view), default).unwrap_err();
                assert_eq!(error.root_cause().to_string(), format!("std/agx renders {} only with look none and the library's lut; use that, or output.display sRGB", display.name()));
            }
        }
        // The Display P3 and sRGB renderings carry no luminance.
        admit(&scene(serde_json::json!({"program":"std/agx"})), Target::for_display(OutputDisplay::DisplayP3)).unwrap();
        admit(&scene(serde_json::json!({"program":"std/agx"})), Target::default()).unwrap();
    }
}
