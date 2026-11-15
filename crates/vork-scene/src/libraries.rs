//! Bundled Slang libraries, shared by validation and every compiler workspace.
//!
//! A module named `<library>/<module>` is imported as `import <library>.<module>;`;
//! a material program is referenced as `"program": "<library>/<module>"`.
pub struct Module {
    pub name: &'static str,
    pub source: &'static str,
    /// Whether the module exports an authored material stage rather than library functions.
    pub material: bool,
}
#[derive(Clone, Copy)]
pub enum ProgramImage {
    Embedded(&'static [u8]),
    /// Path relative to the canonical executable's directory.
    File(&'static str),
}

pub struct Library {
    pub name: &'static str,
    /// `vork help <library>`.
    pub help: &'static str,
    pub modules: &'static [Module],
    /// Render profiles `<library>/<name>`: settings-tier data files.
    pub profiles: &'static [(&'static str, &'static str)],
    /// Data a bundled program reads through an `Image` input the author leaves
    /// unbound. It ships with the library, like the program's source.
    pub images: &'static [ProgramData],
}
/// One bundled program's data for one unbound `Image` input (an OpenEXR source).
pub struct ProgramData {
    pub program: &'static str,
    pub input: &'static str,
    pub image: ProgramImage,
    /// The output displays this data is the program's rendering for; empty: every display. For any other
    /// display the input stays unbound.
    pub displays: &'static [crate::v2::OutputDisplay],
    /// The `[peak_luminance, reference_white]` pairs in cd/m² this data is baked for; empty: any.
    pub luminances: &'static [[f32; 2]],
    /// The input whose library data this stands in for at those displays. The program has that display's
    /// rendering only with the library's own data there: an authored binding of that input is refused.
    pub instead_of: Option<&'static str>,
    /// The enum input and token under which the program reads this data: the only value of that input the
    /// program has this display's rendering for. None: every value.
    pub read_when: Option<(&'static str, &'static str)>,
}

macro_rules! modules {
    ($library:literal: $($module:literal $material:literal),* $(,)?) => {
        &[$(Module {
            name: concat!($library, "/", $module),
            source: include_str!(concat!("../../../libraries/", $library, "/", $module, ".slang")),
            material: $material,
        }),*]
    };
}

// The tables are statics: each is in the executable once, and `LIBRARIES` points at
// it. As consts their bytes were copied into every crate that names `LIBRARIES`
// (six copies of 3.3 MB that does not compress).
/// Blender AgX samples owned by the std display program.
static AGX_LUT: [u8; include_bytes!("../../../libraries/std/agx.exr").len()] = *include_bytes!("../../../libraries/std/agx.exr");
/// Blender's AgX view for Display P3, as its original OCIO cube (libraries/std/device-views-manifest.json).
static AGX_P3: [u8; include_bytes!("../../../libraries/std/agx-p3.exr").len()] = *include_bytes!("../../../libraries/std/agx-p3.exr");
/// Blender's "AgX - HDR 1000 nits" view, as its original OCIO cube: Rec.2100 HLG for a 1000 cd/m² display.
static AGX_HDR: [u8; include_bytes!("../../../libraries/std/agx-pq.exr").len()] = *include_bytes!("../../../libraries/std/agx-pq.exr");

pub const LIBRARIES: &[Library] = &[
    // Fundamental and hard to write correctly. `vork setup` prebuilds its programs.
    Library {
        name: "std",
        help: include_str!("../../../libraries/std/HELP.txt"),
        modules: modules!("std":
            "openpbr_core" false, "texture" false, "normal_filter" false, "hair" true, "hair_bsdf" false,
            "camera" true, "sky" true, "environment_image" true, "grid" false, "pnanovdb" false,
            "color" false, "white_balance" false, "bloom" true, "procedurals" false, "vector" false,
            "agx" true, "agx_destination" false, "standard" true, "view" false, "gaussian_math" false, "gaussian" true),
        profiles: &[
            ("std/previs", include_str!("../../../libraries/std/profiles/previs.json")),
            ("std/draft", include_str!("../../../libraries/std/profiles/draft.json")),
            ("std/final", include_str!("../../../libraries/std/profiles/final.json")),
            ("std/path", include_str!("../../../libraries/std/profiles/path.json")),
        ],
        images: &[
            ProgramData { program: "std/agx", input: "lut", image: ProgramImage::Embedded(&AGX_LUT), displays: &[], luminances: &[], instead_of: None, read_when: None },
            // The destination bakes are the default look's; the other looks are baked for sRGB only.
            ProgramData { program: "std/agx", input: "destination_lut", image: ProgramImage::Embedded(&AGX_P3),
                displays: &[crate::v2::OutputDisplay::DisplayP3], luminances: &[], instead_of: Some("lut"), read_when: Some(("look", "none")) },
            // The view is published for a 1000 cd/m² display. Its two recipes are the manifest's: unit scene
            // white at 203 cd/m² (the default reference white), and the view as published (reference white 100).
            ProgramData { program: "std/agx", input: "destination_lut", image: ProgramImage::Embedded(&AGX_HDR),
                displays: &[crate::v2::OutputDisplay::Rec2100Pq, crate::v2::OutputDisplay::Rec2100Hlg],
                luminances: &[[1000., 203.], [1000., 100.]], instead_of: Some("lut"), read_when: Some(("look", "none")) },
        ],
    },
    // Blender node and socket compatibility for the bridge and Blender-faithful scenes.
    Library {
        name: "blender",
        help: include_str!("../../../libraries/blender/HELP.txt"),
        modules: modules!("blender":
            "nodes" false, "volume" false, "white_balance" false,
            "texture_coordinate" false, "color" false, "vector" false, "procedurals" false, "filmic" true, "display_lut" false),
        profiles: &[],
        images: &[ProgramData { program: "blender/filmic", input: "lut", image: ProgramImage::File("libraries/blender/filmic.exr"), displays: &[], luminances: &[], instead_of: None, read_when: None }],
    },
    Library {
        name: "fluid",
        help: include_str!("../../../libraries/fluid/HELP.txt"),
        modules: modules!("fluid": "config" false, "solver" false, "surface" false, "whitewater" false, "film" false, "particles" false, "volume" false, "reconstruction" false),
        profiles: &[],
        images: &[],
    },
    Library {
        name: "cloth",
        help: include_str!("../../../libraries/cloth/HELP.txt"),
        modules: modules!("cloth": "solver" false, "body" false, "self" false),
        profiles: &[],
        images: &[],
    },


];

impl Library {
    pub fn programs(&self) -> impl Iterator<Item = &'static str> {
        self.modules.iter().filter(|module| module.material).map(|module| module.name)
    }
}
/// The library `vork setup` prebuilds, so a scene using only its programs compiles
/// nothing at render time. Other libraries compile on first use.
pub const PREBUILT: &str = "std";
pub fn prebuilt_programs() -> impl Iterator<Item = &'static str> {
    library(PREBUILT).expect("prebuilt library").programs()
}

/// A bundled profile's file text by `<library>/<name>`.
pub fn profile(name: &str) -> Option<&'static str> {
    LIBRARIES.iter().flat_map(|library| library.profiles).find(|(key, _)| *key == name).map(|(_, text)| *text)
}
pub fn profile_names() -> impl Iterator<Item = &'static str> {
    LIBRARIES.iter().flat_map(|library| library.profiles).map(|(name, _)| *name)
}

pub fn library(name: &str) -> Option<&'static Library> {
    LIBRARIES.iter().find(|library| library.name == name)
}
/// Every bundled module of every library.
pub fn modules() -> impl Iterator<Item = &'static Module> {
    LIBRARIES.iter().flat_map(|library| library.modules)
}
pub fn module_source(name: &str) -> Option<&'static str> {
    modules().find(|module| module.name == name).map(|module| module.source)
}
/// The library data a bundled program's unbound `Image` input reads when its result is for `display`. The
/// caller checks `luminances` against the destination's.
pub fn program_image(program: &str, input: &str, display: crate::v2::OutputDisplay) -> Option<&'static ProgramData> {
    LIBRARIES.iter().flat_map(|library| library.images)
        .find(|data| data.program == program && data.input == input && (data.displays.is_empty() || data.displays.contains(&display)))
}
impl ProgramData {
    /// Whether this data is baked for a destination of these luminances.
    pub fn baked_for(&self, peak: f32, white: f32) -> bool {
        self.luminances.is_empty() || self.luminances.contains(&[peak, white])
    }
}
/// The built-in table material and every bundled material program.
pub fn material_programs<'a>() -> impl Iterator<Item = &'a str> {
    std::iter::once("std/openpbr").chain(modules().filter(|module| module.material).map(|module| module.name))
}

/// Error-only guidance for removed module names; these never resolve to source.
pub fn removed_module(name: &str) -> Option<String> {
    let replacement = match name {
        // std.white_balance is a library, not a program: the stage is the author's own.
        "post/white_balance" | "post.white_balance" =>
            "std.white_balance is a library, not a program: write a composite program that imports it and returns white_balance(color, inputs)",
        "std/particles" | "std.particles" => "use fluid.particles",
        "std/display" | "std.display" => "use std.view: finish_view(srgb, ctx, gamma)",
        _ => return None,
    };
    Some(format!("{name} was removed; {replacement}; see `vork help upgrade`"))
}

#[cfg(test)]
#[test]
fn compatibility_expires_in_06() {
    for name in ["post/white_balance", "std/particles"] {
        assert!(module_source(name).is_none(), "removed module {name} still registered");
    }
}

#[test]
fn permanent_std_homes_are_visible_and_unique() {
    for name in ["std/bloom", "std/procedurals", "std/vector", "std/color", "std/white_balance"] {
        assert!(library("std").unwrap().modules.iter().any(|module| module.name == name));
        assert_eq!(modules().filter(|module| module.name == name).count(), 1);
    }
}
