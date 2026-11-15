//! Shared test fixtures over fonts already shipped in this repository.
//! Four faces come from the private world-type shot (`world_type_fonts`), so tests that
//! ask the store for them are built only with `vork_private_tests`.

use std::path::{Path, PathBuf};

use crate::catalog::Catalog;
use crate::vector::measure::VectorFontStore;

/// Repository root for the bundled test faces; no system-font discovery.
pub(crate) fn fonts_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

pub(crate) fn font_path(rel: &str) -> PathBuf {
    fonts_root().join(match rel {
        "inter/Inter_24pt-Regular.ttf" => {
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-Regular.ttf"
        }
        "lato/Lato-Regular.ttf" => crate::world_type_fonts::WORLD_TYPE_LATO,
        // This subset preserves the A/V direct kerning used by the test.
        "fraunces/Fraunces[SOFT,WONK,opsz,wght]-Regular.ttf" => {
            "crates/moda-text/tests/fonts/Fraunces-SpacingSubset.ttf"
        }
        _ => rel,
    })
}

/// Moda's oracle font corpus, which this repository does not carry. The tests that wait
/// on it are ignored for that reason and load it through these two, so supplying the
/// corpus at `fixtures/oracle/fonts` is all they need.
pub(crate) fn oracle_fonts_root() -> PathBuf {
    fonts_root().join("fixtures/oracle/fonts")
}

pub(crate) fn oracle_store_and_catalog() -> (VectorFontStore, Catalog) {
    let root = oracle_fonts_root();
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(root.join(name)).unwrap()).unwrap()
    };
    let catalog = Catalog::from_catalog_json(&read("catalog.json"));
    (
        VectorFontStore::load_from_manifest_for_tests(&root, &read("MANIFEST.json")),
        catalog,
    )
}

pub(crate) fn store_and_catalog() -> (VectorFontStore, Catalog) {
    let faces = [
        (
            "Inter",
            400,
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-Regular.ttf",
        ),
        (
            "Inter",
            800,
            "crates/moda-engine/tests/fixtures/fonts/inter/Inter_24pt-ExtraBold.ttf",
        ),
        ("Lato", 400, crate::world_type_fonts::WORLD_TYPE_LATO),
        (
            "Noto Sans",
            400,
            "crates/moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf",
        ),
        (
            "Noto Sans Arabic",
            400,
            "crates/vork-std/tests/fonts/NotoSansArabic-Regular.ttf",
        ),
        ("Noto Sans Hebrew", 400, crate::world_type_fonts::WORLD_TYPE_HEBREW),
        ("Noto Sans Thai", 400, crate::world_type_fonts::WORLD_TYPE_THAI),
        ("Noto Sans Devanagari", 400, crate::world_type_fonts::WORLD_TYPE_DEVANAGARI),
        (
            "Fraunces",
            400,
            "crates/moda-text/tests/fonts/Fraunces-SpacingSubset.ttf",
        ),
    ];
    let mut families = std::collections::BTreeMap::<&str, Vec<serde_json::Value>>::new();
    let mut files = serde_json::Map::new();
    for (family, weight, path) in faces {
        let url = format!("test://{path}");
        families.entry(family).or_default().push(serde_json::json!({
            "weight": weight, "style": "normal", "ttf_url": url
        }));
        files.insert(url, serde_json::json!({ "path": path }));
    }
    let platform_fonts: Vec<_> = families
        .into_iter()
        .map(
            |(family, variants)| serde_json::json!({ "family_name": family, "variants": variants }),
        )
        .collect();
    let catalog =
        Catalog::from_catalog_json(&serde_json::json!({ "platform_fonts": platform_fonts }));
    let manifest = serde_json::json!({ "files": files });
    (
        VectorFontStore::load_from_manifest_for_tests(&fonts_root(), &manifest),
        catalog,
    )
}

/// E6.3: a store and catalog over one variable face, Inter (OFL; axes `opsz`
/// 14–32 and `wght` 100–900), vendored in `fixtures/fonts-variable`. The
/// catalog has "Inter Variable" at 400 (the default instance) and 700 (a named
/// instance pinned at `wght` 700), the shape the desktop's system-font catalog
/// gives a variable font.
pub(crate) fn variable_store_and_catalog() -> (VectorFontStore, Catalog) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .join("fixtures/fonts-variable");
    let url = "test://fonts-variable/InterVariable.ttf";
    let catalog = Catalog::from_catalog_json(&serde_json::json!({
        "platform_fonts": [{
            "family_name": "Inter Variable",
            "variants": [
                { "weight": 400, "style": "normal", "ttf_url": url, "variations": { "wght": 400 } },
                { "weight": 700, "style": "normal", "ttf_url": url, "variations": { "wght": 700 } }
            ]
        }]
    }));
    let manifest = serde_json::json!({ "files": { url: { "path": "InterVariable.ttf" } } });
    (
        VectorFontStore::load_from_manifest_for_tests(&root, &manifest),
        catalog,
    )
}
