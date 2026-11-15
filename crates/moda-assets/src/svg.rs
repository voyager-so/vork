//! SVG rasterization via resvg with a catalog-seeded fontdb and the
//! export-aware minimum raster size (`svgRasterizationSizing.ts` +
//! `WebGPUTextureManager.resolveImageBitmap`'s SVG branch).
//!
//! The browser path: HTMLImageElement gives natural CSS-px dimensions from
//! the SVG width/height/viewBox; `createImageBitmap(img, {resizeWidth,
//! resizeHeight})` re-rasterizes the VECTOR at the target size. resvg renders
//! at the same target size via a scale transform — vector-sharp, like Blink.

use anyhow::{anyhow, Context, Result};
use resvg::usvg;
use std::sync::Arc;

use crate::decode::PreparedImage;

/// Build a fontdb seeded from explicit font files (the headless `fonts[]`
/// catalog / vendored fixture fonts). System fonts are deliberately NOT
/// loaded: a headless renderer must be deterministic across hosts; SVGs with
/// <text> and no catalog coverage are flagged by the manifest probe instead.
pub fn build_fontdb(font_paths: &[std::path::PathBuf]) -> usvg::fontdb::Database {
    let mut db = usvg::fontdb::Database::new();
    for p in font_paths {
        let _ = db.load_font_file(p);
    }
    db
}

/// The fontdb SVG `<text>` is laid out with (resvg's).
pub type FontDatabase = usvg::fontdb::Database;

type FontdbProvider = Box<dyn Fn() -> Option<Arc<usvg::fontdb::Database>> + Send + Sync>;
static FONTDB_PROVIDER: std::sync::OnceLock<FontdbProvider> = std::sync::OnceLock::new();

/// Install the process-global source of the engine's fonts for SVG `<text>` (first call wins).
/// `moda-engine` installs one backed by the text engine's font system, so an SVG image's text
/// resolves families the way text nodes do. Without one, SVG text has no fonts and is dropped.
pub fn install_fontdb_provider(provider: FontdbProvider) {
    let _ = FONTDB_PROVIDER.set(provider);
}

/// The engine's fonts for SVG `<text>` (see [`install_fontdb_provider`]); empty when no provider
/// is installed or it has no fonts yet.
pub fn engine_fontdb() -> Arc<usvg::fontdb::Database> {
    static EMPTY: std::sync::OnceLock<Arc<usvg::fontdb::Database>> = std::sync::OnceLock::new();
    FONTDB_PROVIDER
        .get()
        .and_then(|provider| provider())
        .unwrap_or_else(|| {
            EMPTY
                .get_or_init(|| Arc::new(usvg::fontdb::Database::new()))
                .clone()
        })
}

/// Whether a family name reads as serif — the text engine's rule (`moda_text::runs`).
fn is_serif_family_name(name: &str) -> bool {
    let lower = name.to_lowercase();
    if lower.contains("slab") {
        return true;
    }
    lower.match_indices("serif").any(|(idx, _)| {
        !lower[..idx]
            .strip_suffix(|c: char| c == '-' || c.is_whitespace())
            .is_some_and(|rest| rest.ends_with("sans"))
    })
}

/// A requested family as this fontdb spells it. fontdb matches names exactly, CSS and the text
/// engine case-insensitively.
fn fontdb_family_name<'a>(db: &'a usvg::fontdb::Database, name: &str) -> Option<&'a str> {
    db.faces()
        .flat_map(|face| face.families.iter())
        .find(|(n, _)| n.eq_ignore_ascii_case(name))
        .map(|(n, _)| n.as_str())
}

/// usvg options whose font selection follows the text engine: the requested families in order
/// (case-insensitive), and when none is present, the engine's serif or sans fallback face instead of
/// dropping the text, logged like any other substituted font. The engine's fontdb has its generic
/// families pointed at faces it has.
pub fn usvg_options(fontdb: &Arc<usvg::fontdb::Database>) -> usvg::Options<'static> {
    let mut options = usvg::Options {
        fontdb: fontdb.clone(),
        ..usvg::Options::default()
    };
    options.font_resolver.select_font = Box::new(|font, db| {
        use usvg::fontdb::{Family, Query, Stretch, Style, Weight};
        use usvg::{FontFamily, FontStretch, FontStyle};
        let stretch = match font.stretch() {
            FontStretch::UltraCondensed => Stretch::UltraCondensed,
            FontStretch::ExtraCondensed => Stretch::ExtraCondensed,
            FontStretch::Condensed => Stretch::Condensed,
            FontStretch::SemiCondensed => Stretch::SemiCondensed,
            FontStretch::Normal => Stretch::Normal,
            FontStretch::SemiExpanded => Stretch::SemiExpanded,
            FontStretch::Expanded => Stretch::Expanded,
            FontStretch::ExtraExpanded => Stretch::ExtraExpanded,
            FontStretch::UltraExpanded => Stretch::UltraExpanded,
        };
        let style = match font.style() {
            FontStyle::Normal => Style::Normal,
            FontStyle::Italic => Style::Italic,
            FontStyle::Oblique => Style::Oblique,
        };
        let query = |families: &[Family]| {
            db.query(&Query {
                families,
                weight: Weight(font.weight()),
                stretch,
                style,
            })
        };
        let mut serif = false;
        let mut requested = Vec::new();
        let families: Vec<Family> = font
            .families()
            .iter()
            .filter_map(|family| match family {
                FontFamily::Serif => {
                    serif = true;
                    Some(Family::Serif)
                }
                FontFamily::SansSerif => Some(Family::SansSerif),
                FontFamily::Cursive => Some(Family::Cursive),
                FontFamily::Fantasy => Some(Family::Fantasy),
                FontFamily::Monospace => Some(Family::Monospace),
                FontFamily::Named(name) => {
                    serif |= is_serif_family_name(name);
                    requested.push(name.as_str());
                    fontdb_family_name(db, name).map(Family::Name)
                }
            })
            .collect();
        if let Some(id) = query(&families) {
            return Some(id);
        }
        let fallback = query(&[if serif {
            Family::Serif
        } else {
            Family::SansSerif
        }])
        .or_else(|| query(&[Family::SansSerif]))?;
        let drawn = db
            .face(fallback)
            .and_then(|face| face.families.first())
            .map(|(n, _)| n.as_str())
            .unwrap_or_default();
        crate::report::record(crate::report::RuntimeEvent::FontSubstituted {
            node_id: "svg-image".to_string(),
            requested: requested.join(", "),
            drawn: drawn.to_string(),
        });
        Some(fallback)
    });
    options
}

/// Natural dimensions as HTMLImageElement reports them/// Natural dimensions as HTMLImageElement reports them (CSS px, from
/// width/height attrs or viewBox). usvg resolves this the same way.
pub fn svg_natural_size(tree: &usvg::Tree) -> (u32, u32) {
    let s = tree.size();
    // HTMLImageElement.naturalWidth rounds the CSS size.
    let w = s.width().round().max(1.0) as u32;
    let h = s.height().round().max(1.0) as u32;
    (w, h)
}

/// Rasterize SVG bytes at the production target size. `requested_min` is the
/// per-node desired min raster size (clamped; SVG-only). Returns premultiplied
/// RGBA8 (tiny-skia pixmaps are premultiplied) at the resolved target size.
pub fn prepare_svg(
    bytes: &[u8],
    requested_min: Option<(u32, u32)>,
    fontdb: &Arc<usvg::fontdb::Database>,
) -> Result<PreparedImage> {
    let tree = usvg::Tree::from_data(bytes, &usvg_options(fontdb)).context("usvg parse")?;
    let (nw, nh) = svg_natural_size(&tree);
    let (tw, th) = crate::sizing::resolve_target_size(nw, nh, true, requested_min);

    let mut pixmap =
        resvg::tiny_skia::Pixmap::new(tw, th).ok_or_else(|| anyhow!("pixmap alloc {tw}x{th}"))?;
    let sx = tw as f32 / tree.size().width();
    let sy = th as f32 / tree.size().height();
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::from_scale(sx, sy),
        &mut pixmap.as_mut(),
    );

    // Coverage verdict — one sequential pass over the rasterized
    // pixels BEFORE `take()` consumes the pixmap. Full scan, never strided: a
    // logo can occupy well under 1% of a 1024² pixmap, and a stride that misses
    // it would report a perfectly good asset as blank.
    let non_transparent = pixmap.pixels().iter().filter(|p| p.alpha() != 0).count() as u64;
    let coverage = crate::decode::coverage_permille(non_transparent, (tw as u64) * (th as u64));

    Ok(PreparedImage {
        width: tw,
        height: th,
        natural_width: nw,
        natural_height: nh,
        scale_factor: (tw as f64 / nw.max(1) as f64, th as f64 / nh.max(1) as f64),
        rgba_premul: Arc::new(pixmap.take()),
        is_svg: true,
        icc_applied: false,
        coverage_permille: coverage,
        // Only computed for a BLANK raster, so the raw-markup scan (which
        // lowercases up to 256 KiB) never runs on the healthy path — this is
        // also the wasm editor's hot decode path.
        straight: false,
        blank_cause: if coverage > 0 {
            ""
        } else if fontdb.is_empty() && crate::probe::raw_markup_has_text(bytes) {
            // No fonts yet (the wasm editor lane before its first font
            // install, or no engine fonts provider): usvg drops every
            // unlayoutable <text> and a text-only SVG rasterizes to nothing.
            crate::decode::BLANK_CAUSE_SVG_TEXT_NO_FONTDB
        } else {
            crate::decode::BLANK_CAUSE_FULLY_TRANSPARENT
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAR: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80"><rect width="80" height="80" fill="#fdf6e3"/></svg>"##;

    #[test]
    fn svg_rasterizes_at_min_dimension_policy_size() {
        let _report = crate::report::tests::serial_globals();
        let db = Arc::new(build_fontdb(&[]));
        let p = prepare_svg(STAR, Some((160, 160)), &db).unwrap();
        // 80x80 SVG → min-dim 512 → ×2 → 1024×1024 (sizing.rs tests).
        assert_eq!((p.width, p.height), (1024, 1024));
        assert_eq!((p.natural_width, p.natural_height), (80, 80));
        assert_eq!(p.scale_factor, (12.8, 12.8));
        assert!(p.is_svg);
        // Background #fdf6e3 opaque premultiplied = straight.
        assert_eq!(&p.rgba_premul[..4], &[0xfd, 0xf6, 0xe3, 0xff]);
        // Regression sentinel: an opaque fixture must stay non-blank.
        // A false positive here is the failure mode that would make the whole
        // signal untrustworthy, so it is pinned on the existing happy path.
        assert_eq!(p.coverage_permille, 1000);
        assert_eq!(p.blank_cause, "");
    }

    /// The wasm editor lane rasterizes with a deliberately EMPTY
    /// fontdb, so usvg drops the unlayoutable <text> and this SVG rasterizes
    /// FULLY TRANSPARENT while reporting a clean, successful decode. That is the
    /// exact "loaded but paints nothing" case this verdict exists to name.
    #[test]
    fn text_only_svg_with_empty_fontdb_is_blank_and_names_the_cause() {
        let _report = crate::report::tests::serial_globals();
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="80" height="40"><text x="4" y="24">Acme</text></svg>"##;
        let p = prepare_svg(svg, None, &Arc::new(build_fontdb(&[]))).unwrap();
        assert_eq!(p.coverage_permille, 0);
        assert_eq!(p.blank_cause, crate::decode::BLANK_CAUSE_SVG_TEXT_NO_FONTDB);
        // The decode SUCCEEDED — dimensions are real, the pixels just paint
        // nothing. Nothing about the image is a failure.
        assert_eq!((p.natural_width, p.natural_height), (80, 40));
        assert!(p.rgba_premul.iter().all(|b| *b == 0));
    }

    /// A blank SVG with no <text> at all gets the GENERIC cause — the specific
    /// one must not become a catch-all for every transparent rasterization.
    #[test]
    fn blank_textless_svg_reports_the_generic_cause() {
        let _report = crate::report::tests::serial_globals();
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40"><rect width="40" height="40" fill="#336699" fill-opacity="0"/></svg>"##;
        let p = prepare_svg(svg, None, &Arc::new(build_fontdb(&[]))).unwrap();
        assert_eq!(p.coverage_permille, 0);
        assert_eq!(p.blank_cause, crate::decode::BLANK_CAUSE_FULLY_TRANSPARENT);
    }

    /// The engine's fonts as `moda-engine` hands them over: Noto Sans, with the generic sans
    /// family pointed at it (`moda_text::degraded::settle_generic_families`).
    fn engine_like_fontdb() -> Arc<usvg::fontdb::Database> {
        let font = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../moda-engine/tests/fixtures/fonts/noto-sans/NotoSans-Regular.ttf");
        let mut db = build_fontdb(&[font]);
        assert!(!db.is_empty(), "fixture font loads");
        db.set_sans_serif_family("Noto Sans");
        db.set_serif_family("Noto Sans");
        Arc::new(db)
    }

    fn wordmark(family_attr: &str) -> Vec<u8> {
        format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="200" height="60"><text x="100" y="40" text-anchor="middle" font-size="32" {family_attr} fill="#222">WORDMARK</text></svg>"##
        )
        .into_bytes()
    }

    /// An SVG image's `<text>` draws with the engine's fonts instead of vanishing.
    #[test]
    fn svg_text_draws_with_the_engine_fonts() {
        let _report = crate::report::tests::serial_globals();
        let db = engine_like_fontdb();
        let p = prepare_svg(&wordmark(r#"font-family="Noto Sans""#), None, &db).unwrap();
        assert!(p.coverage_permille > 0, "the wordmark paints");
        assert_eq!(p.blank_cause, "");
        // Family names match case-insensitively, as CSS and text nodes do.
        let lower = prepare_svg(&wordmark(r#"font-family="noto sans""#), None, &db).unwrap();
        assert_eq!(lower.rgba_premul, p.rgba_premul);
    }

    /// A family the engine lacks falls back to its sans face (not dropped, not usvg's serif
    /// default) and is reported as a substituted font, like a text node's.
    #[test]
    fn a_missing_svg_font_falls_back_and_is_reported() {
        let _guard = crate::report::tests::serial_globals();
        crate::report::reset();
        let db = engine_like_fontdb();
        let drawn = prepare_svg(
            &wordmark(r#"font-family="Vanished Display, Gone Sans""#),
            None,
            &db,
        )
        .unwrap();
        let noto = prepare_svg(&wordmark(r#"font-family="Noto Sans""#), None, &db).unwrap();
        assert_eq!(
            drawn.rgba_premul, noto.rgba_premul,
            "draws in the fallback face"
        );
        let events = crate::report::drain();
        assert!(
            events.iter().any(|e| matches!(e,
                crate::report::RuntimeEvent::FontSubstituted { requested, drawn, .. }
                    if requested == "Vanished Display, Gone Sans" && drawn == "Noto Sans")),
            "{events:?}"
        );
        // No font-family at all draws too (usvg's default family is not in the engine's set).
        let unnamed = prepare_svg(&wordmark(""), None, &db).unwrap();
        assert_eq!(unnamed.rgba_premul, noto.rgba_premul);
    }

    #[test]
    fn viewbox_only_svg_resolves_dimensions() {
        let _report = crate::report::tests::serial_globals();
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 20"></svg>"##;
        let db = Arc::new(build_fontdb(&[]));
        let p = prepare_svg(svg, None, &db).unwrap();
        assert_eq!((p.natural_width, p.natural_height), (40, 20));
    }
}
