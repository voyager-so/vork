//! Synthesized small caps: `"smcp" 1` / `"c2sc" 1` on a font that has no such
//! GSUB feature (Inter, Georgia, most system faces).
//!
//! Figma, Illustrator and browsers (CSS `font-variant-caps` synthesis) all
//! fake the feature the same way, and so do we: a letter that would become a
//! small cap is drawn as its capital at [`SYNTHETIC_SMALL_CAPS_SCALE`] of the
//! run's size. With `smcp` that is every lowercase letter (upper-cased), with
//! `c2sc` every capital; everything else (existing capitals under `smcp`
//! alone, digits, punctuation, spaces) stays full size.
//!
//! It is a run transform on the vector lane's resolved runs
//! (`vector::node_layout::resolve_vector_runs`), the one step the draw walk,
//! the layout dump and the SVG/PDF extraction all share: each run splits into
//! sub-runs, the reduced ones at the smaller `font_size_px` with
//! `small_caps_full_size` set to the full size. Shaping, advances and drawing
//! read the reduced size; line boxes, baselines, cap/x heights and underline
//! placement read the full one (`InlineStyle::line_font_size`), so a line of
//! all-lowercase small caps keeps the height and baseline of full-size text.
//! A font that has the feature is left alone and shapes it for real.

use crate::font_features::FontFeature;
use crate::runs::MsdfRun;

/// A synthesized small cap's size as a fraction of the run's font size. 0.7
/// is what Chromium and WebKit use for synthesized `font-variant-caps`
/// (Blink's `kSmallCapsFontSizeMultiplier`), and Figma's synthesis matches it.
pub const SYNTHETIC_SMALL_CAPS_SCALE: f64 = 0.7;

pub const SMCP: [u8; 4] = *b"smcp";
pub const C2SC: [u8; 4] = *b"c2sc";

/// Whether the run's features turn `tag` on (the last setting wins, like CSS).
pub fn feature_on(features: &[FontFeature], tag: &[u8; 4]) -> bool {
    features
        .iter()
        .rev()
        .find(|f| &f.tag == tag)
        .is_some_and(|f| f.value != 0)
}

/// Which features to synthesize for a run, given whether its font has them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Synthesis {
    /// Lowercase → reduced capitals.
    pub smcp: bool,
    /// Capitals → reduced capitals.
    pub c2sc: bool,
}

impl Synthesis {
    pub fn any(self) -> bool {
        self.smcp || self.c2sc
    }
}

/// What a run needs synthesized: a feature it turns on that its font lacks.
/// `has_feature` is asked only for features the run turns on.
pub fn synthesis_for(
    features: &[FontFeature],
    mut has_feature: impl FnMut(&[u8; 4]) -> bool,
) -> Synthesis {
    Synthesis {
        smcp: feature_on(features, &SMCP) && !has_feature(&SMCP),
        c2sc: feature_on(features, &C2SC) && !has_feature(&C2SC),
    }
}

/// The reduced form of `c`, or `None` when it stays full size.
fn reduce(c: char, synth: Synthesis) -> Option<String> {
    if synth.smcp && c.is_lowercase() {
        let upper: String = c.to_uppercase().collect();
        if upper.chars().ne(std::iter::once(c)) {
            return Some(upper);
        }
    }
    if synth.c2sc && c.is_uppercase() {
        return Some(c.to_string());
    }
    None
}

/// Split one text run into full-size and reduced sub-runs (see the module
/// docs). Returns the run unchanged (one element) when nothing reduces. The
/// synthesized features are dropped from every piece: the font has no such
/// lookup, and live-text exporters must not ask a browser to fake them again.
pub fn split_run(run: &MsdfRun, synth: Synthesis) -> Vec<MsdfRun> {
    if !synth.any() {
        return vec![run.clone()];
    }
    // (reduced?, text) groups; a combining mark stays with its base letter.
    let mut groups: Vec<(bool, String, Vec<Option<usize>>)> = Vec::new();
    for (index, c) in run.text.chars().enumerate() {
        let (reduced, text) = match reduce(c, synth) {
            Some(up) => (true, up),
            None if crate::text_case::is_combining_mark(c) => {
                (groups.last().is_some_and(|g| g.0), c.to_string())
            }
            None => (false, c.to_string()),
        };
        match groups.last_mut() {
            Some(g) if g.0 == reduced => {
                g.2.extend(std::iter::repeat_n(run.source.get(index), text.chars().count()));
                g.1.push_str(&text);
            },
            _ => {
                let indices = vec![run.source.get(index); text.chars().count()];
                groups.push((reduced, text, indices));
            },
        }
    }
    let features: Vec<FontFeature> = run
        .font_features
        .iter()
        .filter(|f| !((synth.smcp && f.tag == SMCP) || (synth.c2sc && f.tag == C2SC)))
        .copied()
        .collect();
    let mut out = Vec::with_capacity(groups.len().max(1));
    for (i, (reduced, text, source)) in groups.into_iter().enumerate() {
        let mut r = run.clone();
        r.text = text;
        r.source = crate::types::SourceIndices::new(source);
        r.font_features = features.clone();
        r.paragraph_break_before = run.paragraph_break_before && i == 0;
        if reduced {
            r.font_size_px = run.font_size_px * SYNTHETIC_SMALL_CAPS_SCALE;
            r.small_caps_full_size = Some(run.font_size_px);
        }
        out.push(r);
    }
    if out.is_empty() {
        let mut r = run.clone();
        r.font_features = features;
        out.push(r);
    }
    out
}

/// Apply [`split_run`] across a node's runs. `synthesis` decides per run (it
/// resolves the run's font); `None` when no run changes, so the common path
/// allocates nothing.
pub fn expand_runs(
    runs: &[MsdfRun],
    mut synthesis: impl FnMut(&MsdfRun) -> Synthesis,
) -> Option<Vec<MsdfRun>> {
    let wants =
        |r: &MsdfRun| feature_on(&r.font_features, &SMCP) || feature_on(&r.font_features, &C2SC);
    if !runs.iter().any(wants) {
        return None;
    }
    let mut out = Vec::with_capacity(runs.len());
    let mut changed = false;
    for run in runs {
        let synth = if wants(run) {
            synthesis(run)
        } else {
            Synthesis::default()
        };
        if synth.any() {
            changed = true;
            out.extend(split_run(run, synth));
        } else {
            out.push(run.clone());
        }
    }
    changed.then_some(out)
}

#[cfg(test)]
#[cfg_attr(not(vork_private_tests), allow(dead_code, unused_imports))]
mod tests {
    use super::*;
    use crate::node_layout::RichTextNodeSpec;
    use crate::test_fixtures::store_and_catalog;

    fn feat(tag: &[u8; 4]) -> FontFeature {
        FontFeature {
            tag: *tag,
            value: 1,
        }
    }

    fn pieces(text: &str, synth: Synthesis) -> Vec<(String, bool)> {
        let catalog = crate::test_fixtures::store_and_catalog().1;
        let html =
            format!("<p><span style=\"font-family: Inter; font-size: 40px\">{text}</span></p>");
        let runs = crate::runs::extract_msdf_runs_by_paragraph(&html, &catalog);
        split_run(&runs[0], synth)
            .into_iter()
            .map(|r| (r.text, r.small_caps_full_size.is_some()))
            .collect()
    }

    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn smcp_reduces_lowercase_and_keeps_capitals() {
        let smcp = Synthesis {
            smcp: true,
            c2sc: false,
        };
        assert_eq!(
            pieces("Small Caps 2x", smcp),
            vec![
                ("S".into(), false),
                ("MALL".into(), true),
                (" C".into(), false),
                ("APS".into(), true),
                (" 2".into(), false),
                ("X".into(), true),
            ]
        );
        let both = Synthesis {
            smcp: true,
            c2sc: true,
        };
        assert_eq!(
            pieces("Ab 1", both),
            vec![("AB".into(), true), (" 1".into(), false)]
        );
        // c2sc alone: capitals shrink, lowercase stays as typed.
        let c2sc = Synthesis {
            smcp: false,
            c2sc: true,
        };
        assert_eq!(
            pieces("Ab", c2sc),
            vec![("A".into(), true), ("b".into(), false)]
        );
    }

    #[test]
    fn synthesis_only_for_features_the_font_lacks() {
        let on = [feat(&SMCP), feat(&C2SC)];
        assert_eq!(
            synthesis_for(&on, |_| false),
            Synthesis {
                smcp: true,
                c2sc: true
            }
        );
        assert_eq!(
            synthesis_for(&on, |t| t == &SMCP),
            Synthesis {
                smcp: false,
                c2sc: true
            }
        );
        let off = [FontFeature {
            tag: SMCP,
            value: 0,
        }];
        assert!(!synthesis_for(&off, |_| false).any());
        assert!(!synthesis_for(&[], |_| false).any());
    }

    fn spec(html: &str) -> RichTextNodeSpec {
        RichTextNodeSpec {
            node_id: "t".into(),
            html_content: html.into(),
            width: Some(1000.0),
            height: Some(400.0),
            ..Default::default()
        }
    }

    /// Inter has no `smcp`: "Small Caps" with small caps on lays out the
    /// lowercase letters as Inter's CAPITAL glyphs at 70% of the size, the
    /// real capitals unchanged; the line keeps the full-size baseline.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn inter_small_caps_are_synthesized_as_reduced_capitals() {
        let (store, catalog) = store_and_catalog();
        let plain = "<p><span style=\"font-family: Inter; font-size: 40px\">Small Caps</span></p>";
        let caps = "<p><span style=\"font-family: Inter; font-size: 40px; font-feature-settings: &quot;smcp&quot; 1\">Small Caps</span></p>";
        let upper = "<p><span style=\"font-family: Inter; font-size: 40px\">SMALL CAPS</span></p>";
        let extract = |html: &str| {
            crate::vector::extract::extract_vector_text_runs(&spec(html), &store, &catalog)
                .expect("vector lane")
        };
        let plain_runs = extract(plain);
        let caps_runs = extract(caps);
        let upper_runs = extract(upper);
        let glyphs = |runs: &[crate::vector::extract::ExtractedRun]| {
            runs.iter()
                .flat_map(|r| {
                    r.glyphs
                        .iter()
                        .map(move |g| (g.glyph_id, r.font_size_px, g.x_px))
                })
                .collect::<Vec<_>>()
        };
        let (pg, cg, ug) = (glyphs(&plain_runs), glyphs(&caps_runs), glyphs(&upper_runs));
        // Nine glyphs: the space draws nothing.
        assert_eq!((cg.len(), ug.len(), pg.len()), (9, 9, 9));
        // Glyph ids are the capitals' ("SMALL CAPS"), not the lowercase ones.
        let ids = |g: &[(u32, f64, f64)]| g.iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(ids(&cg), ids(&ug));
        assert_ne!(ids(&cg), ids(&pg));
        // S and C full size, the rest at 28px.
        let sizes: Vec<f64> = cg.iter().map(|x| x.1).collect();
        for (i, s) in sizes.iter().enumerate() {
            let want = if i == 0 || i == 5 {
                40.0
            } else {
                40.0 * SYNTHETIC_SMALL_CAPS_SCALE
            };
            assert!(
                (s - want).abs() < 1e-9,
                "glyph {i}: {s} vs {want} ({sizes:?})"
            );
        }
        // Reduced capitals advance less than full ones: "SMALL " is narrower in
        // small caps than in capitals, but wider than lowercase "small ".
        let width = |g: &[(u32, f64, f64)]| g[5].2 - g[0].2;
        assert!(width(&cg) < width(&ug), "{} < {}", width(&cg), width(&ug));
        assert!(width(&cg) > width(&pg) * 0.9);
        // Same baseline as the plain line: line metrics use the full size.
        let base = |runs: &[crate::vector::extract::ExtractedRun]| runs[0].baseline_y_px;
        assert!(caps_runs
            .iter()
            .all(|r| (r.baseline_y_px - base(&plain_runs)).abs() < 1e-9));
        // The synthesized feature is not passed on to live-text exporters.
        assert!(caps_runs.iter().all(|r| !r.font_features.contains("smcp")));
    }

    /// An all-lowercase small-caps line keeps the full-size line box.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn all_reduced_line_keeps_full_size_baseline() {
        let (store, catalog) = store_and_catalog();
        let html = |feat: &str| {
            format!("<p><span style=\"font-family: Inter; font-size: 40px{feat}\">caps</span></p>")
        };
        let run = |h: String| {
            crate::vector::extract::extract_vector_text_runs(&spec(&h), &store, &catalog)
                .expect("vector lane")
                .remove(0)
        };
        let plain = run(html(""));
        let caps = run(html("; font-feature-settings: &quot;smcp&quot; 1"));
        assert_eq!(caps.text, "CAPS");
        assert!((caps.font_size_px - 28.0).abs() < 1e-9);
        assert!((caps.baseline_y_px - plain.baseline_y_px).abs() < 1e-9);
    }

    /// Montserrat has a real `smcp`: no synthesis, the font's own small caps.
    #[test]
    #[cfg(vork_private_tests)] // Reads the private world-type fonts.
    fn a_font_with_smcp_shapes_it_for_real() {
        let (store, catalog) = store_and_catalog();
        let html = "<p><span style=\"font-family: Montserrat; font-size: 40px; font-feature-settings: &quot;smcp&quot; 1\">Caps</span></p>";
        let runs = crate::vector::extract::extract_vector_text_runs(&spec(html), &store, &catalog)
            .expect("vector lane");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].text, "Caps");
        assert_eq!(runs[0].font_size_px, 40.0);
        assert!(runs[0].font_features.contains("smcp"));
    }
}
