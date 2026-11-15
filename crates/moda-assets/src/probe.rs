//! usvg per-asset manifest probe (PLAN M3c / §3 manifest "warnings" axis).
//!
//! A pure CPU walk over SVG bytes answering "can resvg render this at
//! parity?" for the feature manifest. The upstream sanitizer
//! (`lib/images/AuthImageMiddleware.ts:795-816`) already strips scripts /
//! event handlers / javascript: hrefs / foreignObject from everything the
//! browser rendered, so the practical subset is small; the probe flags the
//! residual fidelity risks.

use resvg::usvg;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct SvgProbe {
    /// usvg parsed the document (parse failure ⇒ the browser likely showed
    /// the failed placeholder too, but flag it for a human).
    pub ok: bool,
    pub width: f64,
    pub height: f64,
    /// Document contains <text> — fidelity depends on fontdb seeding.
    pub has_text: bool,
    /// Document references raster images (<image>).
    pub has_image: bool,
    /// Manifest warning strings (axis `svg:*`).
    pub warnings: Vec<String>,
}

/// Whether the RAW markup contains a `<text` element, independent of what usvg
/// kept in the parsed tree.
///
/// The distinction is the whole point: usvg DROPS `<text>` it cannot lay out, so
/// a tree walk reports "no text" for exactly the documents whose text silently
/// vanished. Shared with [`crate::svg::prepare_svg`], which uses it to explain a
/// zero-coverage rasterization — one definition, so the two answers
/// cannot disagree.
///
/// Bounded to the first 256 KiB and case-insensitive; a `<text` beyond that in a
/// quarter-megabyte of markup is not the actionable case either report is after.
pub fn raw_markup_has_text(bytes: &[u8]) -> bool {
    String::from_utf8_lossy(&bytes[..bytes.len().min(256 * 1024)])
        .to_lowercase()
        .contains("<text")
}

pub fn probe_svg(bytes: &[u8], fontdb: &Arc<usvg::fontdb::Database>) -> SvgProbe {
    let tree = match usvg::Tree::from_data(bytes, &crate::svg::usvg_options(fontdb)) {
        Ok(t) => t,
        Err(e) => {
            return SvgProbe {
                ok: false,
                width: 0.0,
                height: 0.0,
                has_text: false,
                has_image: false,
                warnings: vec![format!("svg:parse-error ({e})")],
            }
        }
    };

    let mut probe = SvgProbe {
        ok: true,
        width: tree.size().width() as f64,
        height: tree.size().height() as f64,
        has_text: false,
        has_image: false,
        warnings: Vec::new(),
    };
    walk(tree.root(), &mut probe);
    // usvg drops <text> nodes it cannot lay out (e.g. an unseeded fontdb),
    // so ALSO scan the raw markup — an invisible-by-fontdb text element is
    // exactly the fidelity risk this probe exists to flag.
    if !probe.has_text {
        probe.has_text = raw_markup_has_text(bytes);
    }

    if probe.has_text {
        probe
            .warnings
            .push("svg:text (fidelity depends on catalog-seeded fontdb)".to_string());
    }
    if probe.has_image {
        probe
            .warnings
            .push("svg:embedded-image (raster sub-image)".to_string());
    }
    // Raw-markup scan for the constructs the production sanitizer strips —
    // their presence means the golden was rendered from DIFFERENT (sanitized)
    // markup than the raw bytes we were handed.
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(64 * 1024)]).to_lowercase();
    if head.contains("<script") || head.contains("<foreignobject") {
        probe
            .warnings
            .push("svg:sanitized-content (script/foreignObject present in raw bytes)".to_string());
    }
    probe
}

fn walk(group: &usvg::Group, probe: &mut SvgProbe) {
    for child in group.children() {
        match child {
            usvg::Node::Group(g) => walk(g, probe),
            usvg::Node::Text(_) => probe.has_text = true,
            usvg::Node::Image(_) => probe.has_image = true,
            usvg::Node::Path(_) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svg::build_fontdb;

    #[test]
    fn clean_svg_probes_ok() {
        let db = Arc::new(build_fontdb(&[]));
        let p = probe_svg(
            br##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>"##,
            &db,
        );
        assert!(p.ok);
        assert!(p.warnings.is_empty());
        assert_eq!((p.width, p.height), (10.0, 10.0));
    }

    #[test]
    fn text_svg_warns() {
        let db = Arc::new(build_fontdb(&[]));
        let p = probe_svg(
            br##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><text x="0" y="8">hi</text></svg>"##,
            &db,
        );
        assert!(p.ok);
        assert!(p.has_text);
        assert!(p.warnings.iter().any(|w| w.starts_with("svg:text")));
    }
}
