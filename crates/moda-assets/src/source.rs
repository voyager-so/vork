//! Source classification + data-URI decode + vendored-dir resolution.
//!
//! Mirrors `WebGPUTextureManager.loadTextureAsync`'s source split
//! (`WebGPUTextureManager.ts:725-787`): `data:`/relative sources decode
//! directly; http(s) URLs are classified so the offline prefetch can refuse
//! them. The native CLI adds
//! a vendored-assets lane (`--assets-dir`, PLAN §2) because presigned URLs in
//! committed fixtures expire (`consumers-and-integration.md` §4.4).

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine as _;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssetSource {
    /// `data:` URI — bytes inline (all M0b image/svg fixtures use these).
    DataUri,
    /// http(s) URL — never fetched (documents render offline).
    Http,
    /// Anything else — resolved against `--assets-dir` / the filesystem.
    File,
}

pub fn classify(src: &str) -> AssetSource {
    if src.starts_with("data:") {
        AssetSource::DataUri
    } else if src.starts_with("http://") || src.starts_with("https://") {
        AssetSource::Http
    } else {
        AssetSource::File
    }
}

/// `isSVG` clone (`WebGPUTextureManager.ts:1006-1019`): extension or data-URI
/// mime sniff only. http responses additionally sniff the content type after
/// fetch (`peekBlobUrlContentType` parity is approximated by byte sniffing).
pub fn is_likely_svg(src: &str) -> bool {
    let lower = src.to_lowercase();
    lower.ends_with(".svg") || lower.starts_with("data:image/svg+xml")
}

/// Content sniff used for fetched bytes (the production path detects SVG via
/// the blob Content-Type even without an .svg extension —
/// `WebGPUTextureManager.ts:1006-1019` + the blob content-type peek in the
/// fetch path `WebGPUTextureManager.ts:725-787` — and therefore renders SVG
/// documents that open with XML comments/doctype before `<svg>`; real prod
/// team assets do (WF5 v3: `<!--\ntags: [...]-->\n<svg ...>` icon files).
/// Mirror that tolerance: skip BOM/whitespace, XML decl, comments and
/// DOCTYPE, then require the `<svg` root.
pub fn sniff_svg_bytes(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(4096)];
    let text = String::from_utf8_lossy(head);
    let mut t = text.trim_start_matches('\u{feff}').trim_start();
    loop {
        if let Some(rest) = t.strip_prefix("<?") {
            // XML/processing declaration: `<?xml ...?>`
            match rest.find("?>") {
                Some(i) => t = rest[i + 2..].trim_start(),
                None => return false,
            }
        } else if let Some(rest) = t.strip_prefix("<!--") {
            match rest.find("-->") {
                Some(i) => t = rest[i + 3..].trim_start(),
                None => return false,
            }
        } else if t.starts_with("<!") {
            // DOCTYPE (no internal-subset handling needed for real assets)
            match t.find('>') {
                Some(i) => t = t[i + 1..].trim_start(),
                None => return false,
            }
        } else {
            break;
        }
    }
    t.starts_with("<svg")
}

/// Decode a `data:` URI to (mime, bytes). Supports `;base64` and
/// percent-encoded text payloads (the two forms browsers accept).
pub fn decode_data_uri(src: &str) -> Result<(String, Vec<u8>)> {
    let rest = src
        .strip_prefix("data:")
        .ok_or_else(|| anyhow!("not a data URI"))?;
    let (meta, payload) = rest
        .split_once(',')
        .ok_or_else(|| anyhow!("malformed data URI (no comma)"))?;
    let mut mime = "text/plain";
    let mut is_base64 = false;
    for (i, part) in meta.split(';').enumerate() {
        if i == 0 && !part.is_empty() {
            mime = part;
        } else if part.eq_ignore_ascii_case("base64") {
            is_base64 = true;
        }
    }
    let bytes = if is_base64 {
        base64::engine::general_purpose::STANDARD
            .decode(payload.trim())
            .context("data URI base64 decode")?
    } else {
        percent_decode(payload)
    };
    Ok((mime.to_string(), bytes))
}

fn percent_decode(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) =
                u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or(""), 16)
            {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

/// Resolve a non-http, non-data source against the vendored assets directory
/// (when given) or the filesystem. Tries, in order:
///   1. `<assets_dir>/<src>` (relative vendored path, incl. bare sha256 names)
///   2. `<src>` as an absolute/relative filesystem path
pub fn resolve_file(src: &str, assets_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(dir) = assets_dir {
        let p = dir.join(src);
        if p.is_file() {
            return Ok(p);
        }
    }
    let p = PathBuf::from(src);
    if p.is_file() {
        return Ok(p);
    }
    bail!("asset source '{src}' not found (assets_dir={assets_dir:?})")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_sources() {
        assert_eq!(classify("data:image/png;base64,AAAA"), AssetSource::DataUri);
        assert_eq!(classify("https://x.test/a.png"), AssetSource::Http);
        assert_eq!(classify("http://x.test/a.png"), AssetSource::Http);
        assert_eq!(classify("foo/bar.png"), AssetSource::File);
    }

    #[test]
    fn svg_detection_mirrors_production() {
        assert!(is_likely_svg("data:image/svg+xml;base64,xxx"));
        assert!(is_likely_svg("https://cdn.test/icon.SVG"));
        assert!(!is_likely_svg("https://cdn.test/icon.png"));
        // proxy URL without extension: NOT detected from the src alone
        // (production needs the blob content type) — sniffed from bytes.
        assert!(!is_likely_svg("https://app/api/v2/images/ref/abc"));
        assert!(sniff_svg_bytes(b"<svg xmlns='x'></svg>"));
        assert!(sniff_svg_bytes(b"<?xml version=\"1.0\"?>\n<svg></svg>"));
        assert!(!sniff_svg_bytes(&[0x89, b'P', b'N', b'G']));
        // WF5 v3: prod team icon files open with an XML comment before <svg>
        // (browser parity: production routes on Content-Type, not bytes).
        assert!(sniff_svg_bytes(
            b"<!--\ntags: [car, vehicle, torque]\n-->\n<svg xmlns='x'></svg>"
        ));
        assert!(sniff_svg_bytes(
            b"\xef\xbb\xbf<?xml version=\"1.0\"?><!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"x.dtd\"><!-- c --><svg/>"
        ));
        assert!(!sniff_svg_bytes(
            b"<!DOCTYPE html><html><body>error</body></html>"
        ));
    }

    #[test]
    fn data_uri_base64_and_percent() {
        let (mime, bytes) = decode_data_uri("data:image/png;base64,AQID").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(bytes, vec![1, 2, 3]);
        let (mime, bytes) = decode_data_uri("data:image/svg+xml,%3Csvg%3E").unwrap();
        assert_eq!(mime, "image/svg+xml");
        assert_eq!(bytes, b"<svg>");
    }
}
