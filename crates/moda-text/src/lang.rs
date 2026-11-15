//! A text run's language (`--nf-lang`, or an element's `lang` attribute; the document's `locale` for text that sets
//! none) and what it decides: which regional face draws Han characters. Mirrored by `utils/richtext/lang.ts` (same
//! parse, same canonical form, same chains).
//!
//! Han characters are shared by Chinese, Japanese and Korean, but each region draws many of them differently (直, 骨,
//! 写, 誤 …), so a CJK fallback picks the face of the text's region: `ja` → Noto Sans JP, `ko` → Noto Sans KR,
//! `zh` / `zh-Hans` / `zh-CN` / `zh-SG` → Noto Sans SC, `zh-Hant` / `zh-TW` / `zh-HK` / `zh-MO` → Noto Sans TC. Kana
//! still prefer the Japanese face and Hangul the Korean one whatever the language; text with no language keeps the
//! order every document had before (JP, SC, TC, KR). On a serif primary face the region's Noto Serif face comes first.

/// The CSS custom property a run's language is stored in (inherited, on a `<p>` or a `<span>`); a paragraph's also picks
/// its hyphenation patterns (`paragraph_format`).
pub use crate::paragraph_format::LANG_PROPERTY;

/// A BCP 47 tag in canonical case (`ja`, `zh-Hant`, `zh-TW`, `en-US`): a 2–3 letter language, then optionally a
/// 4-letter script and a 2-letter or 3-digit region. `None` when it doesn't read as one (`""`, `auto`, `japanese`).
pub fn canonical_lang(value: &str) -> Option<String> {
    let v = value.trim();
    let mut parts = v.split(['-', '_']);
    let lang = parts.next()?;
    if !(2..=3).contains(&lang.len()) || !lang.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut out = lang.to_ascii_lowercase();
    let mut script_seen = false;
    let mut region_seen = false;
    for p in parts {
        if !script_seen && !region_seen && p.len() == 4 && p.chars().all(|c| c.is_ascii_alphabetic()) {
            let mut s = p.to_ascii_lowercase();
            s[..1].make_ascii_uppercase();
            out.push('-');
            out.push_str(&s);
            script_seen = true;
        } else if !region_seen
            && ((p.len() == 2 && p.chars().all(|c| c.is_ascii_alphabetic()))
                || (p.len() == 3 && p.chars().all(|c| c.is_ascii_digit())))
        {
            out.push('-');
            out.push_str(&p.to_ascii_uppercase());
            region_seen = true;
        } else {
            return None;
        }
    }
    Some(out)
}

/// The CJK region a language draws Han characters in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CjkRegion {
    Japanese,
    Korean,
    SimplifiedChinese,
    TraditionalChinese,
}

impl CjkRegion {
    /// The region of a canonical tag, `None` for a language that isn't Chinese, Japanese or Korean.
    pub fn of(lang: &str) -> Option<CjkRegion> {
        let mut parts = lang.split('-');
        match parts.next()? {
            "ja" => Some(CjkRegion::Japanese),
            "ko" => Some(CjkRegion::Korean),
            base @ ("zh" | "cmn" | "yue") => {
                // The script decides; else the region; else the language (Cantonese is written in traditional
                // characters, Mandarin and plain `zh` in simplified).
                let subtags: Vec<&str> = parts.collect();
                let traditional = if subtags.contains(&"Hant") {
                    true
                } else if subtags.contains(&"Hans") {
                    false
                } else if subtags.iter().any(|t| matches!(*t, "TW" | "HK" | "MO")) {
                    true
                } else if subtags.iter().any(|t| matches!(*t, "CN" | "SG")) {
                    false
                } else {
                    base == "yue"
                };
                Some(if traditional {
                    CjkRegion::TraditionalChinese
                } else {
                    CjkRegion::SimplifiedChinese
                })
            }
            _ => None,
        }
    }

    /// The region's Noto Sans face.
    pub fn sans_family(self) -> &'static str {
        match self {
            CjkRegion::Japanese => "Noto Sans JP",
            CjkRegion::Korean => "Noto Sans KR",
            CjkRegion::SimplifiedChinese => "Noto Sans SC",
            CjkRegion::TraditionalChinese => "Noto Sans TC",
        }
    }

    /// The region's Noto Serif face.
    pub fn serif_family(self) -> &'static str {
        match self {
            CjkRegion::Japanese => "Noto Serif JP",
            CjkRegion::Korean => "Noto Serif KR",
            CjkRegion::SimplifiedChinese => "Noto Serif SC",
            CjkRegion::TraditionalChinese => "Noto Serif TC",
        }
    }

    /// The region's short name in messages and install commands.
    pub fn code(self) -> &'static str {
        match self {
            CjkRegion::Japanese => "ja",
            CjkRegion::Korean => "ko",
            CjkRegion::SimplifiedChinese => "zh-Hans",
            CjkRegion::TraditionalChinese => "zh-Hant",
        }
    }
}

/// The Han fallback chain for text in `region` (`None`: no language, the order before languages): the region's face
/// first, then the others in the default order.
pub fn han_fallback_families(region: Option<CjkRegion>) -> [&'static str; 4] {
    const DEFAULT: [&str; 4] = ["Noto Sans JP", "Noto Sans SC", "Noto Sans TC", "Noto Sans KR"];
    let Some(region) = region else {
        return DEFAULT;
    };
    let first = region.sans_family();
    let mut out = [first; 4];
    let mut i = 1;
    for f in DEFAULT {
        if f != first {
            out[i] = f;
            i += 1;
        }
    }
    out
}

/// `html` with the document's `locale` as the language of text that sets none: wrapped in a `<div>` that declares
/// it, which every paragraph inherits (a run's own `--nf-lang` or `lang` still wins). Unchanged without a locale.
pub fn html_with_document_lang(html: &str, locale: Option<&str>) -> String {
    match locale.and_then(canonical_lang) {
        Some(lang) if !html.is_empty() => format!("<div style=\"{LANG_PROPERTY}: {lang}\">{html}</div>"),
        _ => html.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_read_in_canonical_case() {
        assert_eq!(canonical_lang("JA").as_deref(), Some("ja"));
        assert_eq!(canonical_lang("zh-hant").as_deref(), Some("zh-Hant"));
        assert_eq!(canonical_lang("zh_tw").as_deref(), Some("zh-TW"));
        assert_eq!(canonical_lang("zh-Hant-HK").as_deref(), Some("zh-Hant-HK"));
        assert_eq!(canonical_lang("es-419").as_deref(), Some("es-419"));
        assert_eq!(canonical_lang(""), None);
        assert_eq!(canonical_lang("japanese"), None);
        assert_eq!(canonical_lang("ja-JP-x"), None);
    }

    #[test]
    fn regions_by_language() {
        assert_eq!(CjkRegion::of("ja"), Some(CjkRegion::Japanese));
        assert_eq!(CjkRegion::of("ko-KR"), Some(CjkRegion::Korean));
        assert_eq!(CjkRegion::of("zh"), Some(CjkRegion::SimplifiedChinese));
        assert_eq!(CjkRegion::of("zh-CN"), Some(CjkRegion::SimplifiedChinese));
        assert_eq!(CjkRegion::of("zh-Hant"), Some(CjkRegion::TraditionalChinese));
        assert_eq!(CjkRegion::of("zh-TW"), Some(CjkRegion::TraditionalChinese));
        assert_eq!(CjkRegion::of("zh-HK"), Some(CjkRegion::TraditionalChinese));
        assert_eq!(CjkRegion::of("zh-Hans-HK"), Some(CjkRegion::SimplifiedChinese));
        assert_eq!(CjkRegion::of("yue"), Some(CjkRegion::TraditionalChinese));
        assert_eq!(CjkRegion::of("en"), None);
    }

    #[test]
    fn han_chains_put_the_region_first() {
        assert_eq!(han_fallback_families(None), ["Noto Sans JP", "Noto Sans SC", "Noto Sans TC", "Noto Sans KR"]);
        assert_eq!(
            han_fallback_families(Some(CjkRegion::Korean)),
            ["Noto Sans KR", "Noto Sans JP", "Noto Sans SC", "Noto Sans TC"]
        );
        assert_eq!(
            han_fallback_families(Some(CjkRegion::TraditionalChinese)),
            ["Noto Sans TC", "Noto Sans JP", "Noto Sans SC", "Noto Sans KR"]
        );
    }

    #[test]
    fn document_lang_wraps_the_html() {
        assert_eq!(html_with_document_lang("<p>a</p>", None), "<p>a</p>");
        assert_eq!(html_with_document_lang("<p>a</p>", Some("bogus tag")), "<p>a</p>");
        assert_eq!(
            html_with_document_lang("<p>a</p>", Some("JA")),
            "<div style=\"--nf-lang: ja\"><p>a</p></div>"
        );
    }
}
