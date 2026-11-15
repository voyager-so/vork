//! Word breaks inside Thai, Lao, Khmer and Myanmar text: these scripts put no spaces between words, so a line may
//! break only where a dictionary finds a word boundary. Without this the complex-script tier's wrap saw a whole
//! sentence as one word and broke it per character, mid-word ("ร้า|น").
//!
//! The dictionaries are ICU's (`icu_segmenter`, the same data the browser's `Intl.Segmenter` and line breaker use);
//! the TS twin is `segmentSeaWords` (`renderers/webgpu/text/vector/cjkLineBreaking.ts`). Only boundaries with a SEA
//! letter on both sides are break opportunities: punctuation, digits and Latin around a SEA run stay with it, as the
//! wrap kept them before. A boundary inside a cluster is dropped: ICU4X's dictionary returns some (Khmer
//! "អាចបង់ប្រាក់" as "អាចប|ង់ប|្|រាក់"), and each piece is shaped on its own, so a cut before a mark or after a
//! coeng drew the subscript as a dotted circle and a full letter.

use std::ops::Range;

use icu_provider::prelude::*;
use icu_segmenter::options::WordBreakOptions;
use icu_segmenter::provider::{
    Baked, SegmenterBreakGraphemeClusterV1, SegmenterBreakWordOverrideV1, SegmenterBreakWordV1,
    SegmenterDictionaryAutoV1, SegmenterDictionaryExtendedV1,
};
use icu_segmenter::WordSegmenter;
use unicode_segmentation::UnicodeSegmentation;

use crate::paragraph_format::LineBreaking;

/// Codepoints segmented by dictionary (`SEA_SEGMENTATION_RANGES`, complexScripts.ts).
pub fn is_sea_script_codepoint(cp: u32) -> bool {
    matches!(cp,
        0x0E00..=0x0E7F // Thai
        | 0x0E80..=0x0EFF // Lao
        | 0x1000..=0x109F // Myanmar
        | 0x1780..=0x17FF // Khmer
        | 0xAA60..=0xAA7F // Myanmar Extended-A
    )
}

/// The baked segmenter data without the Chinese/Japanese dictionary (CJK wraps per character here, by its own rules):
/// leaving it out keeps about 2 MB of dictionary out of the binaries.
struct SeaData;

macro_rules! baked {
    ($($marker:ty),*) => {$(
        impl DataProvider<$marker> for SeaData {
            fn load(&self, req: DataRequest) -> Result<DataResponse<$marker>, DataError> {
                Baked.load(req)
            }
        }
    )*};
}
baked!(
    SegmenterBreakWordV1,
    SegmenterBreakWordOverrideV1,
    SegmenterDictionaryExtendedV1,
    SegmenterBreakGraphemeClusterV1
);

impl DataProvider<SegmenterDictionaryAutoV1> for SeaData {
    fn load(&self, req: DataRequest) -> Result<DataResponse<SegmenterDictionaryAutoV1>, DataError> {
        Err(DataErrorKind::IdentifierNotFound.with_req(SegmenterDictionaryAutoV1::INFO, req))
    }
}

thread_local! {
    static SEGMENTER: Option<WordSegmenter> =
        WordSegmenter::try_new_dictionary_unstable(&SeaData, WordBreakOptions::default()).ok();
}

/// Byte ranges of the SEA runs in `text`.
fn sea_runs(text: &str) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for (i, ch) in text.char_indices() {
        if !is_sea_script_codepoint(ch as u32) {
            continue;
        }
        match runs.last_mut() {
            Some(run) if run.end == i => run.end = i + ch.len_utf8(),
            _ => runs.push(i..i + ch.len_utf8()),
        }
    }
    runs
}

/// Whether a piece may start at byte `at` of `text`: a grapheme cluster boundary (no cut before a vowel sign or other
/// mark) that doesn't follow a sign stacking the next consonant under it (Khmer coeng U+17D2, Myanmar virama U+1039),
/// which grapheme clusters split from that consonant.
fn is_cluster_boundary(text: &str, at: usize, graphemes: &[usize]) -> bool {
    graphemes.binary_search(&at).is_ok() && !matches!(text[..at].chars().next_back(), Some('\u{17D2}' | '\u{1039}'))
}

/// A [`crate::wrap::WordSegmenter`]: `word` (no whitespace) cut at the dictionary's word boundaries inside its SEA
/// runs; the pieces concatenate back to `word`. A word without SEA letters comes back whole.
pub fn segment_sea_words(word: &str, _rules: LineBreaking) -> Vec<String> {
    let runs = sea_runs(word);
    if runs.is_empty() {
        return vec![word.to_string()];
    }
    let mut cuts: Vec<usize> = Vec::new();
    SEGMENTER.with(|segmenter| {
        let Some(segmenter) = segmenter else {
            return;
        };
        for run in &runs {
            let text = &word[run.clone()];
            let graphemes: Vec<usize> = text.grapheme_indices(true).map(|(i, _)| i).collect();
            cuts.extend(
                segmenter
                    .as_borrowed()
                    .segment_str(text)
                    .filter(|&b| b > 0 && b < text.len() && is_cluster_boundary(text, b, &graphemes))
                    .map(|b| run.start + b),
            );
        }
    });
    let mut pieces = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for cut in cuts {
        pieces.push(word[start..cut].to_string());
        start = cut;
    }
    pieces.push(word[start..].to_string());
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(text: &str) -> String {
        segment_sea_words(text, LineBreaking::DEFAULT).join("|")
    }

    #[test]
    fn thai_lao_khmer_and_myanmar_break_between_words() {
        assert_eq!(seg("มาพร้อมดนตรีสดและกิจกรรมสำหรับครอบครัว"), "มา|พร้อม|ดนตรี|สด|และ|กิจกรรม|สำหรับ|ครอบครัว");
        assert_eq!(seg("ខ្ញុំស្រលាញ់អ្នក"), "ខ្ញុំ|ស្រលាញ់|អ្នក");
        assert_eq!(seg("မြန်မာစာ"), "မြန်မာ|စာ");
    }

    #[test]
    fn no_piece_starts_inside_a_cluster() {
        // ICU4X cuts the first one before the coeng (and so before the ro it stacks): "អាចប|ង់ប|្|រាក់".
        for word in ["អាចបង់ប្រាក់", "យើងទទួលយកធានារ៉ាប់រង", "សេវាកម្មសុខភាព", "ကျောင်းသားများ"] {
            let pieces = segment_sea_words(word, LineBreaking::DEFAULT);
            assert_eq!(pieces.concat(), word);
            let graphemes: Vec<usize> = word.grapheme_indices(true).map(|(i, _)| i).collect();
            let mut at = 0;
            for pair in pieces.windows(2) {
                at += pair[0].len();
                assert!(graphemes.contains(&at), "cut inside a cluster: {pieces:?}");
                assert!(!pair[0].ends_with(['\u{17D2}', '\u{1039}']), "cut after a stacking sign: {pieces:?}");
            }
        }
        assert!(segment_sea_words("អាចបង់ប្រាក់", LineBreaking::DEFAULT).iter().any(|p| p.contains("ប្រ")));
    }

    #[test]
    fn punctuation_digits_and_latin_stay_with_the_run() {
        // ฯ is Thai; the comma, the digits and the Latin around the run never start a piece of their own.
        assert_eq!(seg("กรุงเทพฯ,"), "กรุงเทพฯ,");
        assert_eq!(seg("(ผัดไทย)"), "(ผัด|ไทย)");
        assert_eq!(seg("2026ผัดไทย"), "2026ผัด|ไทย");
    }

    #[test]
    fn other_words_come_back_whole() {
        for word in ["state-of-the-art", "日本語", "مرحبا", "नमस्ते", ""] {
            assert_eq!(segment_sea_words(word, LineBreaking::DEFAULT), vec![word.to_string()]);
        }
    }
}
