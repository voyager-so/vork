//! Half-leading baselines — port of `utils/richtext/lineMetrics.ts`.

/// `DEFAULT_ASCENT_FALLBACK_RATIO` (`fontDefaults.ts:18`).
pub const DEFAULT_ASCENT_FALLBACK_RATIO: f64 = 0.8;
/// `DEFAULT_DESCENT_FALLBACK_RATIO` (`fontDefaults.ts:24`).
pub const DEFAULT_DESCENT_FALLBACK_RATIO: f64 = 0.2;

use crate::types::{InlineStyle, Line};

/// `FontMetricsProvider` (`utils/richtext/types.ts:14`).
pub trait FontMetricsProvider {
    fn metrics(&self, style: &InlineStyle) -> (f64, f64); // (ascent, descent)
    /// (cap height, x height) in px — area type's first-baseline options (`crate::area`). Without the font's own
    /// values: 0.7 and 0.5 em.
    fn cap_x_heights(&self, style: &InlineStyle) -> (f64, f64) {
        (style.font_size * 0.7, style.font_size * 0.5)
    }
}

impl<P: FontMetricsProvider + ?Sized> FontMetricsProvider for &P {
    fn metrics(&self, style: &InlineStyle) -> (f64, f64) {
        (**self).metrics(style)
    }
    fn cap_x_heights(&self, style: &InlineStyle) -> (f64, f64) {
        (**self).cap_x_heights(style)
    }
}

pub struct BaselineResult {
    pub baseline_from_top_px: f64,
    pub ascent: f64,
    pub descent: f64,
}

/// `computeBaselineForLine` (`lineMetrics.ts:8-24`) +
/// `computeBaselineForLineWithFallback` (`lineMetrics.ts:33-45`): half-leading
/// baseline (negative leading allowed), 0.8/0.2-em fallback when the provider
/// has no metrics for any segment.
pub fn compute_baseline_for_line_with_fallback<P: FontMetricsProvider>(
    line: &Line,
    line_height_px: f64,
    fallback_font_size: f64,
    provider: &P,
) -> BaselineResult {
    let mut max_ascent = 0.0f64;
    let mut max_descent = 0.0f64;
    for seg in &line.segments {
        let (a, d) = provider.metrics(&seg.style);
        if a > max_ascent {
            max_ascent = a;
        }
        if d > max_descent {
            max_descent = d;
        }
    }
    let content_box = max_ascent + max_descent;
    let leading = line_height_px - content_box;
    let baseline_y = leading / 2.0 + max_ascent;

    let ascent = if max_ascent > 0.0 {
        max_ascent
    } else {
        fallback_font_size * DEFAULT_ASCENT_FALLBACK_RATIO
    };
    let descent = if max_descent > 0.0 {
        max_descent
    } else {
        fallback_font_size * DEFAULT_DESCENT_FALLBACK_RATIO
    };
    let baseline_from_top_px = if max_ascent > 0.0 && max_descent > 0.0 {
        baseline_y
    } else {
        (line_height_px - (ascent + descent)) / 2.0 + ascent
    };
    BaselineResult {
        baseline_from_top_px,
        ascent,
        descent,
    }
}
