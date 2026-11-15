//! The measurer seam the layout walks measure through (`TextMeasurer`); the vector lane implements it
//! (`vector::measure::VectorMeasurer`) with shaped advances.

use crate::types::InlineStyle;

/// `TextMeasurer` (`utils/richtext/layout.ts:4`).
pub trait TextMeasurer {
    fn measure(&self, text: &str, style: &InlineStyle) -> f64;

    /// Figma's Auto line height for `style` in px: the font's ascender − descender + line gap at the line's font
    /// size (`InlineStyle::line_height_auto`). `None` without the font's metrics: the style's multiple applies.
    fn auto_line_height_px(&self, _style: &InlineStyle) -> Option<f64> {
        None
    }
}

/// A measurer whose Auto line height is the style's multiple (120 %): area type's Auto leading, as in Illustrator.
pub struct MultipleLeading<'a, M: TextMeasurer + ?Sized>(pub &'a M);

impl<M: TextMeasurer + ?Sized> TextMeasurer for MultipleLeading<'_, M> {
    fn measure(&self, text: &str, style: &InlineStyle) -> f64 {
        self.0.measure(text, style)
    }
}

/// Auto line height from a font's chosen baseline metrics (`crate::ttf::TtfMetrics`): Figma measures (ascender −
/// descender + line gap) / units per em × size — Arial 100 px is 115 (its hhea line gap counts), Inter 121.
pub fn auto_line_height_from_metrics(m: &crate::ttf::TtfMetrics, font_size_px: f64) -> Option<f64> {
    if m.units_per_em <= 0.0 {
        return None;
    }
    let em = m.ascent + m.descent.abs() + m.line_gap.max(0.0);
    let px = em / m.units_per_em * font_size_px;
    (px.is_finite() && px > 0.0).then_some(px)
}
