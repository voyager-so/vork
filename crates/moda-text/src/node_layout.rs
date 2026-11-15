//! `RichTextNodeSpec` — the rich-text node fields the layout walks consume (the vector lane's
//! `dump_rich_text_node_vector`, text extraction and text geometry).

/// Rich-text node fields the walk consumes, with the TS coercions applied by
/// the CALLER's accessor choice preserved here:
/// `width ?? 400` / `height ?? 200` are nullish (dump-linebox.mjs:156-157),
/// `paragraphSpacing || 0` is falsy (158),
/// `verticalAlign || 'top'` is falsy (98).
#[derive(Debug, Clone, Default)]
pub struct RichTextNodeSpec {
    pub node_id: String,
    pub html_content: String,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub paragraph_spacing: f64,
    /// Internal line-box convention; does not change shaping or baseline metrics.
    pub line_height_policy: crate::types::LineHeightPolicy,
    pub vertical_align: Option<String>,
    /// E6.3 "Truncate text" (`crate::truncation`); `None` when off. Honoured by
    /// the vector extraction (`vector::extract`), not by the oracle dumps.
    pub truncation: Option<crate::truncation::TextTruncation>,
    /// E7.20 auto-width text: no wrapping at the box edge (see
    /// `TextNodeSpec::auto_width`).
    pub auto_width: bool,
    /// Area type (`crate::area`): the text fits its shape, thread and wrap objects instead of wrapping at the box
    /// width. `None` for box text.
    pub area: Option<crate::area::AreaSpec>,
}

impl RichTextNodeSpec {
    /// The width lines wrap at: `width ?? 400`, or unbounded for auto-width text.
    pub fn wrap_width(&self) -> f64 {
        if self.auto_width {
            f64::INFINITY
        } else {
            self.width.unwrap_or(400.0).max(0.0)
        }
    }

    /// Vertical text: the length lines wrap at, `height ?? 200`, or unbounded for auto-width text.
    pub fn inline_wrap_width(&self) -> f64 {
        if self.auto_width {
            f64::INFINITY
        } else {
            self.height.unwrap_or(200.0).max(0.0)
        }
    }
}
