//! Port of `utils/markup/elements/charts/valueAxisBounds.ts`.
//!
//! Explicit value-axis bounds (`y-min` / `y-max`) for every chart type that
//! draws a numeric value axis: bar, line, area, combo and scatter.
//!
//! Two properties the generators rely on:
//!
//!   1. A bound REPLACES the data-derived end of the domain rather than merging
//!      with it, so it both extends and truncates. The only rejections are
//!      bounds that cannot produce a drawable (strictly increasing) domain.
//!   2. A bound survives nicing. `get_display_range_from_ticks` rounds a domain
//!      out to tick boundaries, which would widen an authored `y-min = 3` back
//!      to 0, so callers apply the bounds twice: once to the raw data domain
//!      (before the log-validity decision and the zero-baseline extension) and
//!      once to the niced display range.
//!
//! The TS side logs a warning when it drops an unusable bound; moda-scene has
//! no logger, so this port drops silently — same geometry, which is what the
//! oracle parity test pins.

use super::scales::ValueScale;

#[derive(Clone, Copy, Default)]
pub struct ValueAxisBounds {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl ValueAxisBounds {
    pub fn is_set(&self) -> bool {
        self.min.is_some() || self.max.is_some()
    }
}

/// `resolveValueAxisBounds` — validate authored bounds against the raw data
/// domain. Non-finite values, inverted pairs, and a lone bound sitting past
/// every value in the data are all dropped.
pub fn resolve_value_axis_bounds(
    y_min: Option<f64>,
    y_max: Option<f64>,
    data_domain: (f64, f64),
) -> ValueAxisBounds {
    let min = y_min.filter(|v| v.is_finite());
    let max = y_max.filter(|v| v.is_finite());
    match (min, max) {
        (None, None) => ValueAxisBounds::default(),
        (Some(lo), Some(hi)) => {
            if lo >= hi {
                ValueAxisBounds::default()
            } else {
                ValueAxisBounds {
                    min: Some(lo),
                    max: Some(hi),
                }
            }
        }
        (Some(lo), None) => {
            // A lone `y-min` at or above every value leaves nothing to draw.
            if data_domain.1.is_finite() && lo >= data_domain.1 {
                ValueAxisBounds::default()
            } else {
                ValueAxisBounds {
                    min: Some(lo),
                    max: None,
                }
            }
        }
        (None, Some(hi)) => {
            if data_domain.0.is_finite() && hi <= data_domain.0 {
                ValueAxisBounds::default()
            } else {
                ValueAxisBounds {
                    min: None,
                    max: Some(hi),
                }
            }
        }
    }
}

/// `applyValueAxisBounds` — replace the bounded ends, leave the rest alone.
pub fn apply_value_axis_bounds(domain: (f64, f64), bounds: &ValueAxisBounds) -> (f64, f64) {
    (
        bounds.min.unwrap_or(domain.0),
        bounds.max.unwrap_or(domain.1),
    )
}

/// `clampScaleToPlot` — keep marks inside the plot box once a bound is pinned.
///
/// The value is clamped into the displayed domain BEFORE it reaches the scale,
/// which matters on a log axis: a positive `y-min` can make the domain
/// log-valid while individual rows are still zero or negative, and `ln` of
/// those is NaN. Returns the scale untouched when no bound is set, so unbounded
/// charts keep their exact previous output.
pub fn clamp_scale_to_plot(
    scale: ValueScale,
    range: f64,
    bounds: &ValueAxisBounds,
    domain: (f64, f64),
) -> ValueScale {
    if !bounds.is_set() {
        return scale;
    }
    ValueScale::Clamped {
        inner: Box::new(scale),
        lo: domain.0.min(domain.1),
        hi: domain.0.max(domain.1),
        range,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::charts::scales::{create_linear_scale, create_log_scale};

    #[test]
    fn keeps_usable_bounds_and_drops_undrawable_ones() {
        let b = resolve_value_axis_bounds(Some(0.0), Some(100.0), (10.0, 40.0));
        assert_eq!((b.min, b.max), (Some(0.0), Some(100.0)));

        // Inverted pair: both ends fall back to the data.
        let b = resolve_value_axis_bounds(Some(100.0), Some(10.0), (10.0, 40.0));
        assert_eq!((b.min, b.max), (None, None));

        // Non-finite bounds never reach the domain.
        let b = resolve_value_axis_bounds(Some(f64::INFINITY), None, (10.0, 40.0));
        assert_eq!(b.min, None);

        // A lone bound past every value would leave nothing to draw.
        assert_eq!(
            resolve_value_axis_bounds(Some(500.0), None, (10.0, 40.0)).min,
            None
        );
        assert_eq!(
            resolve_value_axis_bounds(None, Some(5.0), (10.0, 40.0)).max,
            None
        );
        // A lone bound that only extends the axis is kept.
        assert_eq!(
            resolve_value_axis_bounds(None, Some(500.0), (10.0, 40.0)).max,
            Some(500.0)
        );
    }

    #[test]
    fn applies_only_the_pinned_end() {
        let bounds = ValueAxisBounds {
            min: None,
            max: Some(200.0),
        };
        assert_eq!(apply_value_axis_bounds((0.0, 80.0), &bounds), (0.0, 200.0));
    }

    #[test]
    fn clamped_scale_keeps_marks_in_the_plot_box() {
        let bounds = ValueAxisBounds {
            min: Some(50.0),
            max: Some(100.0),
        };
        let scale = clamp_scale_to_plot(
            create_linear_scale((50.0, 100.0), 200.0),
            200.0,
            &bounds,
            (50.0, 100.0),
        );
        assert_eq!(scale.scale(75.0), 100.0);
        // Below the pinned floor / above the ceiling collapse onto the edges
        // instead of drawing outside the plot.
        assert_eq!(scale.scale(10.0), 0.0);
        assert_eq!(scale.scale(400.0), 200.0);
    }

    #[test]
    fn clamped_log_scale_never_emits_nan_for_non_positive_rows() {
        // A positive y-min can make a log domain valid while rows are still
        // zero or negative; `ln` of those is NaN without the input clamp.
        let bounds = ValueAxisBounds {
            min: Some(1.0),
            max: Some(1000.0),
        };
        let scale = clamp_scale_to_plot(
            create_log_scale((1.0, 1000.0), 300.0),
            300.0,
            &bounds,
            (1.0, 1000.0),
        );
        assert_eq!(scale.scale(-5.0), 0.0);
        assert_eq!(scale.scale(0.0), 0.0);
        assert!(scale.scale(10.0).is_finite());
    }

    #[test]
    fn unbounded_scales_pass_through_untouched() {
        let bounds = ValueAxisBounds::default();
        let scale = clamp_scale_to_plot(
            create_linear_scale((0.0, 100.0), 200.0),
            200.0,
            &bounds,
            (0.0, 100.0),
        );
        // No clamp wrapper, so out-of-domain values keep their old placement.
        assert_eq!(scale.scale(150.0), 300.0);
    }
}
