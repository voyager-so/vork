//! Port of `utils/markup/elements/charts/scales.ts` plus the d3-scale
//! internals it delegates to (`d3-scale/src/{band,log,continuous,time}.js`).
//! The UTC time scale (`scaleUtc`) delegates its nice/ticks to the d3-time
//! port in `timeticks.rs`.

use super::ticks::TickError;
use super::jsmath::{js_max, js_min};
use super::ticks::{d3_nice, d3_ticks};
use super::timeticks::{ms_in_date_range, time_nice, utc_ticks};
use std::collections::HashMap;

/// `LinearScale` contract: pixel offsets in [0, range], domain-min → 0.
/// Zero-span domains collapse to 0 (scales.ts:122-140).
#[derive(Clone, Debug)]
pub enum ValueScale {
    Linear {
        min: f64,
        max: f64,
        range: f64,
    },
    /// Base-10 log (scales.ts:160-196). Mapping is d3 continuous with the
    /// natural-log transform; ticks are the d3 loggish algorithm.
    Log {
        min: f64,
        max: f64,
        range: f64,
    },
    /// UTC time scale (`createTimeScale`, scales.ts:226-264). Mapping is the
    /// plain d3-continuous (linear-in-ms) interpolation; only the ticks differ
    /// — they come from the d3-time interval ladder (`timeticks::utc_ticks`).
    /// `min`/`max` are ms-since-epoch (integral, well under 2^53). Zero-span
    /// collapses to 0 like `Linear`.
    Time {
        min: f64,
        max: f64,
        range: f64,
    },
    /// `clampScaleToPlot` (valueAxisBounds.ts): an inner scale whose input is
    /// clamped into `[lo, hi]` (the displayed domain) and whose pixel output is
    /// clamped into `[0, range]`. Only built when the markup pins an explicit
    /// `y-min`/`y-max`, so unbounded charts keep the inner scale verbatim.
    Clamped {
        inner: Box<ValueScale>,
        lo: f64,
        hi: f64,
        range: f64,
    },
}

impl ValueScale {
    pub fn scale(&self, value: f64) -> f64 {
        match *self {
            ValueScale::Linear { min, max, range } => {
                if min == max {
                    return 0.0;
                }
                // d3 scaleLinear: normalize then interpolate [0, range]
                let t = (value - min) / (max - min);
                range * t
            }
            ValueScale::Log { min, max, range } => {
                if min == max {
                    return 0.0;
                }
                let t0 = min.ln();
                let t1 = max.ln();
                let t = (value.ln() - t0) / (t1 - t0);
                range * t
            }
            // d3-continuous (linear) mapping in ms space — identical math to
            // `Linear`; only `ticks()` differs (createTimeScale, scales.ts:237-242).
            ValueScale::Time { min, max, range } => {
                if min == max {
                    return 0.0;
                }
                let t = (value - min) / (max - min);
                range * t
            }
            ValueScale::Clamped {
                ref inner,
                lo,
                hi,
                range,
            } => {
                let pixels = inner.scale(value.max(lo).min(hi));
                // Defence in depth for a degenerate pinned domain: a non-finite
                // pixel would land on a child node's x/y/width.
                if !pixels.is_finite() {
                    return 0.0;
                }
                pixels.max(0.0).min(range)
            }
        }
    }

    /// `scale.ticks(count)`.
    pub fn ticks(&self, count: f64) -> Result<Vec<f64>, TickError> {
        super::ticks::check_count(count)?;
        Ok(match *self {
            ValueScale::Linear { min, max, .. } => generate_ticks(min, max, count)?,
            // createTimeScale.ticks (scales.ts:245-248): d3Scale.ticks(count)
            // mapped to ms. Zero-span returns [min] like the TS wrapper.
            // Realistically the domain comes from `time_nice` over
            // `data.timeRange`, which the TS parser only populates from
            // successfully-parsed finite dates (data.ts) — always integral ms
            // far inside i64/2^53. But a malformed persisted `timeRange` can
            // carry NaN/out-of-range bounds, and the `as i64` cast saturates
            // (NaN → 0, 1e20 → i64::MAX), so guard the boundary first: an
            // Invalid-Date domain yields no ticks (d3 parity), not a bogus
            // 1970-era axis.
            ValueScale::Time { min, max, .. } => {
                if !ms_in_date_range(min) || !ms_in_date_range(max) {
                    return Ok(Vec::new());
                }
                if min == max {
                    return Ok(vec![min]);
                }
                utc_ticks(min as i64, max as i64, count)?
                    .into_iter()
                    .map(|ms| ms as f64)
                    .collect()
            }
            ValueScale::Log { min, max, .. } => {
                if min == max {
                    return Ok(vec![min]);
                }
                // scales.ts:182-194 — decade-capped count, then d3 loggish ticks
                let decades = max.log10() - min.log10();
                let effective = if decades >= 1.0 {
                    js_min(count, decades.floor())
                } else {
                    count
                };
                loggish_ticks(min, max, effective)?
            }
            // Clamping is a mark-placement concern only; ticks come from the
            // inner scale and `filter_ticks_to_domain` drops the out-of-window
            // ones downstream, exactly like the TS wrapper.
            ValueScale::Clamped { ref inner, .. } => inner.ticks(count)?,
        })
    }
}

/// `generateTicks` (scales.ts:51-60): nice the bounds, then tick.
pub fn generate_ticks(min: f64, max: f64, count: f64) -> Result<Vec<f64>, TickError> {
    super::ticks::check_count(count)?;
    if !min.is_finite() || !max.is_finite() {
        return Ok(Vec::new());
    }
    if count <= 0.0 {
        return Ok(Vec::new());
    }
    if min == max {
        return Ok(vec![min]);
    }
    let (nice_min, nice_max) = d3_nice(min, max, count);
    d3_ticks(nice_min, nice_max, count)
}

/// `isLogDomainValid` (scales.ts:71-73).
pub fn is_log_domain_valid(min: f64, max: f64) -> bool {
    min.is_finite() && max.is_finite() && min > 0.0 && max > 0.0
}

/// d3 log.js `pow10` — `+("1e" + x)`.
fn pow10(x: f64) -> f64 {
    if x.is_finite() {
        format!("1e{}", x as i64).parse::<f64>().unwrap_or(f64::NAN)
    } else if x < 0.0 {
        0.0
    } else {
        x
    }
}

/// d3 loggish `scale.ticks(count)` for base 10, positive domain
/// (`d3-scale/src/log.js`).
fn loggish_ticks(u: f64, v: f64, count: f64) -> Result<Vec<f64>, TickError> {
    let (u, v, reversed) = if v < u { (v, u, true) } else { (u, v, false) };
    let mut i = u.log10();
    let mut j = v.log10();
    let n = count;
    let mut z: Vec<f64> = Vec::new();
    if j - i < n {
        // base 10 is integral → the `!(base % 1)` branch
        i = i.floor();
        j = j.ceil();
        if u > 0.0 {
            let mut ii = i;
            while ii <= j {
                for k in 1..10 {
                    let k = k as f64;
                    let t = if ii < 0.0 {
                        k / pow10(-ii)
                    } else {
                        k * pow10(ii)
                    };
                    if t < u {
                        continue;
                    }
                    if t > v {
                        break;
                    }
                    z.push(t);
                }
                ii += 1.0;
            }
        }
        if (z.len() as f64) * 2.0 < n {
            z = d3_ticks(u, v, n)?;
        }
    } else {
        z = d3_ticks(i, j, js_min(j - i, n))?
            .into_iter()
            .map(pow10)
            .collect();
    }
    if reversed {
        z.reverse();
    }
    Ok(z)
}

/// `getDisplayRangeFromTicks` (scales.ts:75-120), 'linear' | 'log' | 'time'.
pub fn get_display_range_from_ticks(domain: (f64, f64), count: f64, kind: &str) -> (f64, f64) {
    let (min, max) = domain;
    if !min.is_finite() || !max.is_finite() {
        return (min, max);
    }
    // Zero-span time domains soft-pad to ±1 day BEFORE the count<=0 early
    // return so single-point time charts still get a visible axis
    // (scales.ts:86-89).
    if min == max && kind == "time" {
        const ONE_DAY_MS: f64 = 24.0 * 60.0 * 60.0 * 1000.0;
        return (min - ONE_DAY_MS, max + ONE_DAY_MS);
    }
    if min == max {
        return (min, max);
    }
    if count <= 0.0 {
        return (min, max);
    }
    if kind == "log" {
        if min <= 0.0 || max <= 0.0 {
            return (min, max);
        }
        // d3.scaleLog().nice(): floor/ceil to powers of 10
        return (pow10(min.log10().floor()), pow10(max.log10().ceil()));
    }
    if kind == "time" {
        // Same f64→i64 boundary guard as ValueScale::Time::ticks: a finite but
        // out-of-JS-Date-range domain (|ms| > 8.64e15) would saturate the cast
        // to i64::MAX/MIN and round-trip back to a *different* finite f64
        // (9.22e18), fabricating a nice'd range where scaleUtc().nice() leaves
        // an Invalid-Date domain untouched. (Non-finite already returned above.)
        if !ms_in_date_range(min) || !ms_in_date_range(max) {
            return (min, max);
        }
        // d3-scaleUtc().nice(count): snap to hour/day/week/month/year
        // boundaries via the d3-time interval ladder (scales.ts:106-117).
        let (lo, hi) = time_nice((min as i64, max as i64), count);
        return (lo as f64, hi as f64);
    }
    d3_nice(min, max, count)
}

/// `createLinearScale` / `createLogScale` (with the log→linear fallback the
/// TS performs inside createLogScale; callers pre-check with
/// `is_log_domain_valid` exactly like the generators do).
pub fn create_linear_scale(domain: (f64, f64), range: f64) -> ValueScale {
    ValueScale::Linear {
        min: domain.0,
        max: domain.1,
        range,
    }
}

pub fn create_log_scale(domain: (f64, f64), range: f64) -> ValueScale {
    if domain.0 <= 0.0 || domain.1 <= 0.0 {
        // scales.ts:163-171 — fall back to linear (warning elided; moda-scene
        // has no logger and the manifest gates this case upstream).
        return create_linear_scale(domain, range);
    }
    ValueScale::Log {
        min: domain.0,
        max: domain.1,
        range,
    }
}

/// `createTimeScale` (scales.ts:226-264). UTC; domain is `[min, max]` ms
/// (callers route the raw data range through
/// `get_display_range_from_ticks(_, _, "time")` first, exactly like the TS).
pub fn create_time_scale(domain: (f64, f64), range: f64) -> ValueScale {
    ValueScale::Time {
        min: domain.0,
        max: domain.1,
        range,
    }
}

/// `createSizeScale` (scales.ts:287-318).
pub fn create_size_scale(
    data_range: (f64, f64),
    size_range: (f64, f64),
    kind: &str,
) -> impl Fn(f64) -> f64 {
    let (data_min, data_max) = data_range;
    let (min_size, max_size) = size_range;
    let radius = kind == "radius";
    move |value: f64| {
        if data_min == data_max {
            return (min_size + max_size) / 2.0;
        }
        if radius {
            // d3.scaleLinear().clamp(true)
            let t = (value - data_min) / (data_max - data_min);
            let t = t.clamp(0.0, 1.0);
            return min_size + (max_size - min_size) * t;
        }
        let effective_data_min = if data_min <= 0.0 { 1.0 } else { data_min };
        let data_offset = if data_min <= 0.0 { 1.0 - data_min } else { 0.0 };
        let clamped = js_max(data_min, js_min(data_max, value));
        let effective = clamped + data_offset;
        let sqrt_ratio = (effective / effective_data_min).sqrt();
        js_min(max_size, min_size * sqrt_ratio)
    }
}

/// `createCategoricalScale` (scales.ts:320-354) over d3 scaleBand with
/// paddingInner = clampedGap, paddingOuter = clampedGap/2, align 0.5.
#[derive(Clone, Debug)]
pub struct CategoricalScale {
    positions: HashMap<String, f64>,
    bandwidth: f64,
}

impl CategoricalScale {
    pub fn new(categories: &[String], width: f64, group_gap: f64) -> CategoricalScale {
        let clamped_gap = if group_gap.is_finite() {
            js_max(0.0, js_min(1.0, group_gap))
        } else {
            0.0
        };
        if categories.is_empty() {
            return CategoricalScale {
                positions: HashMap::new(),
                bandwidth: 0.0,
            };
        }
        // d3-scale band.js rescale(): r0=0, r1=width, round=false, align=0.5
        let n = categories.len() as f64;
        let padding_inner = clamped_gap;
        let padding_outer = clamped_gap / 2.0;
        let start0 = 0.0;
        let stop = width;
        let step = (stop - start0) / 1f64.max(n - padding_inner + padding_outer * 2.0);
        let start = start0 + (stop - start0 - step * (n - padding_inner)) * 0.5;
        let bandwidth = step * (1.0 - padding_inner);
        let mut positions = HashMap::new();
        for (i, c) in categories.iter().enumerate() {
            // first registration wins (d3 ordinal ignores duplicate domain values)
            positions
                .entry(c.clone())
                .or_insert(start + step * i as f64);
        }
        CategoricalScale {
            positions,
            bandwidth,
        }
    }

    pub fn scale(&self, category: &str) -> f64 {
        // unknown categories → 0 (scales.ts:344-349)
        *self.positions.get(category).unwrap_or(&0.0)
    }

    pub fn bandwidth(&self) -> f64 {
        self.bandwidth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn band_scale_matches_d3() {
        // node: scaleBand().domain(['a','b','c']).range([0,300]).paddingInner(.2).paddingOuter(.1)
        //   → a=10, bandwidth=80 (step 100)
        let s = CategoricalScale::new(&["a".into(), "b".into(), "c".into()], 300.0, 0.2);
        assert_eq!(s.scale("a"), 10.0);
        assert_eq!(s.scale("b"), 110.0);
        assert_eq!(s.bandwidth(), 80.0);
        assert_eq!(s.scale("zz"), 0.0);
    }

    #[test]
    fn linear_scale_and_ticks() {
        let s = create_linear_scale((0.0, 250.0), 500.0);
        assert_eq!(s.scale(125.0), 250.0);
        assert_eq!(s.ticks(5.0).unwrap(), vec![0.0, 50.0, 100.0, 150.0, 200.0, 250.0]);
        let zero = create_linear_scale((5.0, 5.0), 100.0);
        assert_eq!(zero.scale(5.0), 0.0);
    }

    #[test]
    fn log_ticks_match_d3() {
        // node: scaleLog().domain([1,1000]).ticks(3) → [1, 10, 100, 1000]
        let s = create_log_scale((1.0, 1000.0), 100.0);
        assert_eq!(s.ticks(3.0).unwrap(), vec![1.0, 10.0, 100.0, 1000.0]);
        // wrapper caps count to decade count: ticks(5) on [1,1000] (3 decades)
        // → effective 3 → powers only
        assert_eq!(s.ticks(5.0).unwrap(), vec![1.0, 10.0, 100.0, 1000.0]);
    }

    #[test]
    fn display_range() {
        assert_eq!(
            get_display_range_from_ticks((0.0, 210.0), 5.0, "linear"),
            (0.0, 250.0)
        );
        assert_eq!(
            get_display_range_from_ticks((3.0, 800.0), 5.0, "log"),
            (1.0, 1000.0)
        );
        assert_eq!(
            get_display_range_from_ticks((5.0, 5.0), 5.0, "linear"),
            (5.0, 5.0)
        );
    }

    #[test]
    fn time_display_range_matches_d3() {
        // node: scaleUtc().domain([2024-01-03, 2024-03-28]).nice(6).domain()
        //   → [2023-12-31, 2024-03-31] (the nice interval is WEEK, so the ends
        //   snap to Sundays — not month starts).
        assert_eq!(
            get_display_range_from_ticks((1_704_240_000_000.0, 1_711_584_000_000.0), 6.0, "time"),
            (1_703_980_800_000.0, 1_711_843_200_000.0)
        );
        // Zero-span time domains soft-pad to ±1 day (even with count <= 0).
        let x = 1_705_276_800_000.0;
        assert_eq!(
            get_display_range_from_ticks((x, x), 5.0, "time"),
            (x - 86_400_000.0, x + 86_400_000.0)
        );
        assert_eq!(
            get_display_range_from_ticks((x, x), 0.0, "time"),
            (x - 86_400_000.0, x + 86_400_000.0)
        );
        // count <= 0 (non-zero-span) returns the raw tuple, like linear/log.
        assert_eq!(
            get_display_range_from_ticks((1_704_240_000_000.0, 1_711_584_000_000.0), 0.0, "time"),
            (1_704_240_000_000.0, 1_711_584_000_000.0)
        );
    }

    #[test]
    fn time_scale_ticks_and_px_match_d3() {
        // node: scaleUtc().domain([2023-12-01, 2024-04-01]).range([0,500])
        let scale = create_time_scale((1_703_980_800_000.0, 1_711_843_200_000.0), 500.0);
        assert_eq!(
            scale.ticks(6.0).unwrap(),
            vec![
                1_704_067_200_000.0,
                1_706_745_600_000.0,
                1_709_251_200_000.0
            ]
        );
        // px (scale(tick)) — bit-exact vs d3-continuous.
        assert_eq!(scale.scale(1_704_067_200_000.0), 5.494_505_494_505_494_5);
        assert_eq!(scale.scale(1_706_745_600_000.0), 175.824_175_824_175_82);
        assert_eq!(scale.scale(1_709_251_200_000.0), 335.164_835_164_835_15);
    }

    #[test]
    fn time_scale_zero_span_collapses() {
        // A zero-span time scale (only reachable by direct construction, since
        // the display range pads single points) collapses to 0 like Linear.
        let scale = create_time_scale((1_705_276_800_000.0, 1_705_276_800_000.0), 400.0);
        assert_eq!(scale.scale(1_705_276_800_000.0), 0.0);
        assert_eq!(scale.ticks(5.0).unwrap(), vec![1_705_276_800_000.0]);
    }

    #[test]
    fn malformed_time_domain_degrades_like_invalid_date() {
        // A persisted `timeRange` with a wrong-typed endpoint reaches us as NaN
        // (DataView::time_range preserves it); an absurd magnitude is finite but
        // outside JS's ±8.64e15 ms Date range. Both would saturate the f64→i64
        // cast (NaN→0 ⇒ a plausible 1970 axis; 1e20→i64::MAX ⇒ a 9.22e18 range),
        // so both must degrade like scaleUtc over an Invalid Date instead.
        let nan = f64::NAN;
        let oor = 1e20; // > JS_DATE_MAX_MS (8.64e15)
        let good = 1_700_000_000_000.0;

        // ticks(): Invalid-Date domain ⇒ no ticks (not a bogus 1970-era axis).
        assert!(create_time_scale((nan, good), 500.0).ticks(6.0).unwrap().is_empty());
        assert!(create_time_scale((good, nan), 500.0).ticks(6.0).unwrap().is_empty());
        assert!(create_time_scale((oor, good), 500.0).ticks(6.0).unwrap().is_empty());
        assert!(create_time_scale((-oor, oor), 500.0).ticks(6.0).unwrap().is_empty());

        // display-range / nice(): leave the (un-nice-able) domain untouched —
        // crucially returning the ORIGINAL out-of-range value, not the
        // i64::MAX round-trip (9.22e18) the saturating cast would manufacture.
        // (NaN != NaN, so the NaN endpoint is checked component-wise.)
        let (lo, hi) = get_display_range_from_ticks((nan, good), 6.0, "time");
        assert!(lo.is_nan() && hi == good);
        assert_eq!(
            get_display_range_from_ticks((oor, good), 6.0, "time"),
            (oor, good)
        );

        // Inert for realistic data: a valid domain still nices/ticks exactly as
        // before the guard (same values as time_*_match_d3 above).
        assert_eq!(
            get_display_range_from_ticks((1_704_240_000_000.0, 1_711_584_000_000.0), 6.0, "time"),
            (1_703_980_800_000.0, 1_711_843_200_000.0)
        );
        assert_eq!(
            create_time_scale((1_703_980_800_000.0, 1_711_843_200_000.0), 500.0).ticks(6.0).unwrap(),
            vec![
                1_704_067_200_000.0,
                1_706_745_600_000.0,
                1_709_251_200_000.0
            ]
        );
    }
}
