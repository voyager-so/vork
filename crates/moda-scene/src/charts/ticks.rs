//! d3-array `ticks` / `tickIncrement` / `nice` — operation-for-operation port
//! of `node_modules/d3-array/src/{ticks,nice}.js` (the versions vendored in
//! this repo), in f64 to match the production JS bit-for-bit.

use super::jsmath::js_round;

/// Maximum output or examined ticks per tick operation.
pub const MAX_TICKS: usize = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickError;

impl std::fmt::Display for TickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("chart ticks exceed the work budget; reduce tick counts")
    }
}
impl std::error::Error for TickError {}
impl From<TickError> for crate::value::Error {
    fn from(error: TickError) -> Self { <Self as serde::de::Error>::custom(error) }
}
impl From<TickError> for String {
    fn from(error: TickError) -> Self { error.to_string() }
}

pub fn check_count(count: f64) -> Result<(), TickError> {
    if count == f64::INFINITY || count > MAX_TICKS as f64 { Err(TickError) } else { Ok(()) }
}

const E10: f64 = 7.0710678118654755; // Math.sqrt(50)
const E5: f64 = 3.1622776601683795; // Math.sqrt(10)
const E2: f64 = std::f64::consts::SQRT_2; // Math.sqrt(2)

/// `tickSpec(start, stop, count)` → [i1, i2, inc].
fn tick_spec(start: f64, stop: f64, count: f64) -> (f64, f64, f64) {
    let step = (stop - start) / count.max(0.0);
    let power = step.log10().floor();
    let error = step / 10f64.powf(power);
    let factor = if error >= E10 {
        10.0
    } else if error >= E5 {
        5.0
    } else if error >= E2 {
        2.0
    } else {
        1.0
    };
    let (i1, i2, inc);
    if power < 0.0 {
        let mut inc_ = 10f64.powf(-power) / factor;
        let mut i1_ = js_round(start * inc_);
        let mut i2_ = js_round(stop * inc_);
        if i1_ / inc_ < start {
            i1_ += 1.0;
        }
        if i2_ / inc_ > stop {
            i2_ -= 1.0;
        }
        inc_ = -inc_;
        i1 = i1_;
        i2 = i2_;
        inc = inc_;
    } else {
        let inc_ = 10f64.powf(power) * factor;
        let mut i1_ = js_round(start / inc_);
        let mut i2_ = js_round(stop / inc_);
        if i1_ * inc_ < start {
            i1_ += 1.0;
        }
        if i2_ * inc_ > stop {
            i2_ -= 1.0;
        }
        i1 = i1_;
        i2 = i2_;
        inc = inc_;
    }
    if i2 < i1 && 0.5 <= count && count < 2.0 {
        return tick_spec(start, stop, count * 2.0);
    }
    (i1, i2, inc)
}

/// d3 `ticks(start, stop, count)`.
pub fn d3_ticks(start: f64, stop: f64, count: f64) -> Result<Vec<f64>, TickError> {
    check_count(count)?;
    if !(count > 0.0) {
        return Ok(Vec::new());
    }
    if start == stop {
        return Ok(vec![start]);
    }
    let reverse = stop < start;
    let (i1, i2, inc) = if reverse {
        tick_spec(stop, start, count)
    } else {
        tick_spec(start, stop, count)
    };
    if !(i2 >= i1) {
        return Ok(Vec::new());
    }
    let length = i2 - i1 + 1.0;
    if !length.is_finite() || length > MAX_TICKS as f64 {
        return Err(TickError);
    }
    let n = length as usize;
    let mut ticks = vec![0.0; n];
    if reverse {
        if inc < 0.0 {
            for (i, t) in ticks.iter_mut().enumerate() {
                *t = (i2 - i as f64) / -inc;
            }
        } else {
            for (i, t) in ticks.iter_mut().enumerate() {
                *t = (i2 - i as f64) * inc;
            }
        }
    } else if inc < 0.0 {
        for (i, t) in ticks.iter_mut().enumerate() {
            *t = (i1 + i as f64) / -inc;
        }
    } else {
        for (i, t) in ticks.iter_mut().enumerate() {
            *t = (i1 + i as f64) * inc;
        }
    }
    Ok(ticks)
}

/// d3 `tickIncrement`.
pub fn tick_increment(start: f64, stop: f64, count: f64) -> f64 {
    tick_spec(start, stop, count).2
}

/// d3-array `tickStep` (`d3-array/src/ticks.js`) — the signed, reverse-aware
/// step size. Used by the d3-time tick-interval heuristic (`timeticks.rs`)
/// for the year/millisecond fallbacks, exactly as `d3-time/src/ticks.js` does.
pub fn tick_step(start: f64, stop: f64, count: f64) -> f64 {
    let reverse = stop < start;
    let inc = if reverse {
        tick_increment(stop, start, count)
    } else {
        tick_increment(start, stop, count)
    };
    (if reverse { -1.0 } else { 1.0 }) * (if inc < 0.0 { 1.0 / -inc } else { inc })
}

/// d3-array `nice(start, stop, count)`.
pub fn d3_nice(mut start: f64, mut stop: f64, count: f64) -> (f64, f64) {
    let mut prestep: Option<f64> = None;
    loop {
        let step = tick_increment(start, stop, count);
        if Some(step) == prestep || step == 0.0 || !step.is_finite() {
            return (start, stop);
        } else if step > 0.0 {
            start = (start / step).floor() * step;
            stop = (stop / step).ceil() * step;
        } else {
            start = (start * step).ceil() / step;
            stop = (stop * step).floor() / step;
        }
        prestep = Some(step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authored_tick_count_exceeds_allocation_budget() {
        assert_eq!(d3_ticks(0.0, 1.0, 1e12), Err(TickError));
        assert_eq!(d3_ticks(0.0, 1.0, f64::INFINITY), Err(TickError));
    }

    #[test]
    fn ticks_match_d3() {
        // node -e "console.log(require('d3-array').ticks(0,210,5))" → [0,50,100,150,200]
        assert_eq!(
            d3_ticks(0.0, 210.0, 5.0).unwrap(),
            vec![0.0, 50.0, 100.0, 150.0, 200.0]
        );
        // node: d3.ticks(0, 1, 5) → [0, 0.2, 0.4, 0.6, 0.8, 1]
        let t = d3_ticks(0.0, 1.0, 5.0).unwrap();
        assert_eq!(t, vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        assert_eq!(d3_ticks(5.0, 5.0, 5.0).unwrap(), vec![5.0]);
        assert!(d3_ticks(0.0, 10.0, 0.0).unwrap().is_empty());
    }

    #[test]
    fn nice_matches_d3() {
        // node: d3.nice(0, 210, 5) → [0, 250]; d3.nice(3, 97, 5) → [0, 100]
        assert_eq!(d3_nice(0.0, 210.0, 5.0), (0.0, 250.0));
        assert_eq!(d3_nice(3.0, 97.0, 5.0), (0.0, 100.0));
        assert_eq!(d3_nice(-30.0, 210.0, 5.0), (-50.0, 250.0));
    }
}
