//! Port of d3-time (`node_modules/d3-time@3.1.0/src/*`) — the UTC interval
//! arithmetic, tick-interval heuristic, `utcTicks`, and the `nice` snapping
//! that `d3-scale`'s `scaleUtc` delegates to. This is the missing piece that
//! kept time-X-axis charts manifest-gated (see `charts::mod` history): the
//! chart engine had the numeric d3-array `ticks`/`nice` (`ticks.rs`) but not
//! d3's calendar intervals.
//!
//! Faithful, operation-for-operation port (like `ticks.rs`), but specialized
//! to the UTC ticker the chart engine instantiates (`scaleUtc`). All arithmetic
//! is on `i64` milliseconds-since-epoch, which is exact for the date range JS
//! `Date` supports and matches d3's float math bit-for-bit (every value stays
//! integral and well under 2^53).
//!
//! The generic `timeInterval` machinery (`floor`/`ceil`/`offset`/`range`/
//! `every`/`filter`, `interval.js`) is reproduced 1:1 with boxed closures so
//! the `.every(step)` filtered intervals (e.g. `second.every(15)`,
//! `hour.every(3)`, `month.every(3)`) behave exactly like d3's.

use super::ticks::TickError;
use super::jsmath::js_max;
use super::ticks::tick_step;
use std::rc::Rc;

const MS_PER_SEC: i64 = 1000;
const MS_PER_MIN: i64 = 60_000;
const MS_PER_HOUR: i64 = 3_600_000;
const MS_PER_DAY: i64 = 86_400_000;
const MS_PER_WEEK: i64 = 604_800_000;

/// JS `Date`'s valid range: `new Date(ms)` past ±8.64e15 ms is an Invalid Date.
/// Domains beyond it can't come from parsed dates (`data.ts` only yields
/// in-range instants); a malformed persisted `timeRange` carrying absurd
/// values degrades like d3-scaleUtc on an Invalid-Date domain (empty ticks /
/// no-op nice) instead of risking an i64 overflow panic in the interval math.
const JS_DATE_MAX_MS: i64 = 8_640_000_000_000_000;

fn in_date_range(ms: i64) -> bool {
    (-JS_DATE_MAX_MS..=JS_DATE_MAX_MS).contains(&ms)
}

/// f64 sibling of [`in_date_range`], for guarding the `f64 → i64` domain cast
/// in `scales.rs` BEFORE it can saturate. The cast is lossy at the extremes
/// (`NaN as i64 == 0`, `1e20 as i64 == i64::MAX`), so a malformed persisted
/// `timeRange` would otherwise slip past the i64 guards above as a plausible
/// 1970-era (or `i64::MAX`-derived) instant. Checking finiteness + range on
/// the f64 keeps the Invalid-Date degrade faithful all the way to the boundary.
/// `JS_DATE_MAX_MS` (8.64e15 < 2^53) is exactly representable in f64.
pub fn ms_in_date_range(ms: f64) -> bool {
    ms.is_finite() && ms.abs() <= JS_DATE_MAX_MS as f64
}

// d3-time `duration.js` (ms). Used only as the bisector keys / year+ms fallback
// divisors in the tick-interval heuristic.
const DURATION_SECOND: f64 = 1000.0;
const DURATION_MINUTE: f64 = 60_000.0;
const DURATION_HOUR: f64 = 3_600_000.0;
const DURATION_DAY: f64 = 86_400_000.0;
const DURATION_WEEK: f64 = 604_800_000.0;
const DURATION_MONTH: f64 = 2_592_000_000.0; // durationDay * 30
const DURATION_YEAR: f64 = 31_536_000_000.0; // durationDay * 365

// ---------------------------------------------------------------------------
// Integer helpers (floor division / euclidean modulo) — JS `Math.floor(a/b)`
// and the always-non-negative component extraction the UTC getters perform.
// ---------------------------------------------------------------------------

/// `Math.floor(a / b)` for integers (toward −∞), matching JS float division
/// then floor on integral operands.
fn fdiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// Euclidean modulo (result in `[0, b)` for `b > 0`).
fn fmod(a: i64, b: i64) -> i64 {
    a - fdiv(a, b) * b
}

// ---------------------------------------------------------------------------
// Proleptic Gregorian calendar (Howard Hinnant's days_from_civil /
// civil_from_days), matching JS `Date` UTC component math exactly over the
// supported range.
// ---------------------------------------------------------------------------

/// Days since 1970-01-01 for the civil date `(y, m, d)` with `m ∈ [1, 12]`.
/// Out-of-range `d` rolls forward exactly like JS `Date.setUTCDate`.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Civil date `(y, m ∈ [1,12], d ∈ [1,31])` from days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `(getUTCFullYear, getUTCMonth+1, getUTCDate)` for a ms timestamp.
fn ms_to_ymd(ms: i64) -> (i64, i64, i64) {
    civil_from_days(fdiv(ms, MS_PER_DAY))
}

/// Time-of-day remainder in `[0, MS_PER_DAY)`.
fn ms_time_of_day(ms: i64) -> i64 {
    ms - fdiv(ms, MS_PER_DAY) * MS_PER_DAY
}

// Interval floors as free functions — shared by the interval constructors and
// the multi-scale tier (`utc_multiscale_tier`) so the two can't drift.
fn floor_second(ms: i64) -> i64 {
    fdiv(ms, MS_PER_SEC) * MS_PER_SEC
}
fn floor_minute(ms: i64) -> i64 {
    fdiv(ms, MS_PER_MIN) * MS_PER_MIN
}
fn floor_hour(ms: i64) -> i64 {
    fdiv(ms, MS_PER_HOUR) * MS_PER_HOUR
}
fn floor_day(ms: i64) -> i64 {
    fdiv(ms, MS_PER_DAY) * MS_PER_DAY
}
/// Floor to the most recent `weekday`-start (0=Sun..=6=Sat) 00:00 UTC. Epoch
/// (1970-01-01) is a Thursday, so `getUTCDay == fmod(days + 4, 7)`; d3's
/// `utcWeekday(i)` subtracts `(getUTCDay + 7 - i) % 7` days.
fn floor_weekday(ms: i64, weekday: i64) -> i64 {
    let days = fdiv(ms, MS_PER_DAY);
    let dow = fmod(days + 4, 7);
    (days - fmod(dow + 7 - weekday, 7)) * MS_PER_DAY
}

/// `utcSunday` floor.
fn floor_week(ms: i64) -> i64 {
    floor_weekday(ms, 0)
}

fn floor_thursday(ms: i64) -> i64 {
    floor_weekday(ms, 4)
}

/// `utcThursday.ceil` — `floor(ms-1), offset(+1 week), floor`.
fn ceil_thursday(ms: i64) -> i64 {
    floor_thursday(ms - 1) + MS_PER_WEEK
}

/// `interval.count` for a weekday interval: whole weeks between the floored
/// ends.
fn weekday_count(start: i64, end: i64, weekday: i64) -> i64 {
    fdiv(
        floor_weekday(end, weekday) - floor_weekday(start, weekday),
        MS_PER_WEEK,
    )
}

/// `UTCdISO` (`d3-time-format/src/locale.js`): the ISO-week anchor Thursday.
fn utc_diso(ms: i64) -> i64 {
    let dow = fmod(fdiv(ms, MS_PER_DAY) + 4, 7);
    if dow >= 4 || dow == 0 {
        floor_thursday(ms)
    } else {
        ceil_thursday(ms)
    }
}
fn floor_month(ms: i64) -> i64 {
    let (y, m, _d) = ms_to_ymd(ms);
    days_from_civil(y, m, 1) * MS_PER_DAY
}
fn floor_year(ms: i64) -> i64 {
    let (y, _m, _d) = ms_to_ymd(ms);
    days_from_civil(y, 1, 1) * MS_PER_DAY
}

// ---------------------------------------------------------------------------
// timeInterval (`interval.js`) — generic floor/ceil/offset/range/every/filter
// over i64 ms. `floori`/`offseti` are pure (return the new ms) rather than
// mutating a Date in place.
// ---------------------------------------------------------------------------

type Floori = Rc<dyn Fn(i64) -> i64>;
type Offseti = Rc<dyn Fn(i64, i64) -> i64>;
type CountFn = Rc<dyn Fn(i64, i64) -> f64>;
type FieldFn = Rc<dyn Fn(i64) -> i64>;
type EveryFn = Rc<dyn Fn(f64) -> Option<TimeInterval>>;

#[derive(Clone)]
pub struct TimeInterval {
    floori: Floori,
    offseti: Offseti,
    count: Option<CountFn>,
    field: Option<FieldFn>,
    /// `year`/`millisecond` ship optimized `.every` overrides in d3.
    every_override: Option<EveryFn>,
}

impl TimeInterval {
    fn new(floori: Floori, offseti: Offseti) -> TimeInterval {
        TimeInterval {
            floori,
            offseti,
            count: None,
            field: None,
            every_override: None,
        }
    }

    pub fn floor(&self, date: i64) -> i64 {
        (self.floori)(date)
    }

    /// `interval.ceil` — `floor(date-1), offset(.,1), floor(.)`.
    pub fn ceil(&self, date: i64) -> i64 {
        let d = (self.floori)(date - 1);
        let d = (self.offseti)(d, 1);
        (self.floori)(d)
    }

    /// `interval.range(start, stop, step=1)`.
    fn range(&self, start: i64, stop: i64, step: i64) -> Result<Vec<i64>, TickError> {
        let mut range = Vec::new();
        let mut start = self.ceil(start);
        if !(start < stop) || !(step > 0) {
            return Ok(range);
        }
        loop {
            let previous = start;
            if range.len() == super::ticks::MAX_TICKS { return Err(TickError); }
            range.push(previous);
            start = (self.offseti)(start, step);
            start = (self.floori)(start);
            if !(previous < start && start < stop) {
                break;
            }
        }
        Ok(range)
    }

    /// `interval.count(start, end)` — floor both, then `Math.floor(count(...))`.
    fn count(&self, start: i64, end: i64) -> f64 {
        let t0 = (self.floori)(start);
        let t1 = (self.floori)(end);
        (self.count.as_ref().expect("count interval")(t0, t1)).floor()
    }

    /// `interval.every(step)`.
    pub fn every(&self, step: f64) -> Option<TimeInterval> {
        if let Some(ev) = &self.every_override {
            return ev(step);
        }
        let step = step.floor();
        if !step.is_finite() || !(step > 0.0) {
            return None;
        }
        if !(step > 1.0) {
            return Some(self.clone());
        }
        let step_i = step as i64;
        let test: Rc<dyn Fn(i64) -> bool> = if let Some(field) = &self.field {
            let f = field.clone();
            Rc::new(move |d: i64| fmod(f(d), step_i) == 0)
        } else {
            let this = self.clone();
            Rc::new(move |d: i64| fmod(this.count(0, d) as i64, step_i) == 0)
        };
        Some(self.filter(test))
    }

    /// `interval.filter(test)`.
    fn filter(&self, test: Rc<dyn Fn(i64) -> bool>) -> TimeInterval {
        let base_floor = self.floori.clone();
        let t1 = test.clone();
        let floori: Floori = Rc::new(move |date: i64| -> i64 {
            let mut d = base_floor(date);
            while !t1(d) {
                d = base_floor(d - 1);
            }
            d
        });
        let base_offset = self.offseti.clone();
        let t2 = test;
        let offseti: Offseti = Rc::new(move |date: i64, step: i64| -> i64 {
            let mut d = date;
            if step < 0 {
                let mut s = step;
                loop {
                    s += 1;
                    if s > 0 {
                        break;
                    }
                    loop {
                        d = base_offset(d, -1);
                        if t2(d) {
                            break;
                        }
                    }
                }
            } else {
                let mut s = step;
                loop {
                    s -= 1;
                    if s < 0 {
                        break;
                    }
                    loop {
                        d = base_offset(d, 1);
                        if t2(d) {
                            break;
                        }
                    }
                }
            }
            d
        });
        TimeInterval::new(floori, offseti)
    }
}

// ---------------------------------------------------------------------------
// The UTC intervals the chart ticker uses
// (`ticker(utcYear, utcMonth, utcSunday, unixDay, utcHour, utcMinute)`).
// ---------------------------------------------------------------------------

/// Fixed-duration UTC interval (second / minute / hour / day): floor/offset are
/// exact multiples of `unit` ms because UTC has no DST.
fn fixed_interval(unit: i64, field: Option<FieldFn>) -> TimeInterval {
    let mut iv = TimeInterval::new(
        Rc::new(move |d| fdiv(d, unit) * unit),
        Rc::new(move |d, step| d + step * unit),
    );
    iv.count = Some(Rc::new(move |s, e| (e - s) as f64 / unit as f64));
    iv.field = field;
    iv
}

fn millisecond() -> TimeInterval {
    let mut iv = TimeInterval::new(Rc::new(|d| d), Rc::new(|d, step| d + step));
    iv.count = Some(Rc::new(|s, e| (e - s) as f64));
    iv.every_override = Some(Rc::new(|k: f64| {
        let k = k.floor();
        if !k.is_finite() || !(k > 0.0) {
            return None;
        }
        if !(k > 1.0) {
            return Some(millisecond());
        }
        let ki = k as i64;
        let mut opt = TimeInterval::new(
            Rc::new(move |d| fdiv(d, ki) * ki),
            Rc::new(move |d, step| d + step * ki),
        );
        opt.count = Some(Rc::new(move |s, e| (e - s) as f64 / ki as f64));
        Some(opt)
    }));
    iv
}

fn second() -> TimeInterval {
    fixed_interval(
        MS_PER_SEC,
        Some(Rc::new(|ms| fmod(fdiv(ms, MS_PER_SEC), 60))), // getUTCSeconds
    )
}

fn utc_minute() -> TimeInterval {
    fixed_interval(
        MS_PER_MIN,
        Some(Rc::new(|ms| fmod(fdiv(ms, MS_PER_MIN), 60))), // getUTCMinutes
    )
}

fn utc_hour() -> TimeInterval {
    fixed_interval(
        MS_PER_HOUR,
        Some(Rc::new(|ms| fmod(fdiv(ms, MS_PER_HOUR), 24))), // getUTCHours
    )
}

/// `unixDay` — the day interval the UTC ticker uses; `field` is
/// `Math.floor(date / durationDay)` (so `day.every(2)` picks even unix days).
fn unix_day() -> TimeInterval {
    fixed_interval(MS_PER_DAY, Some(Rc::new(|ms| fdiv(ms, MS_PER_DAY))))
}

/// `utcSunday` (`utcWeekday(0)`): floor to the most recent Sunday 00:00 UTC.
/// Epoch (1970-01-01) is a Thursday (`getUTCDay == 4`), so the day-of-week is
/// `fmod(days + 4, 7)`.
fn utc_sunday() -> TimeInterval {
    let mut iv = TimeInterval::new(
        Rc::new(floor_week),
        Rc::new(|ms, step| ms + step * MS_PER_WEEK),
    );
    iv.count = Some(Rc::new(|s, e| (e - s) as f64 / DURATION_WEEK));
    iv
}

/// `utcMonth`: floor to the 1st 00:00 UTC; offset preserves day-of-month & time
/// and rolls overflow forward exactly like `Date.setUTCMonth`.
fn utc_month() -> TimeInterval {
    let mut iv = TimeInterval::new(
        Rc::new(floor_month),
        Rc::new(|ms, step| {
            let tod = ms_time_of_day(ms);
            let (y, m, d) = ms_to_ymd(ms);
            let total = y * 12 + (m - 1) + step;
            let ny = fdiv(total, 12);
            let nm0 = fmod(total, 12);
            days_from_civil(ny, nm0 + 1, d) * MS_PER_DAY + tod
        }),
    );
    iv.count = Some(Rc::new(|s, e| {
        let (sy, sm, _) = ms_to_ymd(s);
        let (ey, em, _) = ms_to_ymd(e);
        ((em - 1) - (sm - 1) + (ey - sy) * 12) as f64
    }));
    iv.field = Some(Rc::new(|ms| ms_to_ymd(ms).1 - 1)); // getUTCMonth (0..11)
    iv
}

/// `utcYear`: floor to Jan 1 00:00 UTC; ships d3's optimized `.every(k)`.
fn utc_year() -> TimeInterval {
    let mut iv = TimeInterval::new(
        Rc::new(floor_year),
        Rc::new(|ms, step| {
            let tod = ms_time_of_day(ms);
            let (y, m, d) = ms_to_ymd(ms);
            days_from_civil(y + step, m, d) * MS_PER_DAY + tod
        }),
    );
    iv.count = Some(Rc::new(|s, e| (ms_to_ymd(e).0 - ms_to_ymd(s).0) as f64));
    iv.field = Some(Rc::new(|ms| ms_to_ymd(ms).0)); // getUTCFullYear
    iv.every_override = Some(Rc::new(|k: f64| {
        let k = k.floor();
        if !k.is_finite() || !(k > 0.0) {
            return None;
        }
        let ki = k as i64;
        let opt = TimeInterval::new(
            Rc::new(move |ms| {
                let (y, _m, _d) = ms_to_ymd(ms);
                days_from_civil(fdiv(y, ki) * ki, 1, 1) * MS_PER_DAY
            }),
            Rc::new(move |ms, step| {
                let tod = ms_time_of_day(ms);
                let (y, m, d) = ms_to_ymd(ms);
                days_from_civil(y + step * ki, m, d) * MS_PER_DAY + tod
            }),
        );
        Some(opt)
    }));
    iv
}

// ---------------------------------------------------------------------------
// The tick-interval heuristic (`ticker(...)` in `d3-time/src/ticks.js`).
// ---------------------------------------------------------------------------

/// `(constructor, step, duration_ms)` rows, in ascending duration order.
#[allow(clippy::type_complexity)]
fn tick_intervals() -> [(fn() -> TimeInterval, f64, f64); 18] {
    [
        (second, 1.0, DURATION_SECOND),
        (second, 5.0, 5.0 * DURATION_SECOND),
        (second, 15.0, 15.0 * DURATION_SECOND),
        (second, 30.0, 30.0 * DURATION_SECOND),
        (utc_minute, 1.0, DURATION_MINUTE),
        (utc_minute, 5.0, 5.0 * DURATION_MINUTE),
        (utc_minute, 15.0, 15.0 * DURATION_MINUTE),
        (utc_minute, 30.0, 30.0 * DURATION_MINUTE),
        (utc_hour, 1.0, DURATION_HOUR),
        (utc_hour, 3.0, 3.0 * DURATION_HOUR),
        (utc_hour, 6.0, 6.0 * DURATION_HOUR),
        (utc_hour, 12.0, 12.0 * DURATION_HOUR),
        (unix_day, 1.0, DURATION_DAY),
        (unix_day, 2.0, 2.0 * DURATION_DAY),
        (utc_sunday, 1.0, DURATION_WEEK),
        (utc_month, 1.0, DURATION_MONTH),
        (utc_month, 3.0, 3.0 * DURATION_MONTH),
        (utc_year, 1.0, DURATION_YEAR),
    ]
}

/// `bisector(([,,step]) => step).right(tickIntervals, target)` — bisect-right
/// on the duration column. Mirrors d3's NaN self-comparability guard
/// (`bisector.js`: `if (compare1(x, x) !== 0) return hi`): a NaN `target`
/// returns `hi` (= `len`), routing `utc_tick_interval` to the year branch
/// exactly like d3 (rather than the millisecond branch a plain bisect would
/// pick). Unreachable from JSON configs — `count` is always finite — but kept
/// faithful so the heuristic can't diverge if reused.
fn bisect_right_duration(rows: &[(fn() -> TimeInterval, f64, f64); 18], target: f64) -> usize {
    if target.is_nan() {
        return rows.len();
    }
    let mut lo = 0usize;
    let mut hi = rows.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if rows[mid].2 <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// `utcTickInterval(start, stop, count)`.
pub fn utc_tick_interval(start: i64, stop: i64, count: f64) -> Option<TimeInterval> {
    // `Math.abs(stop - start) / count` — computed in f64 like d3 (exact for
    // realistic ms < 2^53) so a malformed domain whose bounds saturate to
    // i64::MIN/MAX can't overflow the i64 subtraction and panic.
    let target = (stop as f64 - start as f64).abs() / count;
    let rows = tick_intervals();
    let i = bisect_right_duration(&rows, target);
    if i == rows.len() {
        return utc_year().every(tick_step(
            start as f64 / DURATION_YEAR,
            stop as f64 / DURATION_YEAR,
            count,
        ));
    }
    if i == 0 {
        return millisecond().every(js_max(tick_step(start as f64, stop as f64, count), 1.0));
    }
    let prev_dur = rows[i - 1].2;
    let cur_dur = rows[i].2;
    let idx = if target / prev_dur < cur_dur / target {
        i - 1
    } else {
        i
    };
    let (ctor, step, _dur) = rows[idx];
    ctor().every(step)
}

/// `utcTicks(start, stop, count)` — inclusive of `stop` (`+stop + 1`).
pub fn utc_ticks(start: i64, stop: i64, count: f64) -> Result<Vec<i64>, TickError> {
    super::ticks::check_count(count)?;
    // Out-of-Date-range bound → Invalid-Date domain → `[]` (d3 parity); also
    // keeps the interval math (and `stop + 1` below) safely inside i64.
    if !in_date_range(start) || !in_date_range(stop) {
        return Ok(Vec::new());
    }
    // d3 yields `[start]` for a zero-span domain (the millisecond.every(1)
    // range collapses to one tick).
    if start == stop {
        return Ok(vec![start]);
    }
    let reverse = stop < start;
    let (start, stop) = if reverse {
        (stop, start)
    } else {
        (start, stop)
    };
    let mut ticks = match utc_tick_interval(start, stop, count) {
        Some(iv) => iv.range(start, stop + 1, 1)?, // `+stop + 1` (inclusive)
        None => Vec::new(),
    };
    if reverse {
        ticks.reverse();
    }
    Ok(ticks)
}

/// `scaleUtc().domain(domain).nice(count).domain()` — snap the domain ends to
/// the chosen tick interval (d3-scale `nice.js`). Order is preserved.
pub fn time_nice(domain: (i64, i64), count: f64) -> (i64, i64) {
    let (d0, d1) = domain;
    // Out-of-Date-range domain → Invalid Date → `scaleUtc.nice` is a no-op
    // (and the floor/ceil calendar math stays inside i64).
    if !in_date_range(d0) || !in_date_range(d1) {
        return domain;
    }
    let interval = match utc_tick_interval(d0, d1, count) {
        Some(iv) => iv,
        // `tickInterval` null → `scale.nice` is a no-op (domain unchanged).
        None => return domain,
    };
    if d1 < d0 {
        (interval.ceil(d0), interval.floor(d1))
    } else {
        (interval.floor(d0), interval.ceil(d1))
    }
}

// ---------------------------------------------------------------------------
// UTC date components + the d3-scale multi-scale tier (`d3-scale/time.js`
// `tickFormat`), consumed by the d3-time-format port in `timefmt.rs`.
// ---------------------------------------------------------------------------

/// The UTC `Date` getters the d3-time-format directives read.
#[derive(Clone, Copy, Debug)]
pub struct UtcDateParts {
    pub year: i64,    // getUTCFullYear
    pub month0: i64,  // getUTCMonth (0..=11)
    pub day: i64,     // getUTCDate (1..=31)
    pub weekday: i64, // getUTCDay (0=Sun..=6)
    pub hours: i64,   // getUTCHours (0..=23)
    pub minutes: i64, // getUTCMinutes (0..=59)
    pub seconds: i64, // getUTCSeconds (0..=59)
    pub millis: i64,  // getUTCMilliseconds (0..=999)
    /// `1 + utcDay.count(utcYear(d), d)` — day of year (1..=366).
    pub day_of_year: i64,
}

/// Decompose a ms timestamp into its UTC `Date` components.
pub fn utc_date_parts(ms: i64) -> UtcDateParts {
    let (year, m, day) = ms_to_ymd(ms);
    let days = fdiv(ms, MS_PER_DAY);
    let year_start_days = days_from_civil(year, 1, 1);
    UtcDateParts {
        year,
        month0: m - 1,
        day,
        weekday: fmod(days + 4, 7),
        hours: fmod(fdiv(ms, MS_PER_HOUR), 24),
        minutes: fmod(fdiv(ms, MS_PER_MIN), 60),
        seconds: fmod(fdiv(ms, MS_PER_SEC), 60),
        millis: fmod(ms, 1000),
        day_of_year: 1 + (days - year_start_days),
    }
}

/// `%U` — `utcSunday.count(utcYear(d) - 1, d)` (week of year, Sunday-based).
pub fn utc_week_of_year_sunday(ms: i64) -> i64 {
    weekday_count(floor_year(ms) - 1, ms, 0)
}

/// `%W` — `utcMonday.count(utcYear(d) - 1, d)` (week of year, Monday-based).
pub fn utc_week_of_year_monday(ms: i64) -> i64 {
    weekday_count(floor_year(ms) - 1, ms, 1)
}

/// `%V` — ISO 8601 week of year (`formatUTCWeekNumberISO`).
pub fn utc_week_of_year_iso(ms: i64) -> i64 {
    let d = utc_diso(ms);
    let year_start = floor_year(d);
    let jan1_dow = fmod(fdiv(year_start, MS_PER_DAY) + 4, 7);
    weekday_count(year_start, d, 4) + if jan1_dow == 4 { 1 } else { 0 }
}

/// `%G`/`%g` anchor year — the ISO-week-numbering year (`getUTCFullYear` of the
/// ISO anchor Thursday).
pub fn utc_iso_year(ms: i64) -> i64 {
    ms_to_ymd(utc_diso(ms)).0
}

/// Which interval boundary a tick lands on — the selector d3-scale's `calendar`
/// `tickFormat` uses to pick a per-tick format (`d3-scale/src/time.js:24-34`):
/// `second(d) < d ? Millisecond : minute(d) < d ? Second : …`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UtcTier {
    Millisecond,
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Year,
}

/// `tickFormat(date)` tier selection (`d3-scale/src/time.js`).
pub fn utc_multiscale_tier(ms: i64) -> UtcTier {
    if floor_second(ms) < ms {
        UtcTier::Millisecond
    } else if floor_minute(ms) < ms {
        UtcTier::Second
    } else if floor_hour(ms) < ms {
        UtcTier::Minute
    } else if floor_day(ms) < ms {
        UtcTier::Hour
    } else if floor_month(ms) < ms {
        if floor_week(ms) < ms {
            UtcTier::Day
        } else {
            UtcTier::Week
        }
    } else if floor_year(ms) < ms {
        UtcTier::Month
    } else {
        UtcTier::Year
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values from node + the vendored d3 (d3-time@3.1.0, d3-scale@4.0.2):
    //   scaleUtc().domain([min,max]).nice(count).domain()  → time_nice
    //   scaleUtc().domain(nice).ticks(count)                → utc_ticks
    const DAY: i64 = 86_400_000;

    #[test]
    fn calendar_round_trips_and_matches_js() {
        // 2024-01-15 is 19737 days after epoch; civil_from_days inverts.
        assert_eq!(days_from_civil(2024, 1, 15), 19737);
        assert_eq!(civil_from_days(19737), (2024, 1, 15));
        // epoch
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // pre-epoch
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        // leap day
        assert_eq!(civil_from_days(days_from_civil(2024, 2, 29)), (2024, 2, 29));
    }

    #[test]
    fn week_floor_lands_on_sunday() {
        // 1970-01-01 (Thu) floors to 1969-12-28 (Sun).
        let wk = utc_sunday();
        assert_eq!(wk.floor(0), days_from_civil(1969, 12, 28) * DAY);
        // 2024-01-14 is a Sunday → floors to itself.
        let sun = days_from_civil(2024, 1, 14) * DAY;
        assert_eq!(wk.floor(sun), sun);
    }

    #[test]
    fn nice_jan_mar_2024() {
        // node: scaleUtc().domain([2024-01-03, 2024-03-28]).nice(6).domain()
        //   → [2023-12-31, 2024-03-31]. The chosen interval is WEEK (not
        //   month): the bisector heuristic lands ~14d closer to week×2 than
        //   month, so the ends snap to Sunday boundaries.
        let min = days_from_civil(2024, 1, 3) * DAY;
        let max = days_from_civil(2024, 3, 28) * DAY;
        let (lo, hi) = time_nice((min, max), 6.0);
        assert_eq!(lo, days_from_civil(2023, 12, 31) * DAY);
        assert_eq!(hi, days_from_civil(2024, 3, 31) * DAY);
        assert_eq!(lo, 1_703_980_800_000);
        assert_eq!(hi, 1_711_843_200_000);
    }

    #[test]
    fn ticks_jan_mar_2024_month_starts() {
        // Over the niced [2023-12-31, 2024-03-31] range, ticks(6) lands on
        // month starts: node → [2024-01-01, 2024-02-01, 2024-03-01].
        let min = days_from_civil(2024, 1, 3) * DAY;
        let max = days_from_civil(2024, 3, 28) * DAY;
        let (lo, hi) = time_nice((min, max), 6.0);
        let ticks = utc_ticks(lo, hi, 6.0).unwrap();
        assert_eq!(
            ticks,
            vec![
                days_from_civil(2024, 1, 1) * DAY,
                days_from_civil(2024, 2, 1) * DAY,
                days_from_civil(2024, 3, 1) * DAY,
            ]
        );
        assert_eq!(
            ticks,
            vec![1_704_067_200_000, 1_706_745_600_000, 1_709_251_200_000]
        );
    }

    #[test]
    fn ticks_single_day_3h_steps() {
        // domain Jan 15 00:00 .. Jan 16 00:00, count 6 → 3-hour ticks
        let lo = days_from_civil(2024, 1, 15) * DAY;
        let hi = days_from_civil(2024, 1, 16) * DAY;
        let ticks = utc_ticks(lo, hi, 6.0).unwrap();
        let expected: Vec<i64> = (0..=8).map(|h| lo + h * 3 * MS_PER_HOUR).collect();
        assert_eq!(ticks, expected);
    }

    #[test]
    fn ticks_weeks() {
        // node: nice([2024-01-01, 2024-02-15], 7) → [2023-12-31, 2024-02-18];
        //   ticks(7) → weekly Sundays
        let min = days_from_civil(2024, 1, 1) * DAY;
        let max = days_from_civil(2024, 2, 15) * DAY;
        let (lo, hi) = time_nice((min, max), 7.0);
        assert_eq!(lo, days_from_civil(2023, 12, 31) * DAY);
        assert_eq!(hi, days_from_civil(2024, 2, 18) * DAY);
        let ticks = utc_ticks(lo, hi, 7.0).unwrap();
        assert_eq!(
            ticks,
            vec![
                days_from_civil(2023, 12, 31) * DAY,
                days_from_civil(2024, 1, 7) * DAY,
                days_from_civil(2024, 1, 14) * DAY,
                days_from_civil(2024, 1, 21) * DAY,
                days_from_civil(2024, 1, 28) * DAY,
                days_from_civil(2024, 2, 4) * DAY,
                days_from_civil(2024, 2, 11) * DAY,
                days_from_civil(2024, 2, 18) * DAY,
            ]
        );
    }

    #[test]
    fn ticks_multi_year() {
        // node: nice([2019-06-01, 2024-02-01], 5) → [2019-01-01, 2025-01-01]
        let min = days_from_civil(2019, 6, 1) * DAY;
        let max = days_from_civil(2024, 2, 1) * DAY;
        let (lo, hi) = time_nice((min, max), 5.0);
        assert_eq!(lo, days_from_civil(2019, 1, 1) * DAY);
        assert_eq!(hi, days_from_civil(2025, 1, 1) * DAY);
        let ticks = utc_ticks(lo, hi, 5.0).unwrap();
        let years: Vec<i64> = ticks.iter().map(|&t| ms_to_ymd(t).0).collect();
        assert_eq!(years, vec![2019, 2020, 2021, 2022, 2023, 2024, 2025]);
    }

    #[test]
    fn ticks_decade_5y_steps() {
        // node: nice([2000-01-01, 2024-01-01], 5) → [2000-01-01, 2025-01-01];
        //   ticks → every 5 years (year.every(5) via the i===length branch)
        let lo = days_from_civil(2000, 1, 1) * DAY;
        let hi = days_from_civil(2025, 1, 1) * DAY;
        let ticks = utc_ticks(lo, hi, 5.0).unwrap();
        let years: Vec<i64> = ticks.iter().map(|&t| ms_to_ymd(t).0).collect();
        assert_eq!(years, vec![2000, 2005, 2010, 2015, 2020, 2025]);
    }

    #[test]
    fn ticks_seconds_15s_steps() {
        let lo = days_from_civil(2024, 1, 15) * DAY + 12 * MS_PER_HOUR;
        let hi = lo + MS_PER_MIN;
        let ticks = utc_ticks(lo, hi, 6.0).unwrap();
        assert_eq!(
            ticks,
            vec![
                lo,
                lo + 15 * MS_PER_SEC,
                lo + 30 * MS_PER_SEC,
                lo + 45 * MS_PER_SEC,
                lo + 60 * MS_PER_SEC,
            ]
        );
    }

    #[test]
    fn ticks_pre_epoch_years() {
        // node: nice([1965-03-10, 1969-11-20], 5) → [1965-01-01, 1970-01-01]
        let min = days_from_civil(1965, 3, 10) * DAY;
        let max = days_from_civil(1969, 11, 20) * DAY;
        let (lo, hi) = time_nice((min, max), 5.0);
        assert_eq!(lo, days_from_civil(1965, 1, 1) * DAY);
        assert_eq!(hi, days_from_civil(1970, 1, 1) * DAY);
        let ticks = utc_ticks(lo, hi, 5.0).unwrap();
        let years: Vec<i64> = ticks.iter().map(|&t| ms_to_ymd(t).0).collect();
        assert_eq!(years, vec![1965, 1966, 1967, 1968, 1969, 1970]);
        assert_eq!(ticks[0], -157_766_400_000);
        assert_eq!(*ticks.last().unwrap(), 0);
    }

    #[test]
    fn reversed_domain_reverses_ticks() {
        let lo = days_from_civil(2024, 1, 1) * DAY;
        let hi = days_from_civil(2024, 4, 1) * DAY;
        let fwd = utc_ticks(lo, hi, 6.0).unwrap();
        let rev = utc_ticks(hi, lo, 6.0).unwrap();
        let mut fwd_rev = fwd.clone();
        fwd_rev.reverse();
        assert_eq!(rev, fwd_rev);
    }

    #[test]
    fn reversed_multi_year_nice_is_a_noop_like_d3() {
        // d3 `tickStep` returns a NEGATIVE step for a reversed (stop < start)
        // domain, so `utcYear.every(neg)` is null and `scaleUtc().nice()`
        // no-ops — the reversed domain comes back untouched. The forward
        // direction of the same span DOES nice (year boundaries). Pinning
        // both guards against a future "tick_step should use abs" refactor
        // that would silently diverge from d3 on reversed bounds.
        let d1 = days_from_civil(2010, 3, 15) * DAY;
        let d0 = days_from_civil(2020, 9, 20) * DAY;
        // forward: floor/ceil to year boundaries (utcYear.every(1)).
        assert_eq!(
            time_nice((d1, d0), 10.0),
            (
                days_from_civil(2010, 1, 1) * DAY,
                days_from_civil(2021, 1, 1) * DAY
            )
        );
        // reversed: negative tickStep ⇒ every() → null ⇒ nice no-op.
        assert_eq!(time_nice((d0, d1), 10.0), (d0, d1));
    }

    #[test]
    fn degenerate_counts_match_d3() {
        // node parity for pathological counts: utcTicks returns [] for
        // count <= 0 and count = NaN; time_nice is a no-op (domain unchanged).
        // The NaN path routes through the year branch (bisector NaN guard),
        // exactly like d3 — and, like d3, collapses to the same empty result.
        let lo = days_from_civil(2024, 1, 3) * DAY;
        let hi = days_from_civil(2024, 3, 28) * DAY;
        assert!(utc_ticks(lo, hi, 0.0).unwrap().is_empty());
        assert!(utc_ticks(lo, hi, -5.0).unwrap().is_empty());
        assert!(utc_ticks(lo, hi, f64::NAN).unwrap().is_empty());
        assert_eq!(time_nice((lo, hi), f64::NAN), (lo, hi));
        assert_eq!(time_nice((lo, hi), 0.0), (lo, hi));
    }

    #[test]
    fn out_of_date_range_domains_degrade_without_panic() {
        // |ms| beyond JS Date's ±8.64e15 (incl. i64 saturation from absurd f64
        // `as i64`) is an Invalid Date: d3 gives empty ticks and a no-op nice.
        // We must match that and, above all, never overflow-panic.
        let over = JS_DATE_MAX_MS + 1;
        for (a, b) in [
            (i64::MIN, i64::MAX),
            (-over, over),
            (i64::MAX, i64::MIN), // reversed
        ] {
            assert!(utc_ticks(a, b, 5.0).unwrap().is_empty(), "{a}..{b} ticks");
            assert_eq!(time_nice((a, b), 5.0), (a, b), "{a}..{b} nice");
        }
        // A saturated zero-span must NOT return [i64::MAX] — that's an Invalid
        // Date (out-of-range guard runs before the start==stop short-circuit).
        assert!(utc_ticks(i64::MAX, i64::MAX, 5.0).unwrap().is_empty());
        // The exact ±boundary is still valid (in range) and produces ticks.
        assert!(!utc_ticks(JS_DATE_MAX_MS - MS_PER_DAY, JS_DATE_MAX_MS, 5.0).unwrap().is_empty());
        assert!(!utc_ticks(-JS_DATE_MAX_MS, -JS_DATE_MAX_MS + MS_PER_DAY, 5.0).unwrap().is_empty());
    }
}
