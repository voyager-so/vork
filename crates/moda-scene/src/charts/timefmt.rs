//! Port of d3-time-format (`node_modules/d3-time-format@4.1.0/src/locale.js`)
//! `utcFormat`, plus d3-scale's multi-scale time `tickFormat`
//! (`d3-scale/src/time.js`) — the two formatting paths a time X axis needs:
//!
//!  1. No `xTimeFormat` → the multi-scale formatter: each tick is labeled by
//!     the coarsest unit it lands on (`Jan`, `12 PM`, `2024`, …). This is the
//!     whole reason time axes exist as a first-class scale.
//!  2. Explicit `xTimeFormat` → `utcFormat(specifier)` applied to every tick.
//!
//! `newFormat`'s directive engine is ported faithfully for the full set of
//! `utcFormats` directives d3 supports — single-field (`a A b B d e H I j L m M
//! p S u w y Y`), week-of-year / ISO (`U W V G g`), unix timestamps (`Q s`),
//! the quarter/microsecond/zone helpers (`q f Z`), and the locale composites
//! (`c x X`, expanded recursively). A genuinely-unknown directive char emits
//! its literal char, exactly like d3 does for an unmapped specifier.
//!
//! Locale is the d3 default `en-US` (`d3-time-format/src/defaultLocale.js`).

use super::timeticks::{
    utc_date_parts, utc_iso_year, utc_multiscale_tier, utc_week_of_year_iso,
    utc_week_of_year_monday, utc_week_of_year_sunday, UtcDateParts, UtcTier,
};

const SHORT_DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const DAYS: [&str; 7] = [
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
];
const SHORT_MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];
const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
const PERIODS: [&str; 2] = ["AM", "PM"];

/// d3 `pad(value, fill, width)` (`locale.js`). `fill = None` is the `-`
/// modifier (no padding); the absolute value is zero/space-padded to `width`
/// and the sign re-prepended.
fn pad(value: i64, fill: Option<char>, width: usize) -> String {
    let sign = if value < 0 { "-" } else { "" };
    let s = value.unsigned_abs().to_string();
    match fill {
        None => format!("{sign}{s}"),
        Some(f) if s.len() < width => {
            let padding: String = std::iter::repeat(f).take(width - s.len()).collect();
            format!("{sign}{padding}{s}")
        }
        Some(_) => format!("{sign}{s}"),
    }
}

/// One UTC directive (`utcFormats[c]`). `fill` is the resolved pad character
/// (`None` = no pad). Genuinely-unmapped directive chars return `None` so the
/// caller emits the literal char (d3's fallback for an unknown directive).
fn format_directive(dir: char, fill: Option<char>, ms: i64, p: &UtcDateParts) -> Option<String> {
    let hour12 = {
        let h = p.hours % 12;
        if h == 0 {
            12
        } else {
            h
        }
    };
    Some(match dir {
        'a' => SHORT_DAYS[p.weekday as usize].to_string(),
        'A' => DAYS[p.weekday as usize].to_string(),
        'b' | 'h' => SHORT_MONTHS[p.month0 as usize].to_string(),
        'B' => MONTHS[p.month0 as usize].to_string(),
        // Locale composites (`%x` date, `%X` time, `%c` datetime) — the en-US
        // default expansions, formatted recursively like d3's nested newFormat.
        'c' => utc_format("%x, %X", ms),
        'd' => pad(p.day, fill, 2),
        'e' => pad(p.day, fill, 2),
        'f' => format!("{}000", pad(p.millis, fill, 3)), // microseconds
        'G' => pad(utc_iso_year(ms) % 10000, fill, 4),
        'g' => pad(utc_iso_year(ms) % 100, fill, 2),
        'H' => pad(p.hours, fill, 2),
        'I' => pad(hour12, fill, 2),
        'j' => pad(p.day_of_year, fill, 3),
        'L' => pad(p.millis, fill, 3),
        'm' => pad(p.month0 + 1, fill, 2),
        'M' => pad(p.minutes, fill, 2),
        'p' => PERIODS[if p.hours >= 12 { 1 } else { 0 }].to_string(),
        'q' => (1 + p.month0 / 3).to_string(),
        'Q' => ms.to_string(),                  // unix ms
        's' => ms.div_euclid(1000).to_string(), // unix seconds (floored)
        'S' => pad(p.seconds, fill, 2),
        'u' => (if p.weekday == 0 { 7 } else { p.weekday }).to_string(),
        'U' => pad(utc_week_of_year_sunday(ms), fill, 2),
        'V' => pad(utc_week_of_year_iso(ms), fill, 2),
        'w' => p.weekday.to_string(),
        'W' => pad(utc_week_of_year_monday(ms), fill, 2),
        'x' => utc_format("%-m/%-d/%Y", ms),
        'X' => utc_format("%-I:%M:%S %p", ms),
        // JS `%` is truncating (keeps the sign), so pre-year-0 dates render
        // e.g. "-05" / "-0005" rather than a euclidean "95".
        'y' => pad(p.year % 100, fill, 2),
        'Y' => pad(p.year % 10000, fill, 4),
        'Z' => "+0000".to_string(),
        '%' => "%".to_string(),
        _ => return None,
    })
}

/// d3 `newFormat(specifier, utcFormats)` (`locale.js`) — formats a UTC ms
/// timestamp through a `%`-directive specifier.
pub fn utc_format(specifier: &str, ms: i64) -> String {
    let parts = utc_date_parts(ms);
    let chars: Vec<char> = specifier.chars().collect();
    let n = chars.len();
    let mut out = String::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c != '%' {
            out.push(c);
            i += 1;
            continue;
        }
        // c == '%': read optional pad modifier, then the directive char.
        i += 1;
        if i >= n {
            // trailing '%' with no directive — d3 pushes the empty format.
            break;
        }
        let (fill, dir) = match chars[i] {
            '-' => (None, chars.get(i + 1).copied()),
            '_' => (Some(' '), chars.get(i + 1).copied()),
            '0' => (Some('0'), chars.get(i + 1).copied()),
            other => {
                // No modifier; 'e' defaults to space pad, everything else '0'.
                let fill = if other == 'e' { Some(' ') } else { Some('0') };
                i += 1;
                match format_directive(other, fill, ms, &parts) {
                    Some(s) => out.push_str(&s),
                    None => out.push(other), // unmapped → literal char
                }
                continue;
            }
        };
        // A pad modifier was consumed; the directive is the next char.
        i += 2;
        match dir {
            Some(d) => match format_directive(d, fill, ms, &parts) {
                Some(s) => out.push_str(&s),
                None => out.push(d),
            },
            None => {} // pad modifier at end of string — nothing to format.
        }
    }
    out
}

/// d3-scale `calendar` multi-scale `tickFormat(date)` (`time.js:24-34`): pick a
/// format by the coarsest interval boundary the tick lands on.
pub fn utc_multiscale_format(ms: i64) -> String {
    let specifier = match utc_multiscale_tier(ms) {
        UtcTier::Millisecond => ".%L",
        UtcTier::Second => ":%S",
        UtcTier::Minute => "%I:%M",
        UtcTier::Hour => "%I %p",
        UtcTier::Day => "%a %d",
        UtcTier::Week => "%b %d",
        UtcTier::Month => "%B",
        UtcTier::Year => "%Y",
    };
    utc_format(specifier, ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle: node + d3-time-format@4.1.0 utcFormat, and d3-scale@4.0.2
    // scaleUtc().tickFormat() for the multi-scale labels.
    const D_2024_01_15: i64 = 1_705_276_800_000; // 2024-01-15T00:00:00Z (Mon)
    const D_MID: i64 = 1_718_459_130_250; // 2024-06-15T13:45:30.250Z (Sat)
    const D_2020_END: i64 = 1_609_459_199_000; // 2020-12-31T23:59:59Z (Thu)

    #[test]
    fn custom_specifiers_match_d3() {
        assert_eq!(utc_format("%Y-%m-%d", D_2024_01_15), "2024-01-15");
        assert_eq!(utc_format("%Y-%m-%d", D_MID), "2024-06-15");
        assert_eq!(utc_format("%Y-%m-%d", D_2020_END), "2020-12-31");
        assert_eq!(utc_format("%b %d, %Y", D_2024_01_15), "Jan 15, 2024");
        assert_eq!(utc_format("%b %d, %Y", D_2020_END), "Dec 31, 2020");
        assert_eq!(utc_format("%H:%M", D_2024_01_15), "00:00");
        assert_eq!(utc_format("%H:%M", D_MID), "13:45");
        assert_eq!(utc_format("%I:%M %p", D_2024_01_15), "12:00 AM");
        assert_eq!(utc_format("%I:%M %p", D_MID), "01:45 PM");
        assert_eq!(utc_format("%I:%M %p", D_2020_END), "11:59 PM");
        assert_eq!(utc_format("%B %Y", D_2024_01_15), "January 2024");
        assert_eq!(utc_format("%B %Y", D_MID), "June 2024");
        assert_eq!(utc_format("%j", D_2024_01_15), "015");
        assert_eq!(utc_format("%j", D_MID), "167");
        assert_eq!(utc_format("%j", D_2020_END), "366"); // 2020 leap year
        assert_eq!(utc_format("%a", D_2024_01_15), "Mon");
        assert_eq!(utc_format("%A", D_2024_01_15), "Monday");
        assert_eq!(utc_format("%A", D_MID), "Saturday");
        assert_eq!(utc_format(".%L", D_MID), ".250");
        assert_eq!(utc_format(".%L", D_2024_01_15), ".000");
        assert_eq!(utc_format("%y", D_2024_01_15), "24");
        assert_eq!(utc_format("%y", D_2020_END), "20");
    }

    #[test]
    fn dash_modifier_unpads() {
        // node: utcFormat("%-m/%-d/%y") → "1/15/24"
        assert_eq!(utc_format("%-m/%-d/%y", D_2024_01_15), "1/15/24");
        assert_eq!(utc_format("%-m/%-d/%y", D_MID), "6/15/24");
        assert_eq!(utc_format("%-m/%-d/%y", D_2020_END), "12/31/20");
    }

    #[test]
    fn multiscale_labels_match_d3() {
        // From scaleUtc().tickFormat() over the niced ranges (oracle):
        // year/month boundaries
        assert_eq!(utc_multiscale_format(1_704_067_200_000), "2024"); // Jan 1 → year
        assert_eq!(utc_multiscale_format(1_706_745_600_000), "February"); // Feb 1
        assert_eq!(utc_multiscale_format(1_709_251_200_000), "March"); // Mar 1
                                                                       // single-day hour ticks
        assert_eq!(utc_multiscale_format(1_705_276_800_000), "Mon 15"); // day boundary
        assert_eq!(utc_multiscale_format(1_705_287_600_000), "03 AM"); // hour
        assert_eq!(utc_multiscale_format(1_705_320_000_000), "12 PM"); // noon
                                                                       // week vs day: Jan 14 2024 is a Sunday (week boundary) → "%b %d"
        assert_eq!(utc_multiscale_format(1_705_190_400_000), "Jan 14");
        // minutes / seconds
        assert_eq!(utc_multiscale_format(1_705_320_900_000), "12:15"); // 12:15 → minute
        assert_eq!(utc_multiscale_format(1_705_320_015_000), ":15"); // :15 → second
                                                                     // sub-second → millisecond
        assert_eq!(utc_multiscale_format(1_705_320_000_250), ".250");
    }

    #[test]
    fn negative_year_truncating_modulo() {
        // node: -0005-06-15T12:00Z (ms = -62310686400000) → %Y "-0005", %y "-05"
        // (JS truncating %, not euclidean), %B %d "June 15".
        const Y_NEG5: i64 = -62_310_686_400_000;
        assert_eq!(utc_format("%Y", Y_NEG5), "-0005");
        assert_eq!(utc_format("%y", Y_NEG5), "-05");
        assert_eq!(utc_format("%B %d", Y_NEG5), "June 15");
    }

    #[test]
    fn week_iso_unix_and_composite_directives_match_d3() {
        // node + d3-time-format@4.1.0 for 2024-01-15T00:00:00Z (Mon):
        assert_eq!(utc_format("%U", D_2024_01_15), "02"); // week of year (Sun)
        assert_eq!(utc_format("%W", D_2024_01_15), "03"); // week of year (Mon)
        assert_eq!(utc_format("%V", D_2024_01_15), "03"); // ISO week
        assert_eq!(utc_format("%G", D_2024_01_15), "2024"); // ISO year
        assert_eq!(utc_format("%g", D_2024_01_15), "24"); // ISO year (2-digit)
        assert_eq!(utc_format("%Q", D_2024_01_15), "1705276800000"); // unix ms
        assert_eq!(utc_format("%s", D_2024_01_15), "1705276800"); // unix seconds
        assert_eq!(utc_format("%u", D_2024_01_15), "1"); // ISO weekday (Mon=1)
        assert_eq!(utc_format("%w", D_2024_01_15), "1"); // weekday (Sun=0)
        assert_eq!(utc_format("%q", D_2024_01_15), "1"); // quarter
                                                         // Locale composites (en-US default): date / time / datetime.
        assert_eq!(utc_format("%x", D_MID), "6/15/2024");
        assert_eq!(utc_format("%X", D_MID), "1:45:30 PM");
        assert_eq!(utc_format("%c", D_MID), "6/15/2024, 1:45:30 PM");
        // 2020-12-31 is ISO week 53 of 2020.
        assert_eq!(utc_format("%V", D_2020_END), "53");
    }

    #[test]
    fn unknown_directive_emits_literal() {
        // A genuinely-unmapped directive char (not in d3's utcFormats) emits
        // the literal char; the leading '%' is dropped (d3 parity).
        assert_eq!(utc_format("%C", D_2024_01_15), "C");
        assert_eq!(utc_format("%n", D_2024_01_15), "n");
        // literal percent
        assert_eq!(utc_format("100%%", D_2024_01_15), "100%");
    }
}
