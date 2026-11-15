//! Raw row-array → `ParsedData` coercion — the native port of
//! `rowsToParsedChartData` (`utils/markup/generatedSourceShorthand/chartDataRows.ts`)
//! plus `parseDateString` (`utils/markup/elements/charts/data.ts:97-299`).
//!
//! Agents sometimes write the "promoted" nested chart form with a raw row
//! array (`generatedSource: { data: [{ category, value }, ...] }`). The TS
//! lane coerces that shape into the correlated `{ columns, rows, series,
//! categories, valueRange }` bundle at node hydration
//! (`createGeneratedGraphicNode` → `normalizeGeneratedGraphicSource`,
//! nodes.ts:1945-1975) and renders a real chart. Before this port the native
//! lane preserved the raw array, read it as empty data, and drew a ghost
//! chart (title/subtitle only) — the one chart shape where the browser
//! genuinely out-rendered native (`generated:chart` / `raw-array-data`).
//!
//! Fail-closed contract: everywhere the TS THROWS (missing value/category
//! columns, an uncoercible value cell) this port returns `None`, and the
//! migration keeps the raw array so `chart_gate_reason` still gates the doc
//! to the browser instead of silently rendering something wrong.
//!
//! JS semantics are copied faithfully ("copy the bugs", PLAN §0 rule 3):
//! `parseFloat` prefix parsing, `String(...)` coercion, first-seen column
//! order, `[min(0,…), max(0,…)]` value ranges, time-mode row drops with
//! 1-indexed `parseErrors`, and the deliberate `parseDateString` quirks
//! (UTC-only math, `Date.UTC` 0-99 year-remap bypass, ISO round-trip
//! overflow rejection, the fixed d3 human-readable format list).

use crate::types::JsonMap;
use crate::value::{Map, Value};

use super::jsmath::js_num_to_string;

const STANDARD_COLUMNS: [&str; 7] = ["category", "series", "value", "x", "y", "size", "label"];

/// Emit a finite f64 the way `JSON.stringify` would: integral safe values as
/// integer JSON numbers (JS has no int/float distinction; `70` never prints
/// as `70.0`).
fn json_num(x: f64) -> Value {
    if x.is_finite() && x.fract() == 0.0 && x.abs() < 9_007_199_254_740_992.0 {
        Value::from(x as i64)
    } else {
        Value::from(x)
    }
}

/// JS property read on a row that may not be an object. Arrays read numeric
/// index keys (JS arrays are objects with index properties); every other
/// non-object reads `undefined` (`None`).
fn row_get<'a>(row: &'a Value, key: &str) -> Option<&'a Value> {
    match row {
        Value::Object(m) => m.get(key),
        Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get(i)),
        _ => None,
    }
}

/// `String(value)` — JS ToString for the JSON value domain.
fn js_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => js_num_to_string(n.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::Null => String::new(),
                other => js_to_string(other),
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".to_string(),
    }
}

/// JS `parseFloat` — trims leading whitespace, parses the longest numeric
/// prefix (sign, decimal, exponent, `Infinity`), NaN when none.
fn js_parse_float(s: &str) -> f64 {
    let t = s.trim_start();
    let bytes = t.as_bytes();
    let mut i = 0usize;
    let mut sign = 1.0f64;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        if bytes[i] == b'-' {
            sign = -1.0;
        }
        i += 1;
    }
    if t[i..].starts_with("Infinity") {
        return sign * f64::INFINITY;
    }
    let start = i;
    let mut seen_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        seen_digit = true;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            seen_digit = true;
        }
    }
    if !seen_digit {
        return f64::NAN;
    }
    let mut end = i;
    // optional exponent — only valid with at least one digit after e[+-]?
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let exp_digits_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > exp_digits_start {
            end = j;
        }
    }
    let mantissa = &t[start..end];
    match mantissa.parse::<f64>() {
        Ok(v) => sign * v,
        Err(_) => f64::NAN,
    }
}

/// `toNumeric` (chartDataRows.ts:46-53) — `None` where the TS throws.
fn to_numeric(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(n)) => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            if f.is_finite() {
                return Some(f);
            }
            None
        }
        Some(Value::String(s)) => {
            let parsed = js_parse_float(s);
            if parsed.is_finite() {
                return Some(parsed);
            }
            None
        }
        _ => None,
    }
}

/// `toTime` (chartDataRows.ts:55-59).
fn to_time(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(n)) => n.as_f64().filter(|f| f.is_finite()),
        Some(Value::String(s)) => parse_date_string(s),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// parseDateString (utils/markup/elements/charts/data.ts:97-299) — UTC-only
// date parsing with deliberate JS quirks preserved.
// ---------------------------------------------------------------------------

/// Days from 1970-01-01 for a civil date (proleptic Gregorian; Howard
/// Hinnant's `days_from_civil`). `d` may exceed the month length — overflow
/// rolls forward exactly like `Date.UTC` does.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    // Normalize to day 1, then add the day offset so overflow rolls.
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let m0 = if m > 2 { m - 3 } else { m + 9 }; // Mar=0 … Feb=11
    let doy = (153 * m0 + 2) / 5; // day-of-year for day 1 of month
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146097 + doe - 719468 + (d - 1)
}

/// Civil date (y, m 1-12, d 1-31) from days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// UTC milliseconds for civil components; day overflow rolls forward like
/// `Date.UTC` (month must already be 1-12). No 0-99 year remap — every TS
/// call site that could see such years bypasses the remap via
/// `setUTCFullYear`, so true-year math is the correct mirror.
fn utc_ms(y: i64, m: i64, d: i64, h: i64, min: i64, s: i64, ms: i64) -> f64 {
    let days = days_from_civil(y, m, d);
    (days * 86_400_000 + h * 3_600_000 + min * 60_000 + s * 1000 + ms) as f64
}

fn all_ascii_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

fn parse_2(s: &[u8], i: usize) -> i64 {
    ((s[i] - b'0') as i64) * 10 + (s[i + 1] - b'0') as i64
}

/// The ISO family: prefix `YYYY-MM-DD` followed by end-of-string or `T`.
/// Mirrors data.ts:141-257 (round-trip overflow rejection, UTC-only frames).
fn parse_iso_family(t: &str) -> Option<Option<f64>> {
    // Outer None = "not ISO-shaped, fall through to human formats";
    // Some(None) = ISO-shaped but invalid (TS returns null).
    let b = t.as_bytes();
    let iso_prefix = b.len() >= 10
        && b[..4].iter().all(u8::is_ascii_digit)
        && b[4] == b'-'
        && b[5..7].iter().all(u8::is_ascii_digit)
        && b[7] == b'-'
        && b[8..10].iter().all(u8::is_ascii_digit)
        && (b.len() == 10 || b[10] == b'T');
    if !iso_prefix {
        return None;
    }
    let y: i64 = t[..4].parse().ok()?;
    let m = parse_2(b, 5);
    let d = parse_2(b, 8);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Some(None);
    }
    if b.len() == 10 {
        // Pure `YYYY-MM-DD` → utcParse('%Y-%m-%d'), UTC frame. Overflow
        // (e.g. 2023-02-29) rolls forward, then the round-trip check
        // rejects it.
        let ms = utc_ms(y, m, d, 0, 0, 0, 0);
        let (ya, ma, da) = civil_from_days(days_from_civil(y, m, d));
        if ya != y || ma != m || da != d {
            return Some(None);
        }
        return Some(Some(ms));
    }

    // Time component present. Split off a trailing offset if any
    // (`Z` | `±hh:mm` | `±hhmm`), matching /(Z|[+-]\d{2}:?\d{2})$/.
    let rest = &t[11..]; // after 'T'
    let (time_part, offset_minutes, has_offset) = split_offset(rest)?;

    // hh:mm[:ss[.frac]] — frac 1-3 digits on the bare-time branch
    // (data.ts:192), 1+ digits of ANY length (truncated to ms, V8-style —
    // `Date.parse('…00.1234567890123Z')` keeps the first three) on the offset
    // branch (isoParse → native Date parsing).
    let max_frac = if has_offset { usize::MAX } else { 3 };
    let (h, min, s, ms) = parse_time_components(time_part, max_frac)?;
    if h > 23 || min > 59 || s > 59 {
        return Some(None);
    }
    let utc = utc_ms(y, m, d, h, min, s, ms);
    let instant = utc - (offset_minutes as f64) * 60_000.0;
    // Round-trip: extract Y/M/D in the input's claimed frame (UTC for bare
    // time, wall-time at the claimed offset otherwise) and compare.
    let claimed_days = days_from_civil(y, m, d);
    let (ya, ma, da) = civil_from_days(claimed_days);
    if ya != y || ma != m || da != d {
        return Some(None);
    }
    Some(Some(instant))
}

/// Split a `hh:mm…` tail into (time, offset minutes, had_offset). Returns
/// `None` for a malformed tail (TS: regex fails → parse null), INCLUDING an
/// out-of-range offset: V8 rejects offset hours > 23 or minutes > 59
/// (`Date.parse('…+24:00')` and `'…+23:99'` are both NaN; `'…+23:59'` is
/// valid), and the offset branch mirrors isoParse → native Date parsing.
fn split_offset(rest: &str) -> Option<(&str, i64, bool)> {
    if let Some(stripped) = rest.strip_suffix('Z') {
        return Some((stripped, 0, true));
    }
    let b = rest.as_bytes();
    // ±hh:mm
    if b.len() >= 6 {
        let i = b.len() - 6;
        if (b[i] == b'+' || b[i] == b'-')
            && b[i + 1].is_ascii_digit()
            && b[i + 2].is_ascii_digit()
            && b[i + 3] == b':'
            && b[i + 4].is_ascii_digit()
            && b[i + 5].is_ascii_digit()
        {
            let sign = if b[i] == b'+' { 1 } else { -1 };
            let h = parse_2(b, i + 1);
            let m = parse_2(b, i + 4);
            if h > 23 || m > 59 {
                return None;
            }
            return Some((&rest[..i], sign * (h * 60 + m), true));
        }
    }
    // ±hhmm
    if b.len() >= 5 {
        let i = b.len() - 5;
        if (b[i] == b'+' || b[i] == b'-') && b[i + 1..].iter().all(u8::is_ascii_digit) {
            let sign = if b[i] == b'+' { 1 } else { -1 };
            let h = parse_2(b, i + 1);
            let m = parse_2(b, i + 3);
            if h > 23 || m > 59 {
                return None;
            }
            return Some((&rest[..i], sign * (h * 60 + m), true));
        }
    }
    Some((rest, 0, false))
}

/// `hh:mm[:ss[.frac]]` with strict two-digit fields; frac 1..=max_frac
/// digits, truncated/padded to milliseconds.
fn parse_time_components(t: &str, max_frac: usize) -> Option<(i64, i64, i64, i64)> {
    let b = t.as_bytes();
    if b.len() < 5
        || !b[0].is_ascii_digit()
        || !b[1].is_ascii_digit()
        || b[2] != b':'
        || !b[3].is_ascii_digit()
        || !b[4].is_ascii_digit()
    {
        return None;
    }
    let h = parse_2(b, 0);
    let min = parse_2(b, 3);
    if b.len() == 5 {
        return Some((h, min, 0, 0));
    }
    if b[5] != b':' || b.len() < 8 || !b[6].is_ascii_digit() || !b[7].is_ascii_digit() {
        return None;
    }
    let s = parse_2(b, 6);
    if b.len() == 8 {
        return Some((h, min, s, 0));
    }
    if b[8] != b'.' {
        return None;
    }
    let frac = &t[9..];
    if frac.is_empty() || frac.len() > max_frac || !all_ascii_digits(frac) {
        return None;
    }
    let mut ms_str: String = frac.chars().take(3).collect();
    while ms_str.len() < 3 {
        ms_str.push('0');
    }
    let ms: i64 = ms_str.parse().ok()?;
    Some((h, min, s, ms))
}

const MONTHS_FULL: [&str; 12] = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];
const MONTHS_ABBR: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

#[derive(Clone, Copy)]
enum Tok {
    MonthFull,
    MonthAbbr,
    Day,  // %d / %e — 1-2 digits
    Year, // %Y — exactly 4 digits
    Lit(char),
}

/// One d3 `utcParse` pattern over the input; strict literal matching, full
/// consumption, case-insensitive month names — mirroring d3-time-format's
/// parseSpecifier. Returns (month 1-12, day, year).
fn try_pattern(input: &str, toks: &[Tok]) -> Option<(i64, i64, i64)> {
    let mut month: i64 = 1;
    let mut day: i64 = 1; // d3 default d = 1
    let mut year: i64 = 1900; // d3 default y = 1900 (always overwritten: every pattern has %Y)
    let mut rest = input;
    for tok in toks {
        match tok {
            Tok::MonthFull | Tok::MonthAbbr => {
                let names: &[&str; 12] = if matches!(tok, Tok::MonthFull) {
                    &MONTHS_FULL
                } else {
                    &MONTHS_ABBR
                };
                let lower = rest.to_lowercase();
                let idx = names.iter().position(|name| lower.starts_with(name))?;
                month = idx as i64 + 1;
                rest = &rest[names[idx].len()..];
            }
            Tok::Day => {
                let n = rest.bytes().take(2).take_while(u8::is_ascii_digit).count();
                if n == 0 {
                    return None;
                }
                day = rest[..n].parse().ok()?;
                rest = &rest[n..];
            }
            Tok::Year => {
                if rest.len() < 4 || !rest.as_bytes()[..4].iter().all(u8::is_ascii_digit) {
                    return None;
                }
                year = rest[..4].parse().ok()?;
                rest = &rest[4..];
            }
            Tok::Lit(c) => {
                rest = rest.strip_prefix(*c)?;
            }
        }
    }
    if !rest.is_empty() {
        return None; // d3: the entire string must be consumed
    }
    Some((month, day, year))
}

/// The `HUMAN_DATE_FORMATS` list (data.ts:58-75), reduced: `%e` parses
/// identically to `%d` in d3 (both accept 1-2 digits), so the `%e` variants
/// can never match anything the preceding `%d` pattern missed — first-match
/// semantics are preserved without them.
fn human_patterns() -> Vec<Vec<Tok>> {
    use Tok::*;
    vec![
        // "%B %d, %Y" — "January 15, 2024"
        vec![MonthFull, Lit(' '), Day, Lit(','), Lit(' '), Year],
        // "%B %d %Y"
        vec![MonthFull, Lit(' '), Day, Lit(' '), Year],
        // "%b %d, %Y"
        vec![MonthAbbr, Lit(' '), Day, Lit(','), Lit(' '), Year],
        // "%b %d %Y"
        vec![MonthAbbr, Lit(' '), Day, Lit(' '), Year],
        // "%d %B %Y"
        vec![Day, Lit(' '), MonthFull, Lit(' '), Year],
        // "%d %b %Y"
        vec![Day, Lit(' '), MonthAbbr, Lit(' '), Year],
        // "%B %Y"
        vec![MonthFull, Lit(' '), Year],
        // "%b %Y"
        vec![MonthAbbr, Lit(' '), Year],
    ]
}

/// `parseDateString` (data.ts:97-299). Returns UTC ms or `None`, matching the
/// TS `number | null` contract value-for-value (probe-verified against the
/// production TS on the full quirk matrix — see the module tests).
pub fn parse_date_string(raw: &str) -> Option<f64> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }

    // Bare year "2024" → Jan 1 UTC (setUTCFullYear semantics: no 0-99 remap).
    if t.len() == 4 && all_ascii_digits(t) {
        let year: i64 = t.parse().ok()?;
        return Some(utc_ms(year, 1, 1, 0, 0, 0, 0));
    }
    // Bare year-month "2024-01" → 1st of month UTC; month validated.
    if t.len() == 7 && all_ascii_digits(&t[..4]) && &t[4..5] == "-" && all_ascii_digits(&t[5..]) {
        let y: i64 = t[..4].parse().ok()?;
        let m: i64 = t[5..].parse().ok()?;
        if !(1..=12).contains(&m) {
            return None;
        }
        return Some(utc_ms(y, m, 1, 0, 0, 0, 0));
    }
    // Purely-numeric non-year inputs rejected (d3 year-2000 anchor trap).
    if all_ascii_digits(t) {
        return None;
    }

    if let Some(result) = parse_iso_family(t) {
        return result;
    }

    // Human-readable fallback: fixed d3 pattern list, first match wins.
    let mut parsed: Option<(i64, i64, i64)> = None;
    for pattern in human_patterns() {
        if let Some(p) = try_pattern(t, &pattern) {
            parsed = Some(p);
            break;
        }
    }
    let (m, d, y) = parsed?;
    let days = days_from_civil(y, m, d); // overflow rolls forward, like Date.UTC
    let (_, _, day_actual) = civil_from_days(days);

    // Day-overflow round-trip: exactly one standalone 1-2 digit number in
    // 1-31 is the user's literal day claim (data.ts:287-296).
    let mut candidates: Vec<i64> = Vec::new();
    for run in t.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if (1..=2).contains(&run.len()) && all_ascii_digits(run) {
            let n: i64 = run.parse().unwrap();
            if (1..=31).contains(&n) {
                candidates.push(n);
            }
        }
    }
    if candidates.len() == 1 && candidates[0] != day_actual {
        return None;
    }
    Some((days * 86_400_000) as f64)
}

// ---------------------------------------------------------------------------
// rowsToParsedChartData (chartDataRows.ts:61-279)
// ---------------------------------------------------------------------------

/// Port of `rowsToParsedChartData`. `time_mode` mirrors
/// `options.xScale === 'time'`. Returns `None` exactly where the TS throws
/// (missing required columns, uncoercible value/x/y/size cells) so the
/// caller can fail closed — keep the raw array and let `chart_gate_reason`
/// gate the doc to the browser rather than render wrong data.
pub fn rows_to_parsed_chart_data(raw_rows: &[Value], time_mode: bool) -> Option<JsonMap> {
    if raw_rows.is_empty() {
        // "clear the chart" — empty bundle, not a failure.
        let mut out = Map::new();
        out.insert("columns".into(), Value::Array(vec![]));
        out.insert("rows".into(), Value::Array(vec![]));
        out.insert("series".into(), Value::Array(vec![]));
        out.insert("categories".into(), Value::Array(vec![]));
        out.insert(
            "valueRange".into(),
            Value::Array(vec![Value::from(0), Value::from(0)]),
        );
        return Some(out);
    }

    // Column discovery, first-seen order. JS `Object.keys` on an array row
    // yields its index strings; non-objects contribute nothing.
    let mut column_order: Vec<String> = Vec::new();
    for row in raw_rows {
        match row {
            Value::Object(m) => {
                for key in m.keys() {
                    if !column_order.iter().any(|c| c == key) {
                        column_order.push(key.to_string());
                    }
                }
            }
            Value::Array(a) => {
                for i in 0..a.len() {
                    let key = i.to_string();
                    if !column_order.iter().any(|c| c == &key) {
                        column_order.push(key);
                    }
                }
            }
            _ => {}
        }
    }

    let has = |name: &str| column_order.iter().any(|c| c == name);
    let has_x = has("x");
    let has_y = has("y");
    let has_category = has("category");
    let has_series = has("series");
    let has_value = has("value");
    let has_size = has("size");
    let has_label = has("label");
    let is_scatter = has_x && has_y;
    // Time column resolution (chartDataRows.ts:106-114).
    #[derive(PartialEq, Clone, Copy)]
    enum TimeCol {
        X,
        Category,
        Series,
        None,
    }
    let time_column = if time_mode {
        if is_scatter {
            TimeCol::X
        } else if has_category {
            TimeCol::Category
        } else if has_series {
            TimeCol::Series
        } else {
            TimeCol::None
        }
    } else {
        TimeCol::None
    };

    if !is_scatter && (!has_value || (!has_category && !has_series)) {
        return None; // TS throws
    }

    let mut rows: Vec<Value> = Vec::new();
    let mut series_list: Vec<String> = Vec::new();
    let mut categories_list: Vec<String> = Vec::new();
    let mut parse_errors: Vec<String> = Vec::new();

    let mut value_min = f64::INFINITY;
    let mut value_max = f64::NEG_INFINITY;
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut size_min = f64::INFINITY;
    let mut size_max = f64::NEG_INFINITY;
    let mut time_min = f64::INFINITY;
    let mut time_max = f64::NEG_INFINITY;

    for (i, row) in raw_rows.iter().enumerate() {
        // Time-mode pre-check: drop rows with unparseable dates, recording a
        // 1-indexed diagnostic (chartDataRows.ts:150-163).
        let mut row_time: Option<f64> = None;
        if time_column != TimeCol::None {
            let key = match time_column {
                TimeCol::X => "x",
                TimeCol::Category => "category",
                TimeCol::Series => "series",
                TimeCol::None => unreachable!(),
            };
            let raw_date = row_get(row, key);
            row_time = to_time(raw_date);
            if row_time.is_none() {
                let rendered = match raw_date {
                    Some(Value::String(s)) => s.clone(),
                    // `String(rawDate ?? '')` — null/undefined render as ''.
                    Some(Value::Null) | None => String::new(),
                    Some(other) => js_to_string(other),
                };
                parse_errors.push(format!(
                    "Invalid date \"{rendered}\" in chart data row {}; row skipped.",
                    i + 1
                ));
                continue;
            }
        }

        // Point literal `{ category: '', value: 0 }` — those two keys always
        // lead, in this order (JS insertion-order preservation).
        let mut point = Map::new();
        point.insert("category".into(), Value::String(String::new()));
        point.insert("value".into(), Value::from(0));

        if has_category {
            let cat = match row_get(row, "category") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Null) | None => String::new(), // String(x ?? '')
                Some(other) => js_to_string(other),
            };
            if !cat.is_empty() && !categories_list.iter().any(|c| c == &cat) {
                categories_list.push(cat.clone());
            }
            point.insert("category".into(), Value::String(cat));
            if time_column == TimeCol::Category {
                let rt = row_time.unwrap();
                point.insert("xTime".into(), json_num(rt));
                time_min = time_min.min(rt);
                time_max = time_max.max(rt);
            }
        }

        if has_series {
            if let Some(Value::String(s)) = row_get(row, "series") {
                if !series_list.iter().any(|x| x == s) {
                    series_list.push(s.clone());
                }
                point.insert("series".into(), Value::String(s.clone()));
            }
            if time_column == TimeCol::Series {
                let rt = row_time.unwrap();
                point.insert("xTime".into(), json_num(rt));
                time_min = time_min.min(rt);
                time_max = time_max.max(rt);
            }
        }

        if is_scatter {
            let x = if time_column == TimeCol::X {
                row_time.unwrap()
            } else {
                to_numeric(row_get(row, "x"))?
            };
            let y = to_numeric(row_get(row, "y"))?;
            point.insert("x".into(), json_num(x));
            point.insert("y".into(), json_num(y));
            point.insert("value".into(), json_num(y));
            if time_column == TimeCol::X {
                let rt = row_time.unwrap();
                point.insert("xTime".into(), json_num(rt));
                time_min = time_min.min(rt);
                time_max = time_max.max(rt);
            }
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_min = y_min.min(y);
            y_max = y_max.max(y);
            value_min = value_min.min(y);
            value_max = value_max.max(y);
            if has_size {
                if let Some(size_raw) = row_get(row, "size") {
                    let size = to_numeric(Some(size_raw))?;
                    point.insert("size".into(), json_num(size));
                    size_min = size_min.min(size);
                    size_max = size_max.max(size);
                }
            }
        } else if has_value {
            let value = to_numeric(row_get(row, "value"))?;
            point.insert("value".into(), json_num(value));
            value_min = value_min.min(value);
            value_max = value_max.max(value);
        }

        if has_label {
            if let Some(label) = row_get(row, "label") {
                // `typeof label === 'string' ? label : String(label)` — note
                // null renders "null" here (no ?? '' in the TS).
                let s = match label {
                    Value::String(s) => s.clone(),
                    other => js_to_string(other),
                };
                point.insert("label".into(), Value::String(s));
            }
        }

        // Non-standard columns → metadata (numbers coerced when possible).
        for key in &column_order {
            if STANDARD_COLUMNS.contains(&key.as_str()) {
                continue;
            }
            let Some(raw_meta) = row_get(row, key) else {
                continue; // undefined
            };
            let as_number = match raw_meta {
                Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                Value::String(s) => js_parse_float(s),
                _ => f64::NAN,
            };
            let entry = if as_number.is_finite() {
                json_num(as_number)
            } else {
                raw_meta.clone()
            };
            point
                .entry("metadata")
                .or_insert_with(|| Value::Object(Map::new()))
                .as_object_mut()
                .unwrap()
                .insert(key.clone().into(), entry);
        }

        rows.push(Value::Object(point));
    }

    let resolved_min = if value_min.is_finite() {
        value_min.min(0.0)
    } else {
        0.0
    };
    let resolved_max = if value_max.is_finite() {
        value_max.max(0.0)
    } else {
        0.0
    };

    let mut out = Map::new();
    out.insert(
        "columns".into(),
        Value::Array(
            column_order
                .iter()
                .map(|c| Value::String(c.clone()))
                .collect(),
        ),
    );
    out.insert("rows".into(), Value::Array(rows));
    out.insert(
        "series".into(),
        Value::Array(series_list.into_iter().map(Value::String).collect()),
    );
    out.insert(
        "categories".into(),
        Value::Array(categories_list.into_iter().map(Value::String).collect()),
    );
    out.insert(
        "valueRange".into(),
        Value::Array(vec![json_num(resolved_min), json_num(resolved_max)]),
    );

    if is_scatter {
        out.insert("isScatterFormat".into(), Value::Bool(true));
        if x_min.is_finite() && x_max.is_finite() {
            out.insert(
                "xRange".into(),
                Value::Array(vec![json_num(x_min), json_num(x_max)]),
            );
        }
        if y_min.is_finite() && y_max.is_finite() {
            out.insert(
                "yRange".into(),
                Value::Array(vec![json_num(y_min), json_num(y_max)]),
            );
        }
        if size_min.is_finite() && size_max.is_finite() {
            out.insert(
                "sizeRange".into(),
                Value::Array(vec![json_num(size_min), json_num(size_max)]),
            );
        }
    }

    if time_mode && time_min.is_finite() && time_max.is_finite() {
        out.insert(
            "timeRange".into(),
            Value::Array(vec![json_num(time_min), json_num(time_max)]),
        );
    }

    if !parse_errors.is_empty() {
        out.insert(
            "parseErrors".into(),
            Value::Array(parse_errors.into_iter().map(Value::String).collect()),
        );
    }

    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    fn rows(v: Value) -> Vec<Value> {
        v.as_array().unwrap().clone()
    }

    // ------ rowsToParsedChartData (mirrors chartDataRows.test.ts) ------

    #[test]
    fn categorical_bundle_from_category_value_rows() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "Q1", "value": 70},
                {"category": "Q2", "value": 80},
                {"category": "Q3", "value": 90},
                {"category": "Q4", "value": 75}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(out["columns"], json!(["category", "value"]));
        assert_eq!(
            out["rows"],
            json!([
                {"category": "Q1", "value": 70},
                {"category": "Q2", "value": 80},
                {"category": "Q3", "value": 90},
                {"category": "Q4", "value": 75}
            ])
        );
        assert_eq!(out["series"], json!([]));
        assert_eq!(out["categories"], json!(["Q1", "Q2", "Q3", "Q4"]));
        assert_eq!(out["valueRange"], json!([0, 90]));
    }

    #[test]
    fn multi_series_bundle() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "Q1", "series": "Products", "value": 180},
                {"category": "Q1", "series": "Services", "value": 75},
                {"category": "Q2", "series": "Products", "value": 210},
                {"category": "Q2", "series": "Services", "value": 90}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(out["columns"], json!(["category", "series", "value"]));
        assert_eq!(out["series"], json!(["Products", "Services"]));
        assert_eq!(out["categories"], json!(["Q1", "Q2"]));
        assert_eq!(out["valueRange"], json!([0, 210]));
        assert_eq!(out["rows"].as_array().unwrap().len(), 4);
        // Row key order matches the JS insertion order.
        assert_eq!(
            out["rows"][0],
            json!({"category": "Q1", "value": 180, "series": "Products"})
        );
        assert_eq!(
            crate::value::to_string(&out["rows"][0]).unwrap(),
            r#"{"category":"Q1","value":180,"series":"Products"}"#
        );
    }

    #[test]
    fn scatter_bundle() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([{"x": 1, "y": 10}, {"x": 2, "y": 30}, {"x": 4, "y": 20}])),
            false,
        )
        .unwrap();
        assert_eq!(out["columns"], json!(["x", "y"]));
        assert_eq!(out["isScatterFormat"], json!(true));
        assert_eq!(out["xRange"], json!([1, 4]));
        assert_eq!(out["yRange"], json!([10, 30]));
        assert_eq!(out["valueRange"], json!([0, 30]));
    }

    #[test]
    fn scatter_size_column() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"x": 1, "y": 10, "size": 20},
                {"x": 2, "y": 30, "size": 60},
                {"x": 4, "y": 20, "size": 40}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(out["columns"], json!(["x", "y", "size"]));
        assert_eq!(out["sizeRange"], json!([20, 60]));
    }

    #[test]
    fn label_column_preserved() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "Q1", "value": 180000, "label": "$180K"},
                {"category": "Q2", "value": 210000, "label": "$210K"}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(out["columns"], json!(["category", "value", "label"]));
        assert_eq!(out["rows"][0]["label"], json!("$180K"));
        assert_eq!(out["rows"][0]["value"], json!(180000));
    }

    #[test]
    fn column_union_first_seen_order() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "Q1", "value": 70},
                {"category": "Q2", "value": 80, "label": "big"}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(out["columns"], json!(["category", "value", "label"]));
    }

    #[test]
    fn fails_closed_when_required_columns_missing() {
        // TS throws for these → the port returns None.
        assert!(rows_to_parsed_chart_data(&rows(json!([{"foo": 1, "bar": 2}])), false).is_none());
        // Capitalized keys are non-standard → same fail-closed outcome
        // (mirrors the assess.rs raw-array gating test's doc shape).
        assert!(
            rows_to_parsed_chart_data(&rows(json!([{"Category": "A", "Value": 1}])), false)
                .is_none()
        );
    }

    #[test]
    fn fails_closed_on_uncoercible_value_cell() {
        assert!(rows_to_parsed_chart_data(
            &rows(json!([{"category": "Q1", "value": "not-a-number"}])),
            false
        )
        .is_none());
        assert!(rows_to_parsed_chart_data(
            &rows(json!([{"category": "Q1", "value": null}])),
            false
        )
        .is_none());
        assert!(rows_to_parsed_chart_data(&rows(json!([{"x": 1, "y": true}])), false).is_none());
    }

    #[test]
    fn coerces_numeric_string_values() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "Q1", "value": "70"},
                {"category": "Q2", "value": "80.5"}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(out["rows"][0]["value"], json!(70));
        assert_eq!(out["rows"][1]["value"], json!(80.5));
        assert_eq!(out["valueRange"], json!([0, 80.5]));
    }

    #[test]
    fn time_mode_categorical() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "2024-01-01", "value": 100},
                {"category": "2024-02-01", "value": 140},
                {"category": "2024-03-01", "value": 120}
            ])),
            true,
        )
        .unwrap();
        // Date.UTC(2024, 0, 1) / Date.UTC(2024, 2, 1)
        assert_eq!(
            out["timeRange"],
            json!([1704067200000i64, 1709251200000i64])
        );
        assert_eq!(out["rows"][0]["xTime"], json!(1704067200000i64));
        assert_eq!(out["rows"][0]["category"], json!("2024-01-01"));
        assert!(out.get("parseErrors").is_none());
        // xTime slots between value and series-position (category, value, xTime).
        assert_eq!(
            crate::value::to_string(&out["rows"][0]).unwrap(),
            r#"{"category":"2024-01-01","value":100,"xTime":1704067200000}"#
        );
    }

    #[test]
    fn time_mode_scatter() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([{"x": "2024-01-01", "y": 10}, {"x": "2024-06-01", "y": 30}])),
            true,
        )
        .unwrap();
        assert_eq!(out["isScatterFormat"], json!(true));
        assert_eq!(out["rows"][0]["x"], json!(1704067200000i64));
        assert_eq!(out["rows"][0]["xTime"], json!(1704067200000i64));
        assert!(out.get("timeRange").is_some());
    }

    #[test]
    fn non_time_mode_has_no_xtime() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "2024-01-01", "value": 100},
                {"category": "2024-02-01", "value": 140}
            ])),
            false,
        )
        .unwrap();
        assert!(out.get("timeRange").is_none());
        assert!(out["rows"][0].get("xTime").is_none());
    }

    #[test]
    fn time_mode_drops_unparseable_rows_with_errors() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "2024-01-01", "value": 100},
                {"category": "TBD", "value": 999},
                {"category": "not-a-date", "value": 42},
                {"category": "2024-03-01", "value": 140}
            ])),
            true,
        )
        .unwrap();
        assert_eq!(out["rows"].as_array().unwrap().len(), 2);
        let errs = out["parseErrors"].as_array().unwrap();
        assert_eq!(errs.len(), 2);
        assert_eq!(
            errs[0],
            json!("Invalid date \"TBD\" in chart data row 2; row skipped.")
        );
        assert_eq!(
            errs[1],
            json!("Invalid date \"not-a-date\" in chart data row 3; row skipped.")
        );
    }

    #[test]
    fn empty_dataset_returns_empty_bundle() {
        let out = rows_to_parsed_chart_data(&[], false).unwrap();
        assert_eq!(
            Value::Object(out),
            json!({
                "columns": [], "rows": [], "series": [], "categories": [],
                "valueRange": [0, 0]
            })
        );
    }

    #[test]
    fn metadata_captures_non_standard_columns() {
        let out = rows_to_parsed_chart_data(
            &rows(json!([
                {"category": "Q1", "value": 1, "weight": "12.5", "note": "hi"}
            ])),
            false,
        )
        .unwrap();
        assert_eq!(
            out["rows"][0]["metadata"],
            json!({"weight": 12.5, "note": "hi"})
        );
    }

    // ------ parseDateString (probe-verified vs the production TS) ------

    #[test]
    fn parse_date_string_matches_ts_probe() {
        // Ground truth captured by executing the production parseDateString
        // (npx tsx) — see the PR body. Every value below is the TS output.
        let cases: &[(&str, Option<f64>)] = &[
            ("2024", Some(1704067200000.0)),
            ("0001", Some(-62135596800000.0)),
            ("0050", Some(-60589296000000.0)),
            ("2024-01", Some(1704067200000.0)),
            ("2024-13", None),
            ("2024-00", None),
            ("5", None),
            ("50", None),
            ("100", None),
            ("", None),
            ("2024-01-15", Some(1705276800000.0)),
            ("2023-02-29", None),
            ("0050-01-15", Some(-60588086400000.0)),
            ("2024-02-29", Some(1709164800000.0)),
            ("2024-01-15T12:00:00Z", Some(1705320000000.0)),
            ("2024-01-15T12:00:00+05:00", Some(1705302000000.0)),
            ("2024-01-15T12:00:00-0530", Some(1705339800000.0)),
            ("2024-02-31T12:00:00Z", None),
            ("2024-01-15T12:00", Some(1705320000000.0)),
            ("2024-01-15T12:00:30.5", Some(1705320030500.0)),
            ("2024-01-15T12:00:30.25", Some(1705320030250.0)),
            ("2024-01-15T12:00:30.123", Some(1705320030123.0)),
            ("2024-01-15T24:00:00Z", None),
            ("2024-01-15T23:59:60Z", None),
            ("2024-01-15T12:00:00.1234Z", Some(1705320000123.0)),
            ("2024-01-15T12:00:00.123456789Z", Some(1705320000123.0)),
            // Offset-range + long-fraction edges (probe-verified vs the
            // production TS 2026-07-17): V8 rejects offset hours > 23 /
            // minutes > 59 and accepts fractions of ANY length (ms truncation).
            ("2024-01-15T12:00:00+24:00", None),
            ("2024-01-15T12:00:00+23:99", None),
            ("2024-01-15T12:00:00+23:59", Some(1705233660000.0)),
            ("2024-01-15T12:00:00+2359", Some(1705233660000.0)),
            ("2024-01-15T12:00:00+2400", None),
            (
                "2024-01-15T12:00:00.1234567890123+05:00",
                Some(1705302000123.0),
            ),
            ("2024-01-15T12:00:00.1234567890123Z", Some(1705320000123.0)),
            ("January 15, 2024", Some(1705276800000.0)),
            ("january 15, 2024", Some(1705276800000.0)),
            ("Jan 15 2024", Some(1705276800000.0)),
            ("15 January 2024", Some(1705276800000.0)),
            ("15 Jan 2024", Some(1705276800000.0)),
            ("Jan 5, 2024", Some(1704412800000.0)),
            ("February 29, 2023", None),
            ("April 31, 2024", None),
            ("January 2024", Some(1704067200000.0)),
            ("Jan 2024", Some(1704067200000.0)),
            ("JAN 2024", Some(1704067200000.0)),
            ("Sep 2024", Some(1725148800000.0)),
            ("Sept 2024", None),
            ("not-a-date", None),
            ("TBD", None),
            ("01/02/2024", None),
            ("  2024-01-15  ", Some(1705276800000.0)),
            ("March 5 2021", Some(1614902400000.0)),
            ("5 March 2021", Some(1614902400000.0)),
            ("December 31, 1999", Some(946598400000.0)),
            ("2024-1-5", None),
            ("2024-01-15T", None),
            ("Jan  15, 2024", None),
            ("Jan 15,2024", None),
        ];
        for (input, expected) in cases {
            assert_eq!(
                parse_date_string(input),
                *expected,
                "parseDateString({input:?}) diverged from the TS"
            );
        }
    }
}
