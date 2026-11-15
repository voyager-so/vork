//! Port of `utils/markup/elements/charts/format.ts`:
//!  - the Intl.NumberFormat paths (value labels, pie labels) — en-US currency
//!    / compact / compact-currency semantics including the `minimumFraction-
//!    Digits = min(default, max)` quirk that yields "$1.0K" before the TS
//!    strips ".0";
//!  - the d3-format axis-tick paths — a faithful subset of
//!    `node_modules/d3-format/src/*` covering the 8 specifiers the chart
//!    engine instantiates (',', ',~r', '.2~s', '$,.0f', '$,~r', '$.2~s',
//!    '.0%', '.1%').

use super::ticks::TickError;
use super::jsmath::{
    format_decimal_parts, group_thousands, intl_fixed, is_integer, js_num_to_string,
    js_number_strict, to_fixed, to_precision,
};
use super::scales::generate_ticks;

pub const CHARACTER_WIDTH_FONT_SIZE_RATIO: f64 = 0.65;

/// `labelEms` (constants.ts): a label's estimated width in em, `narrow_em` a UTF-16 unit and 1 em for a wide
/// character (CJK ideographs, kana, Hangul, full-width forms).
pub fn label_ems(label: &str, narrow_em: f64) -> f64 {
    label.encode_utf16().map(|c| if is_wide_unit(c) { 1.0 } else { narrow_em }).sum()
}

fn is_wide_unit(c: u16) -> bool {
    matches!(c,
        0x1100..=0x115f
        | 0x2e80..=0x303e
        | 0x3041..=0x33ff
        | 0x3400..=0x4dbf
        | 0x4e00..=0x9fff
        | 0xa000..=0xa4cf
        | 0xac00..=0xd7a3
        | 0xf900..=0xfaff
        | 0xfe30..=0xfe4f
        | 0xff00..=0xff60
        | 0xffe0..=0xffe6)
}
pub const LABEL_WIDTH_PADDING: f64 = 10.0;

// ---------------------------------------------------------------------------
// formatNumber / formatPercent / Intl paths (format.ts:65-152)
// ---------------------------------------------------------------------------

/// `formatNumber` (format.ts:65-71).
pub fn format_number(value: f64) -> String {
    if !value.is_finite() {
        return String::new();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    if is_integer(value) {
        return js_num_to_string(value);
    }
    let fixed = to_fixed(value, 2);
    trim_trailing_zeros_regex(&fixed)
}

/// `.replace(/\.?0+$/, '')` on a toFixed(2) string.
fn trim_trailing_zeros_regex(s: &str) -> String {
    let mut end = s.len();
    let b = s.as_bytes();
    while end > 0 && b[end - 1] == b'0' {
        end -= 1;
    }
    if end > 0 && b[end - 1] == b'.' {
        end -= 1;
    }
    // The TS regex requires at least one trailing 0 to strip the dot, but
    // toFixed(2) output always ends in digits; \.?0+$ unanchored-dot is fine.
    if end == s.len() {
        s.to_string()
    } else {
        // regex /\.?0+$/ only strips when the suffix is zeros (with optional
        // preceding dot) — the scan above implements exactly that.
        s[..end].to_string()
    }
}

/// `formatPercent` (format.ts:76-79).
pub fn format_percent(value: f64) -> String {
    let percent_value = if value.abs() <= 1.0 {
        value * 100.0
    } else {
        value
    };
    format!("{}%", format_number(percent_value))
}

/// `formatCurrency` (format.ts:84-93) — Intl en-US USD, max 2 fraction
/// digits (min 2 by currency default), then strip a trailing ".00".
pub fn format_currency(value: f64) -> String {
    let s = intl_currency_usd(value);
    s.strip_suffix(".00").map(str::to_string).unwrap_or(s)
}

fn intl_currency_usd(value: f64) -> String {
    if value.is_nan() {
        return "$NaN".to_string();
    }
    let neg = value < 0.0;
    let fixed = intl_fixed(value.abs(), 0, 2); // halfExpand on the shortest decimal, as ICU
    let (int_part, frac_part) = fixed.split_once('.').unwrap_or((fixed.as_str(), ""));
    let grouped = group_thousands(int_part);
    let body = if frac_part.is_empty() {
        grouped
    } else {
        format!("{grouped}.{frac_part}")
    };
    if neg {
        format!("-${body}")
    } else {
        format!("${body}")
    }
}

/// Intl compact short, en-US. `min_frac_1` mirrors the currency variant's
/// effective `minimumFractionDigits = 1` (so "1.0K"); the plain variant uses
/// `minimumFractionDigits = 0` (so "1K").
fn intl_compact(value: f64, min_frac_1: bool) -> String {
    if value.is_nan() {
        return "NaN".to_string();
    }
    let neg = value < 0.0;
    let a = value.abs();
    const UNITS: [(f64, i32, &str); 4] =
        [(1e12, 12, "T"), (1e9, 9, "B"), (1e6, 6, "M"), (1e3, 3, "K")];
    let mut div = 1.0;
    let mut shift = 0;
    let mut suffix = "";
    for &(d, e, s) in UNITS.iter() {
        if a >= d {
            div = d;
            shift = e;
            suffix = s;
            break;
        }
    }
    // round in scaled units to 1 fraction digit (on the decimal digits, as ICU), then re-bucket if the
    // rounded value reaches 1000 of the current unit
    let mut fixed = intl_fixed(a, shift, 1);
    if js_number_strict(&fixed) >= 1000.0 && div < 1e12 {
        let idx = UNITS.iter().position(|&(d, _, _)| d == div);
        let next = match idx {
            Some(i) if i > 0 => UNITS[i - 1],
            None => (1e3, 3, "K"), // was plain, bump to K
            _ => (div, shift, suffix),
        };
        shift = next.1;
        suffix = next.2;
        fixed = intl_fixed(a, shift, 1);
    }
    let body = if min_frac_1 {
        fixed
    } else {
        trim_trailing_zeros_regex(&fixed)
    };
    let (int_part, frac_part) = body.split_once('.').unwrap_or((body.as_str(), ""));
    let grouped = group_thousands(int_part);
    let joined = if frac_part.is_empty() {
        grouped
    } else {
        format!("{grouped}.{frac_part}")
    };
    let sign = if neg { "-" } else { "" };
    format!("{sign}{joined}{suffix}")
}

/// `formatCompact` (format.ts:98-104).
pub fn format_compact(value: f64) -> String {
    if value.is_nan() {
        return format_number(value);
    }
    intl_compact(value, false)
}

/// `formatCompactCurrency` (format.ts) — Intl gives e.g. "$1.0M" or "$0.0";
/// the TS strips `/\.0(?=[A-Z]|$)/i`.
pub fn format_compact_currency(value: f64) -> String {
    let neg = value < 0.0;
    let compact = intl_compact(value.abs(), true);
    let with_symbol = if neg {
        format!("-${compact}")
    } else {
        format!("${compact}")
    };
    strip_dot_zero_before_letter(&with_symbol)
}

fn strip_dot_zero_before_letter(s: &str) -> String {
    // replace first /\.0(?=[A-Za-z]|$)/
    let bytes = s.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b'.'
            && bytes[i + 1] == b'0'
            && bytes.get(i + 2).is_none_or(|b| b.is_ascii_alphabetic())
        {
            return format!("{}{}", &s[..i], &s[i + 2..]);
        }
    }
    s.to_string()
}

// ---------------------------------------------------------------------------
// A chart's number language (format.ts chartNumberSymbols / withChartLocale /
// localizeNumberLabel): `config.locale` picks the decimal and group separators
// from `number_symbols.json`, the table the TS twin imports.
// ---------------------------------------------------------------------------

/// A language's number separators and percent form (`ChartNumberSymbols`).
#[derive(Clone, Debug, PartialEq)]
pub struct NumberSymbols {
    pub decimal: String,
    pub group: String,
    pub min_group: u32,
    /// Where a percentage isn't written "12%": the number as `#` ("#\u{a0}%", "%#").
    pub percent: Option<String>,
}

fn number_symbols_table() -> &'static std::collections::HashMap<String, NumberSymbols> {
    static TABLE: std::sync::OnceLock<std::collections::HashMap<String, NumberSymbols>> =
        std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let json: crate::value::Map =
            crate::value::from_str(include_str!("number_symbols.json")).unwrap_or_default();
        json.into_iter()
            .filter_map(|(tag, v)| {
                let symbols = NumberSymbols {
                    decimal: v.get("decimal")?.as_str()?.to_string(),
                    group: v.get("group")?.as_str()?.to_string(),
                    min_group: v.get("minGroup")?.as_u64()? as u32,
                    percent: v.get("percent").and_then(|p| p.as_str()).map(str::to_string),
                };
                Some((tag.to_string(), symbols))
            })
            .collect()
    })
}

/// `chartNumberSymbols`: the longest prefix of the BCP 47 tag in the table, else
/// none (en-US's separators).
pub fn chart_number_symbols(locale: Option<&str>) -> Option<NumberSymbols> {
    let lower = locale?.to_ascii_lowercase().replace('_', "-");
    let head = ["-u-", "-x-"]
        .iter()
        .filter_map(|ext| lower.find(ext))
        .min()
        .map_or(lower.as_str(), |i| &lower[..i]);
    let parts: Vec<&str> = head.split('-').collect();
    let table = number_symbols_table();
    (1..=parts.len())
        .rev()
        .find_map(|n| table.get(&parts[..n].join("-")).cloned())
}

thread_local! {
    /// The separators of the chart being generated (`activeSymbols`).
    static ACTIVE_SYMBOLS: std::cell::RefCell<Option<NumberSymbols>> =
        const { std::cell::RefCell::new(None) };
}

/// `withChartLocale`: run a chart's generator with its numbers in `locale`'s separators.
pub fn with_chart_locale<T>(locale: Option<&str>, run: impl FnOnce() -> T) -> T {
    let previous = ACTIVE_SYMBOLS.with(|a| a.replace(chart_number_symbols(locale)));
    struct Restore(Option<NumberSymbols>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            ACTIVE_SYMBOLS.with(|a| *a.borrow_mut() = previous);
        }
    }
    let _restore = Restore(previous);
    run()
}

/// `localizeNumberLabel`: an en-US label in `symbols`' separators and percent form.
pub fn localize_number_label(label: &str, symbols: Option<&NumberSymbols>) -> String {
    let Some(symbols) = symbols else {
        return label.to_string();
    };
    let percented = match &symbols.percent {
        Some(pattern) => localize_percent(label, pattern),
        None => label.to_string(),
    };
    let chars: Vec<char> = percented.chars().collect();
    let digit = |i: usize| chars.get(i).is_some_and(char::is_ascii_digit);
    let mut out = String::with_capacity(label.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        match c {
            ',' => {
                // `/(^|[^\d,.])(\d),(\d{3})(?![\d,])/`: a lone group after one digit isn't written
                // where the first group needs two digits.
                let lone = symbols.min_group > 1
                    && i >= 1
                    && digit(i - 1)
                    && (i < 2 || !(digit(i - 2) || chars[i - 2] == ',' || chars[i - 2] == '.'))
                    && digit(i + 1)
                    && digit(i + 2)
                    && digit(i + 3)
                    && !(digit(i + 4) || chars.get(i + 4) == Some(&','));
                if !lone {
                    out.push_str(&symbols.group);
                }
            }
            '.' => out.push_str(&symbols.decimal),
            _ => out.push(c),
        }
    }
    out
}

/// `/(\d[\d,.]*)%/g` → `pattern` with the number for its `#`.
fn localize_percent(label: &str, pattern: &str) -> String {
    let chars: Vec<char> = label.chars().collect();
    let mut out = String::with_capacity(label.len() + 4);
    let mut start = 0; // chars before `start` are in `out`
    for (i, &c) in chars.iter().enumerate() {
        if c != '%' || i == 0 || !chars[i - 1].is_ascii_digit() {
            continue;
        }
        let mut j = i;
        while j > start && (chars[j - 1].is_ascii_digit() || chars[j - 1] == ',' || chars[j - 1] == '.') {
            j -= 1;
        }
        while !chars[j].is_ascii_digit() {
            j += 1;
        }
        out.extend(&chars[start..j]);
        let number: String = chars[j..i].iter().collect();
        out.push_str(&pattern.replacen('#', &number, 1));
        start = i + 1;
    }
    out.extend(&chars[start..]);
    out
}

fn localize_active(label: String) -> String {
    ACTIVE_SYMBOLS.with(|a| match a.borrow().as_ref() {
        Some(symbols) => localize_number_label(&label, Some(symbols)),
        None => label,
    })
}

/// `formatNumberWithStyle` (format.ts), in the active chart's separators.
pub fn format_number_with_style(value: f64, format: &str) -> String {
    localize_active(format_number_with_style_en_us(value, format))
}

fn format_number_with_style_en_us(value: f64, format: &str) -> String {
    if !value.is_finite() {
        return String::new();
    }
    match format {
        "currency" | "currency-full" => format_currency(value),
        "percent" => format_percent(value),
        "compact" => format_compact(value),
        "compact-currency" => format_compact_currency(value),
        _ => format_grouped_number(value),
    }
}

/// `formatGroupedNumber` (format.ts): `format_number` with en-US thousands grouping ("4,310",
/// "12,480.5"), as the auto axis ticks read ("5,000").
pub fn format_grouped_number(value: f64) -> String {
    let plain = format_number(value);
    let (sign, rest) = match plain.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", plain.as_str()),
    };
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    format!("{sign}{}{}", group_thousands(&rest[..end]), &rest[end..])
}

// ---------------------------------------------------------------------------
// Auto-detection + density (format.ts:164-254)
// ---------------------------------------------------------------------------

/// `autoDetectAxisFormat` — `format` may be absent (undefined) on malformed
/// configs; `None != "auto"` short-circuits exactly like the TS `!==`.
pub fn auto_detect_axis_format<'a>(ticks: &[f64], explicit: Option<&'a str>) -> Option<&'a str> {
    if explicit != Some("auto") {
        return explicit;
    }
    if ticks.is_empty() {
        return Some("auto");
    }
    let max_abs = ticks.iter().fold(f64::NEG_INFINITY, |m, t| {
        let a = t.abs();
        if a.is_nan() || m.is_nan() {
            f64::NAN
        } else if a > m {
            a
        } else {
            m
        }
    });
    if max_abs >= 100_000.0 {
        Some("compact")
    } else {
        Some("auto")
    }
}

/// `autoDetectValueFormat` (format.ts:180-192).
pub fn auto_detect_value_format<'a>(values: &[f64], explicit: Option<&'a str>) -> &'a str {
    let max_abs = || {
        values.iter().fold(
            f64::NEG_INFINITY,
            |m, v| if v.abs() > m { v.abs() } else { m },
        )
    };
    if explicit == Some("currency") {
        if values.is_empty() {
            return "currency";
        }
        if max_abs() >= 100_000.0 {
            return "compact-currency";
        }
        return "currency-full";
    }
    if explicit != Some("auto") {
        return explicit.unwrap_or("");
    }
    if values.is_empty() {
        return "auto";
    }
    if max_abs() >= 100_000.0 {
        "compact"
    } else {
        "auto"
    }
}

/// `applyNumberDensity` (format.ts:194-214).
pub fn apply_number_density<'a>(format: &'a str, density: &str) -> &'a str {
    if density == "auto" {
        return format;
    }
    if density == "compact" {
        return match format {
            "currency" | "currency-full" => "compact-currency",
            "auto" => "compact",
            other => other,
        };
    }
    if density == "full" {
        return match format {
            "currency" | "compact-currency" => "currency-full",
            other => other,
        };
    }
    format
}

/// `getValueFormatDowngradeOrder` (format.ts:216-225).
pub fn value_format_downgrade_order(format: &str) -> Vec<&str> {
    match format {
        "currency" => vec!["currency", "compact-currency", "compact"],
        "compact-currency" => vec!["compact-currency", "compact"],
        other => vec![other],
    }
}

/// `estimateFormattedValueWidth` (format.ts:227-230).
pub fn estimate_formatted_value_width(value: f64, format: &str, font_size: f64) -> f64 {
    let label = format_number_with_style(value, format);
    (label.chars().count() as f64 * font_size * CHARACTER_WIDTH_FONT_SIZE_RATIO).ceil()
        + LABEL_WIDTH_PADDING
}

/// `resolveEffectiveValueLabelFormat` (format.ts:232-254).
pub fn resolve_effective_value_label_format<'a>(
    values: &[f64],
    preferred_format: Option<&'a str>,
    density: &str,
    available_width: f64,
    font_size: f64,
) -> &'a str
where
    'static: 'a,
{
    let detected = auto_detect_value_format(values, preferred_format);
    let resolved: &str = apply_number_density(detected, density);
    // Lifetimes: every candidate string is a 'static literal or the caller's
    // `preferred_format`; the downgrade order only introduces literals.
    let candidates = value_format_downgrade_order(resolved);
    for format in &candidates {
        let widest = values.iter().fold(0.0_f64, |m, &v| {
            let w = estimate_formatted_value_width(v, format, font_size);
            if w > m {
                w
            } else {
                m
            }
        });
        if widest <= available_width {
            return widest_lifetime(format, preferred_format);
        }
    }
    widest_lifetime(candidates[candidates.len() - 1], preferred_format)
}

/// Map a candidate back to either a 'static literal or the caller's str.
fn widest_lifetime<'a>(candidate: &str, preferred: Option<&'a str>) -> &'a str {
    match candidate {
        "currency" => "currency",
        "currency-full" => "currency-full",
        "compact-currency" => "compact-currency",
        "compact" => "compact",
        "percent" => "percent",
        "auto" => "auto",
        other => {
            if Some(other) == preferred {
                preferred.unwrap()
            } else {
                "auto"
            }
        }
    }
}

/// `getEffectiveAxisFormat` (format.ts:263-271).
pub fn get_effective_axis_format<'a>(
    display_range: (f64, f64),
    tick_count: f64,
    explicit: Option<&'a str>,
) -> Result<(Vec<f64>, Option<&'a str>), TickError> {
    let ticks = generate_ticks(display_range.0, display_range.1, tick_count)?;
    let format = auto_detect_axis_format(&ticks, explicit);
    Ok((ticks, format))
}

// ---------------------------------------------------------------------------
// Axis tick formatting — d3-format subset (format.ts:273-386)
// ---------------------------------------------------------------------------

const MINUS: &str = "\u{2212}"; // d3 locale default minus

/// d3 `formatRounded` (formatRounded.js).
fn format_rounded(x: f64, p: usize) -> String {
    match format_decimal_parts(x, Some(p)) {
        None => js_num_to_string(x),
        Some((coefficient, exponent)) => {
            let e = exponent;
            if e < 0 {
                format!("0.{}{}", "0".repeat((-e - 1) as usize), coefficient)
            } else if coefficient.len() as i32 > e + 1 {
                format!(
                    "{}.{}",
                    &coefficient[..(e + 1) as usize],
                    &coefficient[(e + 1) as usize..]
                )
            } else {
                format!(
                    "{}{}",
                    coefficient,
                    "0".repeat((e + 1) as usize - coefficient.len())
                )
            }
        }
    }
}

/// d3 `formatPrefixAuto` (formatPrefixAuto.js) — returns (mantissa,
/// prefix_exponent) where prefix_exponent picks the SI letter.
fn format_prefix_auto(x: f64, p: usize) -> (String, Option<i32>) {
    match format_decimal_parts(x, Some(p)) {
        None => (js_num_to_string(x), None), // toPrecision path for 0 — only hit for x=0
        Some((coefficient, exponent)) => {
            let prefix_exponent = ((exponent as f64 / 3.0).floor().clamp(-8.0, 8.0)) as i32 * 3;
            let i = exponent - prefix_exponent + 1;
            let n = coefficient.len() as i32;
            let s = if i == n {
                coefficient
            } else if i > n {
                format!("{}{}", coefficient, "0".repeat((i - n) as usize))
            } else if i > 0 {
                format!(
                    "{}.{}",
                    &coefficient[..i as usize],
                    &coefficient[i as usize..]
                )
            } else {
                // less than 1y — re-round
                let again = format_decimal_parts(x, Some((p as i32 + i - 1).max(0) as usize + 0))
                    .map(|(c, _)| c)
                    .unwrap_or_default();
                format!("0.{}{}", "0".repeat((1 - i) as usize - 1), again)
            };
            (s, Some(prefix_exponent))
        }
    }
}

/// d3 `formatTrim` (formatTrim.js).
fn format_trim(s: &str) -> String {
    let b = s.as_bytes();
    let n = b.len();
    let mut i0: i32 = -1;
    let mut i1: i32 = -1;
    let mut i = 1;
    while i < n {
        match b[i] {
            b'.' => {
                i0 = i as i32;
                i1 = i as i32;
            }
            b'0' => {
                if i0 == 0 {
                    i0 = i as i32;
                }
                i1 = i as i32;
            }
            c => {
                if !c.is_ascii_digit() {
                    break;
                }
                if i0 > 0 {
                    i0 = 0;
                }
            }
        }
        i += 1;
    }
    if i0 > 0 {
        format!("{}{}", &s[..i0 as usize], &s[(i1 + 1) as usize..])
    } else {
        s.to_string()
    }
}

#[derive(Clone, Copy)]
struct D3Spec {
    currency: bool,
    comma: bool,
    precision: Option<usize>,
    trim: bool,
    ty: char, // '\0' = none, 'r', 's', 'f', '%'
}

/// The d3-format pipeline (locale.js newFormat) for the subset of specifiers
/// the chart engine uses (no width/fill/align/zero/sign variants).
fn d3_format(spec: D3Spec, value: f64) -> String {
    let mut ty = spec.ty;
    let mut trim = spec.trim;
    let mut precision = spec.precision;
    if ty == '\0' {
        if precision.is_none() {
            precision = Some(12);
        }
        trim = true;
        ty = 'g';
    }
    let precision = match precision {
        None => 6,
        Some(p) => {
            if matches!(ty, 'g' | 'p' | 'r' | 's') {
                p.clamp(1, 21)
            } else {
                p.min(20)
            }
        }
    };

    let value_negative = value < 0.0 || (value == 0.0 && value.is_sign_negative());
    let abs = value.abs();
    let mut prefix_exponent: Option<i32> = None;
    let mut formatted = if value.is_nan() {
        "NaN".to_string()
    } else {
        match ty {
            'g' => to_precision(abs, precision),
            'r' => format_rounded(abs, precision),
            's' => {
                let (m, pe) = format_prefix_auto(abs, precision);
                prefix_exponent = pe;
                m
            }
            'f' => to_fixed(abs, precision),
            '%' => to_fixed(abs * 100.0, precision),
            _ => js_num_to_string(abs),
        }
    };
    if trim {
        formatted = format_trim(&formatted);
    }
    let mut value_negative = value_negative;
    if value_negative && js_number_strict(&formatted) == 0.0 {
        value_negative = false;
    }
    let value_prefix = format!(
        "{}{}",
        if value_negative { MINUS } else { "" },
        if spec.currency { "$" } else { "" }
    );
    let si_letter = match prefix_exponent {
        Some(pe) if ty == 's' => {
            const PREFIXES: [&str; 17] = [
                "y", "z", "a", "f", "p", "n", "µ", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y",
            ];
            PREFIXES[(8 + pe / 3) as usize]
        }
        _ => "",
    };
    let mut value_suffix = format!("{}{}", si_letter, if ty == '%' { "%" } else { "" });

    // maybeSuffix split: integer part vs fractional/exponential tail
    let mut int_part = formatted.clone();
    if matches!(ty, 'd' | 'e' | 'f' | 'g' | 'p' | 'r' | 's' | '%') {
        if let Some(pos) = formatted.bytes().position(|c| !c.is_ascii_digit()) {
            let c = formatted.as_bytes()[pos];
            let tail = if c == b'.' {
                format!(".{}", &formatted[pos + 1..])
            } else {
                formatted[pos..].to_string()
            };
            value_suffix = format!("{tail}{value_suffix}");
            int_part = formatted[..pos].to_string();
        }
    }
    let grouped = if spec.comma {
        group_thousands(&int_part)
    } else {
        int_part
    };
    format!("{value_prefix}{grouped}{value_suffix}")
}

// The 8 pinned formatter instances (format.ts:298-309).
fn axis_auto_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: false,
            comma: true,
            precision: None,
            trim: false,
            ty: '\0',
        },
        v,
    )
}
fn axis_auto_fractional_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: false,
            comma: true,
            precision: None,
            trim: true,
            ty: 'r',
        },
        v,
    )
}
fn axis_compact_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: false,
            comma: false,
            precision: Some(2),
            trim: true,
            ty: 's',
        },
        v,
    )
}
fn axis_currency_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: true,
            comma: true,
            precision: Some(0),
            trim: false,
            ty: 'f',
        },
        v,
    )
}
fn axis_currency_fractional_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: true,
            comma: true,
            precision: None,
            trim: true,
            ty: 'r',
        },
        v,
    )
}
fn axis_compact_currency_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: true,
            comma: false,
            precision: Some(2),
            trim: true,
            ty: 's',
        },
        v,
    )
}
fn axis_percent_integer_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: false,
            comma: false,
            precision: Some(0),
            trim: false,
            ty: '%',
        },
        v,
    )
}
fn axis_percent_decimal_formatter(v: f64) -> String {
    d3_format(
        D3Spec {
            currency: false,
            comma: false,
            precision: Some(1),
            trim: false,
            ty: '%',
        },
        v,
    )
}

/// `formatAuto` (format.ts:311-318).
fn format_auto(value: f64) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    if is_integer(value) {
        axis_auto_formatter(value)
    } else {
        axis_auto_fractional_formatter(value)
    }
}

/// `remapSiPrefix` (format.ts): trailing k → K and G → B, so axis ticks use
/// the same K / M / B / T suffixes as the Intl value labels.
fn remap_si_prefix(label: &str) -> String {
    if let Some(p) = label.strip_suffix('k') {
        format!("{p}K")
    } else if let Some(p) = label.strip_suffix('G') {
        format!("{p}B")
    } else {
        label.to_string()
    }
}

/// `formatPercentTick` (format.ts:329-346).
fn format_percent_tick(value: f64) -> String {
    let fractional = if value.abs() <= 1.0 {
        value
    } else {
        value / 100.0
    };
    let rounded = super::jsmath::js_round(fractional * 100.0);
    if (rounded - fractional * 100.0).abs() < 1e-9 {
        axis_percent_integer_formatter(fractional)
    } else {
        axis_percent_decimal_formatter(fractional)
    }
}

/// `formatAxisTick` (format.ts), in the active chart's separators.
pub fn format_axis_tick(value: f64, format: Option<&str>) -> String {
    localize_active(format_axis_tick_en_us(value, format))
}

fn format_axis_tick_en_us(value: f64, format: Option<&str>) -> String {
    if !value.is_finite() {
        return String::new();
    }
    match format {
        Some("currency") | Some("currency-full") => {
            // A fractional tick keeps its digits ("$1.4", not "$1"): whole dollars repeat labels on a narrow domain.
            if value.fract() != 0.0 {
                axis_currency_fractional_formatter(value)
            } else {
                axis_currency_formatter(value)
            }
        }
        Some("compact-currency") => {
            if value.abs() > 0.0 && value.abs() < 1.0 {
                axis_currency_fractional_formatter(value)
            } else {
                remap_si_prefix(&axis_compact_currency_formatter(value))
            }
        }
        Some("compact") => {
            if value.abs() > 0.0 && value.abs() < 1.0 {
                format_auto(value)
            } else {
                remap_si_prefix(&axis_compact_formatter(value))
            }
        }
        Some("percent") => format_percent_tick(value),
        _ => format_auto(value),
    }
}

// ---------------------------------------------------------------------------
// Time axis tick formatting (d3-time-format / d3-scale multi-scale)
// ---------------------------------------------------------------------------

/// `createTimeTickFormatter` (format.ts:413-423). With an explicit
/// `xTimeFormat` specifier, every tick is formatted by `utcFormat(specifier)`;
/// without one (or an empty string — falsy in the TS), the d3-scale multi-scale
/// formatter picks a per-tick format. Returns a closure over ms-since-epoch
/// (f64, matching the numeric-axis tick values).
pub fn create_time_tick_formatter(specifier: Option<&str>) -> Box<dyn Fn(f64) -> String> {
    match specifier.filter(|s| !s.is_empty()) {
        Some(spec) => {
            let spec = spec.to_string();
            Box::new(move |value: f64| super::timefmt::utc_format(&spec, value as i64))
        }
        None => Box::new(|value: f64| super::timefmt::utc_multiscale_format(value as i64)),
    }
}

#[cfg(test)]
mod tests {

    /// `labelEms` (chartWideLabels.test.ts): a wide character is one em, any other UTF-16 unit the narrow width.
    #[test]
    fn label_ems_counts_a_wide_character_as_one_em() {
        let near = |a: f64, b: f64| assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        near(super::label_ems("Revenue", 0.6), 7.0 * 0.6);
        near(super::label_ems("精密部品", 0.6), 4.0);
        near(super::label_ems("매출 Q1", 0.65), 2.0 + 3.0 * 0.65);
        near(super::label_ems("ＡＢ", 0.65), 2.0);
    }

    use super::*;

    #[test]
    fn auto_value_labels_group_thousands() {
        // chartGroupedLabels.test.ts: a bar reads "4,310" under a "5,000" tick.
        assert_eq!(format_number_with_style(4310.0, "auto"), "4,310");
        assert_eq!(format_number_with_style(98000.0, "auto"), "98,000");
        assert_eq!(format_grouped_number(12480.5), "12,480.5");
        assert_eq!(format_grouped_number(-1234567.891), "-1,234,567.89");
        assert_eq!(format_grouped_number(999.0), "999");
        assert_eq!(format_grouped_number(0.25), "0.25");
        assert_eq!(format_axis_tick(5000.0, None), "5,000");
    }

    #[test]
    fn intl_paths_match_node() {
        // verified against node Intl.NumberFormat / the TS helpers
        assert_eq!(format_number(45.599999999999994), "45.6");
        assert_eq!(format_number(120.0), "120");
        assert_eq!(format_percent(0.25), "25%");
        assert_eq!(format_percent(0.4567), "45.67%");
        assert_eq!(format_currency(1234.0), "$1,234");
        assert_eq!(format_currency(1234.5), "$1,234.50");
        assert_eq!(format_currency(-1234.0), "-$1,234");
        assert_eq!(format_compact(1234.0), "1.2K");
        assert_eq!(format_compact(1000.0), "1K");
        assert_eq!(format_compact(999.0), "999");
        assert_eq!(format_compact(1_500_000.0), "1.5M");
        assert_eq!(format_compact(999_950.0), "1M");
        // Intl rounds the shortest decimal (what JS prints), not the binary value: 334650 / 1000 is 334.6499… in
        // binary, and 1.005 is 1.00499…
        assert_eq!(format_compact(334_650.0), "334.7K");
        assert_eq!(format_compact(-334_650.0), "-334.7K");
        assert_eq!(format_compact(1.15), "1.2");
        assert_eq!(format_compact(0.35), "0.4");
        assert_eq!(format_compact(8.25), "8.3");
        assert_eq!(format_compact(0.05), "0.1");
        assert_eq!(format_currency(1.005), "$1.01");
        assert_eq!(format_currency(2.675), "$2.68");
        assert_eq!(format_compact_currency(1_200_000.0), "$1.2M");
        assert_eq!(format_compact_currency(1_000_000.0), "$1M");
        assert_eq!(format_compact_currency(-1500.0), "-$1.5K");
        assert_eq!(format_compact_currency(0.0), "$0");
        assert_eq!(format_compact_currency(5.0), "$5");
        assert_eq!(format_compact_currency(5.5), "$5.5");
    }

    #[test]
    fn axis_ticks_match_d3() {
        // node: d3.format(',')(1234) = "1,234"; d3.format(',~r')(0.5) = "0.5"
        assert_eq!(format_axis_tick(0.0, Some("auto")), "0");
        assert_eq!(format_axis_tick(1234.0, Some("auto")), "1,234");
        assert_eq!(format_axis_tick(0.5, Some("auto")), "0.5");
        assert_eq!(format_axis_tick(12.3, Some("auto")), "12.3");
        // d3.format('.2~s')(100000) = "100k", remapped to the Intl "K"; (1500000) = "1.5M"
        assert_eq!(format_axis_tick(100000.0, Some("compact")), "100K");
        assert_eq!(format_axis_tick(1500000.0, Some("compact")), "1.5M");
        assert_eq!(format_axis_tick(1.5e9, Some("compact")), "1.5B"); // G→B remap
                                                                      // d3.format('$,.0f')(1234) = "$1,234"
        assert_eq!(format_axis_tick(1234.0, Some("currency")), "$1,234");
        assert_eq!(format_axis_tick(0.5, Some("currency")), "$0.5");
        assert_eq!(format_axis_tick(1.4, Some("currency")), "$1.4");
        assert_eq!(format_axis_tick(1.6000000000000001, Some("currency")), "$1.6");
        assert_eq!(format_axis_tick(2.0, Some("currency")), "$2");
        assert_eq!(format_axis_tick(2.2, Some("currency-full")), "$2.2");
        assert_eq!(
            format_axis_tick(1500000.0, Some("compact-currency")),
            "$1.5M"
        );
        // compact ticks use the same suffixes as the compact value labels
        assert_eq!(
            format_axis_tick(100000.0, Some("compact-currency")),
            "$100K"
        );
        assert_eq!(format_compact_currency(98000.0), "$98K");
        assert_eq!(format_axis_tick(-2500.0, Some("compact")), "\u{2212}2.5K");
        // percent: stacked-percent passes 20, 40 … (already-percent inputs)
        assert_eq!(format_axis_tick(20.0, Some("percent")), "20%");
        assert_eq!(format_axis_tick(0.5, Some("percent")), "50%");
        assert_eq!(format_axis_tick(0.123, Some("percent")), "12.3%");
        // negatives use U+2212
        assert_eq!(format_axis_tick(-50.0, Some("auto")), "\u{2212}50");
        // undefined format falls through to auto (chart-gated path)
        assert_eq!(format_axis_tick(0.0, None), "0");
    }

    /// The cases the TS twin asserts (test/chartNumberLocale.test.ts): ticks
    /// and value labels in their language's separators, en-US for the rest.
    #[test]
    fn chart_number_locales_match_the_shared_cases() {
        let cases: Vec<crate::value::Value> =
            crate::value::from_str(include_str!("../../tests/fixtures/chart_number_locales.json")).unwrap();
        assert!(cases.len() > 100);
        for c in &cases {
            let locale = c["locale"].as_str();
            let value = c["value"].as_f64().unwrap();
            let format = c["format"].as_str().unwrap();
            let got = with_chart_locale(locale, || {
                if c["kind"] == "tick" {
                    format_axis_tick(value, Some(format))
                } else {
                    format_number_with_style(value, format)
                }
            });
            assert_eq!(got, c["expected"].as_str().unwrap(), "{c}");
        }
        assert_eq!(format_number_with_style(1234.5, "auto"), "1,234.5", "outside a chart: en-US");
    }

    #[test]
    fn a_locale_tag_resolves_by_its_longest_known_prefix() {
        let de = chart_number_symbols(Some("de")).unwrap();
        assert_eq!((de.decimal.as_str(), de.group.as_str()), (",", "."));
        assert_eq!(chart_number_symbols(Some("de-DE-u-nu-latn")), Some(de));
        assert_eq!(chart_number_symbols(Some("pt_BR")).unwrap().group, ".");
        assert_eq!(chart_number_symbols(Some("pt-PT")).unwrap().group, "\u{a0}");
        assert_eq!(chart_number_symbols(Some("en-US")), None);
        assert_eq!(chart_number_symbols(Some("xx")), None);
        assert_eq!(chart_number_symbols(None), None);
    }
}
