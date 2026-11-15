//! Scene JSON from JavaScript hosts. JS strings are UTF-16 and may hold a lone surrogate (half an emoji, e.g.
//! `'😀'.slice(0, 1)` in an agent's script); `JSON.stringify` writes it as the escape `\ud83d`, which is valid JSON
//! text but not a Unicode scalar, so serde_json refuses the whole document ("unexpected end of hex escape"). One such
//! character in any text layer made every screenshot and export of its document fail.
//!
//! [`parse_value`] parses as usual; only when that fails does it replace each lone surrogate escape with `\ufffd`
//! (the replacement character, as a browser draws it) and parse again.

use crate::value::Value;

/// Parses JSON text, reading a lone surrogate escape (`\ud800` without its pair) as U+FFFD.
pub fn parse_value(raw: &str) -> crate::value::Result<Value> {
    match crate::value::from_str(raw) {
        Ok(v) => Ok(v),
        Err(first) => match replace_lone_surrogate_escapes(raw) {
            Some(fixed) => crate::value::from_str(&fixed),
            None => Err(first),
        },
    }
}

fn hex4(b: &[u8]) -> Option<u16> {
    let s = std::str::from_utf8(b.get(..4)?).ok()?;
    u16::from_str_radix(s, 16).ok()
}

/// `raw` with every `\uXXXX` escape of an unpaired surrogate replaced by `\ufffd`, or `None` when it has none.
/// Escapes are read pairwise (`\\` is one escape), so a literal backslash before `ud800` is left alone.
fn replace_lone_surrogate_escapes(raw: &str) -> Option<String> {
    let b = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut changed = false;
    let mut in_string = false;
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if !in_string {
            in_string = c == b'"';
            out.push(c);
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                in_string = false;
                out.push(c);
                i += 1;
            }
            b'\\' if b.get(i + 1) == Some(&b'u') => {
                let Some(unit) = hex4(&b[i + 2..]) else {
                    out.push(c);
                    i += 1;
                    continue;
                };
                let high = (0xD800..=0xDBFF).contains(&unit);
                let low = (0xDC00..=0xDFFF).contains(&unit);
                let paired_low = high
                    && b.get(i + 6) == Some(&b'\\')
                    && b.get(i + 7) == Some(&b'u')
                    && hex4(&b[(i + 8).min(b.len())..]).is_some_and(|u| (0xDC00..=0xDFFF).contains(&u));
                if paired_low {
                    out.extend_from_slice(&b[i..i + 12]);
                    i += 12;
                } else if high || low {
                    out.extend_from_slice(b"\\ufffd");
                    changed = true;
                    i += 6;
                } else {
                    out.extend_from_slice(&b[i..i + 6]);
                    i += 6;
                }
            }
            b'\\' => {
                out.push(c);
                if let Some(&next) = b.get(i + 1) {
                    out.push(next);
                }
                i += 2;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    // Only ASCII escapes were rewritten, so the bytes are still UTF-8.
    changed.then(|| String::from_utf8(out).expect("UTF-8 in, ASCII rewrites"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lone_surrogate_reads_as_the_replacement_character() {
        assert!(crate::value::from_str::<Value>(r#"{"t":"ok \ud800 x"}"#).is_err(), "serde_json refuses it");
        assert_eq!(parse_value(r#"{"t":"ok \ud800 x"}"#).unwrap()["t"], "ok \u{FFFD} x");
        assert_eq!(parse_value(r#"{"t":"\udc00"}"#).unwrap()["t"], "\u{FFFD}");
        assert_eq!(parse_value(r#"{"t":"a\ud83d"}"#).unwrap()["t"], "a\u{FFFD}", "at the end of a string");
        assert_eq!(parse_value(r#"["\ud83d\ud83d\ude00"]"#).unwrap()[0], "\u{FFFD}😀", "a high before a pair");
    }

    #[test]
    fn valid_json_is_unchanged() {
        assert_eq!(parse_value(r#"{"t":"\ud83d\ude00 \u00e9 \\ud800"}"#).unwrap()["t"], "😀 é \\ud800", "a pair, a BMP escape, an escaped backslash");
        assert!(parse_value(r#"{"t": }"#).is_err(), "other syntax errors stay errors");
        assert_eq!(replace_lone_surrogate_escapes(r#"{"k":"\ud800"}"#).as_deref(), Some(r#"{"k":"\ufffd"}"#));
        assert_eq!(replace_lone_surrogate_escapes(r#"{"\\ud800":1}"#), None);
    }
}
