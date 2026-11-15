//! JS-semantics helpers over `serde_json::Value` for the packing port.
//!
//! The production packers read node `fillShaderParams` (a JSON object) with
//! plain JS accessors: `typeof v === 'number'` guards (`num()` in the shader
//! files), `getPathValue` dotted-path lookup, `Array.isArray` + string
//! filters, `??`/`||` fallbacks. These helpers mirror those EXACT semantics
//! (PLAN §0.3 "copy the bugs") so the packed buffers stay byte-identical.

use serde_json::Value;

/// `getPathValue` (`defineProceduralShader.ts:719-729`): dotted-path walk;
/// any non-object intermediate → undefined.
pub fn get_path_value<'v>(source: &'v Value, path: &str) -> Option<&'v Value> {
    let mut current = source;
    for segment in path.split('.') {
        current = current.as_object()?.get(segment)?;
    }
    Some(current)
}

/// `isFiniteNumber(v)`: JS `typeof v === 'number' && Number.isFinite(v)`.
/// JSON numbers are always finite, so this is "is a JSON number".
pub fn as_finite_number(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// `num(v, fallback)` (shader-file helper): `typeof v === 'number' ? v :
/// fallback` — same as `as_finite_number` for JSON input.
pub fn num(v: Option<&Value>, fallback: f64) -> f64 {
    v.and_then(as_finite_number).unwrap_or(fallback)
}

/// `isStringArray` (`defineProceduralShader.ts:753-755`).
pub fn as_string_array(v: &Value) -> Option<Vec<String>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        out.push(item.as_str()?.to_string());
    }
    Some(out)
}

/// JS `Math.round`: round half toward +∞ (`floor(x + 0.5)`).
pub fn js_round(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    (x + 0.5).floor()
}

/// `padColorArray(value, fallback, maxItems)` (`mesh/colorPacking.ts:16-19`):
/// keep string elements of an array value (non-arrays → empty), append the
/// fallback palette, truncate to `maxItems`.
pub fn pad_color_array(value: Option<&Value>, fallback: &[&str], max_items: usize) -> Vec<String> {
    let mut colors: Vec<String> = match value.and_then(Value::as_array) {
        Some(arr) => arr
            .iter()
            .filter_map(|c| c.as_str().map(str::to_string))
            .collect(),
        None => Vec::new(),
    };
    colors.extend(fallback.iter().map(|s| s.to_string()));
    colors.truncate(max_items);
    colors
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn path_lookup() {
        let v = json!({"a": {"b": {"c": 3.5}}, "x": 1});
        assert_eq!(
            get_path_value(&v, "a.b.c").and_then(as_finite_number),
            Some(3.5)
        );
        assert!(get_path_value(&v, "a.b.c.d").is_none());
        assert!(get_path_value(&v, "missing").is_none());
        // Non-object intermediate.
        assert!(get_path_value(&v, "x.y").is_none());
    }

    #[test]
    fn round_half_up() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0); // JS Math.round(-2.5) === -2
        assert_eq!(js_round(3.49), 3.0);
    }

    #[test]
    fn pad_color_array_semantics() {
        let v = json!(["#111", 5, "#222"]);
        // Non-string entries filtered, fallback appended, truncated.
        assert_eq!(
            pad_color_array(Some(&v), &["#aaa", "#bbb", "#ccc"], 3),
            vec!["#111", "#222", "#aaa"]
        );
        // Non-array → fallback only.
        assert_eq!(
            pad_color_array(Some(&json!("nope")), &["#aaa"], 1),
            vec!["#aaa"]
        );
    }
}
