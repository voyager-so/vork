//! Vork's JSON boundary: recursively sorted object keys and explicit float parsing
//! precision, independent of features unified by embedded engines.
//!
//! All serde_json types and parsing APIs are shared with upstream. Only the six
//! text/byte output helpers below differ: object keys are sorted, arrays retain
//! their order. Use these helpers for files, protocol replies and cache identities.
use serde::Serialize;
pub use serde_json::*;
use std::io::Write;

fn sorted<T: ?Sized + Serialize>(value: &T) -> Result<Value> {
    // Serialize first instead of to_value: f32 must retain its short f32 decimal
    // spelling. arbitrary_precision preserves these tokens (and exact clocks)
    // through this pass without converting them to f64.
    let bytes = serde_json::to_vec(value)?;
    let mut deserializer = serde_json::Deserializer::from_slice(&bytes);
    // These bytes are our own serialized output, not user input. Sorting must
    // not impose the input parser's depth limit on values we can serialize.
    deserializer.disable_recursion_limit();
    let mut value = <Value as serde::Deserialize>::deserialize(&mut deserializer)?;
    value.sort_all_objects();
    Ok(value)
}

pub fn to_vec<T: ?Sized + Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(&sorted(value)?)
}
pub fn to_vec_pretty<T: ?Sized + Serialize>(value: &T) -> Result<Vec<u8>> {
    serde_json::to_vec_pretty(&sorted(value)?)
}
pub fn to_string<T: ?Sized + Serialize>(value: &T) -> Result<String> {
    serde_json::to_string(&sorted(value)?)
}
pub fn to_string_pretty<T: ?Sized + Serialize>(value: &T) -> Result<String> {
    serde_json::to_string_pretty(&sorted(value)?)
}
pub fn to_writer<W: Write, T: ?Sized + Serialize>(writer: W, value: &T) -> Result<()> {
    serde_json::to_writer(writer, &sorted(value)?)
}
pub fn to_writer_pretty<W: Write, T: ?Sized + Serialize>(writer: W, value: &T) -> Result<()> {
    serde_json::to_writer_pretty(writer, &sorted(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_depth_is_independent_of_input_limit() {
        let mut value = Value::Null;
        let mut expected = "null".to_owned();
        for _ in 0..160 {
            let mut object = Map::new();
            object.insert("z".into(), value);
            object.insert("a".into(), Value::from(0));
            value = Value::Object(object);
            expected = format!(r#"{{"a":0,"z":{expected}}}"#);
        }
        // User input keeps its existing admission limit.
        assert!(
            from_str::<Value>(&expected)
                .unwrap_err()
                .to_string()
                .contains("recursion limit exceeded")
        );
        assert_eq!(to_string(&value).unwrap(), expected);
        assert_eq!(to_vec(&value).unwrap(), expected.as_bytes());
        let mut bytes = Vec::new();
        to_writer(&mut bytes, &value).unwrap();
        assert_eq!(bytes, expected.as_bytes());

        let mut sorted_value = value.clone();
        sorted_value.sort_all_objects();
        let pretty = serde_json::to_string_pretty(&sorted_value).unwrap();
        assert_eq!(to_string_pretty(&value).unwrap(), pretty);
        assert_eq!(to_vec_pretty(&value).unwrap(), pretty.as_bytes());
        bytes.clear();
        to_writer_pretty(&mut bytes, &value).unwrap();
        assert_eq!(bytes, pretty.as_bytes());
    }

    #[test]
    fn nested_keys_are_sorted_and_arrays_keep_their_order() {
        let a: Value = from_str(r#"{"z":[{"b":1,"a":2},3],"a":{"z":0,"a":1}}"#).unwrap();
        let b: Value = from_str(r#"{"a":{"a":1,"z":0},"z":[{"a":2,"b":1},3]}"#).unwrap();
        let expected = r#"{"a":{"a":1,"z":0},"z":[{"a":2,"b":1},3]}"#;
        assert_eq!(to_string(&a).unwrap(), expected);
        assert_eq!(to_vec(&a).unwrap(), to_vec(&b).unwrap());
        assert_eq!(to_vec_pretty(&a).unwrap(), to_vec_pretty(&b).unwrap());
        let mut bytes = Vec::new();
        to_writer(&mut bytes, &a).unwrap();
        assert_eq!(bytes, expected.as_bytes());
        bytes.clear();
        to_writer_pretty(&mut bytes, &a).unwrap();
        assert_eq!(bytes, to_string_pretty(&b).unwrap().as_bytes());
    }

    #[test]
    fn numeric_tokens_keep_typed_float_and_exact_decimal_spelling() {
        for value in [0.1f32, -0.0, f32::MIN_POSITIVE, f32::MAX, f32::NAN] {
            assert_eq!(to_vec(&value).unwrap(), serde_json::to_vec(&value).unwrap());
        }
        for value in [0.1f64, -0.0, f64::MIN_POSITIVE, f64::MAX] {
            assert_eq!(to_vec(&value).unwrap(), serde_json::to_vec(&value).unwrap());
        }
        for text in [
            "9007199254740993",
            "29.970000000000001",
            "1e-1000",
            "1.2300",
        ] {
            let value: Value = from_str(text).unwrap();
            assert_eq!(to_string(&value).unwrap(), text);
        }
        // This decimal differs by one ULP with serde_json's fast parser.
        let value: f64 = from_str("0.84551240822557006").unwrap();
        assert_eq!(
            value.to_bits(),
            "0.84551240822557006".parse::<f64>().unwrap().to_bits()
        );
    }
}
