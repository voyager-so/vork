//! Any `Serialize` written straight into a typed [`Value`]: what `serde_json::to_value` followed by the
//! conversion gave, without the `serde_json::Value` between them. A build whose serde_json keeps number tokens
//! as text (vork's) wrote every number as decimal text there and parsed it back here; a number is now a number
//! from the start.
//!
//! The rules are serde_json's value serializer's, so a record reads back as it did:
//! - an integer is an integer and a finite `f64` is that `f64`; a non-finite float is null; an `f32` is the
//!   `f64` its shortest decimal text reads as ([`Number::from_f32_shortest`]);
//! - `None`, unit and a unit struct are null; a unit variant is its name; a newtype is its content; a variant
//!   with content is a one-key object under its name;
//! - a map's keys are strings: text, a character, a boolean, an integer or a finite float are written as text,
//!   anything else is refused;
//! - where serde_json keeps number tokens as text, its own number token (how it writes such a number) is read as
//!   that number, and a 128-bit integer wider than 64 bits is the number its digits spell; elsewhere that struct
//!   name is ordinary data and such an integer is refused, as serde_json does in each build;
//! - where serde_json has raw JSON text (`RawValue`), it is read as the JSON it holds; elsewhere its struct name
//!   is ordinary data;
//! - a refusal is serde_json's own error for it.

use super::{number_marker_reserved, Error, Key, Map, Number, Result, Value, NUMBER_TOKEN};
use serde::ser::{Error as _, Impossible, Serialize};

/// Writes a typed [`Value`].
pub struct Serializer;

/// serde_json's raw-text struct: how a `RawValue` is written where serde_json has them.
const RAW_TOKEN: &str = "$serde_json::private::RawValue";

/// A struct of one field, both named `NAME`, holding `text`: the shape of serde_json's number and raw-text
/// structs, and with any other field name a malformed one.
struct Envelope<const RAW: bool, const WELL_FORMED: bool>(&'static str);
impl<const RAW: bool, const WELL_FORMED: bool> Serialize for Envelope<RAW, WELL_FORMED> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let name = if RAW { RAW_TOKEN } else { NUMBER_TOKEN };
        let mut envelope = serializer.serialize_struct(name, 1)?;
        envelope.serialize_field(if WELL_FORMED { name } else { "malformed" }, self.0)?;
        envelope.end()
    }
}
/// Whether this build's serde_json reads [`RAW_TOKEN`] as raw JSON text (its `raw_value` feature, which any
/// crate in the build can turn on). Where it does not, the name is an ordinary struct, here as there.
fn raw_marker_reserved() -> bool {
    static RESERVED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RESERVED.get_or_init(|| matches!(serde_json::to_value(Envelope::<true, true>("[1]")), Ok(serde_json::Value::Array(_))))
}

/// A map with one key serde_json refuses.
struct BadKey(f64);
impl Serialize for BadKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(Some(1))?;
        if self.0.is_nan() { map.serialize_entry(&self.0, &())?; } else { map.serialize_entry(&(), &())?; }
        map.end()
    }
}
/// serde_json's own error for what `probe` asks of its value serializer, so a refusal here is the error it
/// was there, category included. `said` stands in only if this build's serde_json accepts the probe.
fn upstream<T: Serialize>(probe: T, said: &str) -> Error {
    serde_json::to_value(probe).err().unwrap_or_else(|| Error::custom(said))
}
fn key_must_be_a_string() -> Error {
    upstream(BadKey(0.), "key must be a string")
}
fn float_key_must_be_finite() -> Error {
    upstream(BadKey(f64::NAN), "float key must be finite (got NaN or +/-inf)")
}
fn number_out_of_range() -> Error {
    upstream(u128::MAX, "number out of range")
}
fn invalid_number() -> Error {
    upstream(Envelope::<false, false>("1"), "invalid number")
}
fn invalid_raw_value() -> Error {
    upstream(Envelope::<true, false>("1"), "invalid raw value")
}
/// An integer wider than 64 bits. A build whose serde_json keeps number tokens as text writes its digits, which
/// read back as the nearest `f64`; any other build refuses it.
fn wide_integer(digits: &str) -> Result<Value> {
    if number_marker_reserved() {
        Number::from_token(digits).map(Value::Number).ok_or_else(number_out_of_range)
    } else {
        Err(number_out_of_range())
    }
}

impl serde::Serializer for Serializer {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = SerializeVec;
    type SerializeTuple = SerializeVec;
    type SerializeTupleStruct = SerializeVec;
    type SerializeTupleVariant = SerializeTupleVariant;
    type SerializeMap = SerializeMap;
    type SerializeStruct = SerializeMap;
    type SerializeStructVariant = SerializeStructVariant;

    fn serialize_bool(self, value: bool) -> Result<Value> {
        Ok(Value::Bool(value))
    }
    fn serialize_i8(self, value: i8) -> Result<Value> {
        self.serialize_i64(i64::from(value))
    }
    fn serialize_i16(self, value: i16) -> Result<Value> {
        self.serialize_i64(i64::from(value))
    }
    fn serialize_i32(self, value: i32) -> Result<Value> {
        self.serialize_i64(i64::from(value))
    }
    fn serialize_i64(self, value: i64) -> Result<Value> {
        Ok(Value::Number(value.into()))
    }
    fn serialize_i128(self, value: i128) -> Result<Value> {
        if let Ok(value) = u64::try_from(value) {
            Ok(Value::Number(value.into()))
        } else if let Ok(value) = i64::try_from(value) {
            Ok(Value::Number(value.into()))
        } else {
            wide_integer(&value.to_string())
        }
    }
    fn serialize_u8(self, value: u8) -> Result<Value> {
        self.serialize_u64(u64::from(value))
    }
    fn serialize_u16(self, value: u16) -> Result<Value> {
        self.serialize_u64(u64::from(value))
    }
    fn serialize_u32(self, value: u32) -> Result<Value> {
        self.serialize_u64(u64::from(value))
    }
    fn serialize_u64(self, value: u64) -> Result<Value> {
        Ok(Value::Number(value.into()))
    }
    fn serialize_u128(self, value: u128) -> Result<Value> {
        match u64::try_from(value) {
            Ok(value) => Ok(Value::Number(value.into())),
            Err(_) => wide_integer(&value.to_string()),
        }
    }
    fn serialize_f32(self, value: f32) -> Result<Value> {
        Ok(Value::from(value))
    }
    fn serialize_f64(self, value: f64) -> Result<Value> {
        Ok(Value::from(value))
    }
    fn serialize_char(self, value: char) -> Result<Value> {
        Ok(Value::String(value.to_string()))
    }
    fn serialize_str(self, value: &str) -> Result<Value> {
        Ok(Value::String(value.to_owned()))
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<Value> {
        Ok(Value::Array(value.iter().map(|&byte| Value::Number(byte.into())).collect()))
    }
    fn serialize_unit(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value> {
        self.serialize_unit()
    }
    fn serialize_unit_variant(self, _name: &'static str, _index: u32, variant: &'static str) -> Result<Value> {
        self.serialize_str(variant)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, value: &T) -> Result<Value> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value> {
        let mut object = Map::with_capacity(1);
        object.insert(Key::from(variant), value.serialize(Serializer)?);
        Ok(Value::Object(object))
    }
    fn serialize_none(self) -> Result<Value> {
        self.serialize_unit()
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Value> {
        value.serialize(self)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<SerializeVec> {
        Ok(SerializeVec { list: Vec::with_capacity(len.unwrap_or(0)) })
    }
    fn serialize_tuple(self, len: usize) -> Result<SerializeVec> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<SerializeVec> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<SerializeTupleVariant> {
        Ok(SerializeTupleVariant { name: variant, list: Vec::with_capacity(len) })
    }
    fn serialize_map(self, len: Option<usize>) -> Result<SerializeMap> {
        Ok(SerializeMap::Map { map: Map::with_capacity(len.unwrap_or(0)), next_key: None })
    }
    fn serialize_struct(self, name: &'static str, len: usize) -> Result<SerializeMap> {
        // The name is serde_json's only in a build that keeps number tokens as text; elsewhere it is a struct.
        if name == NUMBER_TOKEN && number_marker_reserved() {
            Ok(SerializeMap::Token { raw: false, out: None })
        } else if name == RAW_TOKEN && raw_marker_reserved() {
            Ok(SerializeMap::Token { raw: true, out: None })
        } else {
            self.serialize_map(Some(len))
        }
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<SerializeStructVariant> {
        Ok(SerializeStructVariant { name: variant, map: Map::with_capacity(len) })
    }
    fn collect_str<T: ?Sized + std::fmt::Display>(self, value: &T) -> Result<Value> {
        Ok(Value::String(value.to_string()))
    }
}

pub struct SerializeVec {
    list: Vec<Value>,
}
pub struct SerializeTupleVariant {
    name: &'static str,
    list: Vec<Value>,
}
pub enum SerializeMap {
    Map { map: Map, next_key: Option<Key> },
    /// serde_json's number token or raw JSON text: a struct of one field, its text.
    Token { raw: bool, out: Option<Value> },
}
pub struct SerializeStructVariant {
    name: &'static str,
    map: Map,
}

impl serde::ser::SerializeSeq for SerializeVec {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.list.push(value.serialize(Serializer)?);
        Ok(())
    }
    fn end(self) -> Result<Value> {
        Ok(Value::Array(self.list))
    }
}
impl serde::ser::SerializeTuple for SerializeVec {
    type Ok = Value;
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        serde::ser::SerializeSeq::serialize_element(self, value)
    }
    fn end(self) -> Result<Value> {
        serde::ser::SerializeSeq::end(self)
    }
}
impl serde::ser::SerializeTupleStruct for SerializeVec {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        serde::ser::SerializeSeq::serialize_element(self, value)
    }
    fn end(self) -> Result<Value> {
        serde::ser::SerializeSeq::end(self)
    }
}
impl serde::ser::SerializeTupleVariant for SerializeTupleVariant {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.list.push(value.serialize(Serializer)?);
        Ok(())
    }
    fn end(self) -> Result<Value> {
        let mut object = Map::with_capacity(1);
        object.insert(Key::from(self.name), Value::Array(self.list));
        Ok(Value::Object(object))
    }
}
impl serde::ser::SerializeMap for SerializeMap {
    type Ok = Value;
    type Error = Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        match self {
            SerializeMap::Map { next_key, .. } => {
                *next_key = Some(key.serialize(MapKeySerializer)?);
                Ok(())
            }
            SerializeMap::Token { .. } => unreachable!("serde_json's tokens are written as structs"),
        }
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        match self {
            SerializeMap::Map { map, next_key } => {
                // serde's contract: serialize_key comes first.
                let key = next_key.take().expect("serialize_value called before serialize_key");
                map.insert(key, value.serialize(Serializer)?);
                Ok(())
            }
            SerializeMap::Token { .. } => unreachable!("serde_json's tokens are written as structs"),
        }
    }
    fn end(self) -> Result<Value> {
        match self {
            SerializeMap::Map { map, .. } => Ok(Value::Object(map)),
            SerializeMap::Token { .. } => unreachable!("serde_json's tokens are written as structs"),
        }
    }
}
impl serde::ser::SerializeStruct for SerializeMap {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
        match self {
            SerializeMap::Map { map, .. } => {
                map.insert(Key::from(key), value.serialize(Serializer)?);
                Ok(())
            }
            SerializeMap::Token { raw, out } => {
                if key == if *raw { RAW_TOKEN } else { NUMBER_TOKEN } {
                    *out = Some(value.serialize(TokenText { raw: *raw })?);
                    Ok(())
                } else {
                    Err(if *raw { invalid_raw_value() } else { invalid_number() })
                }
            }
        }
    }
    fn end(self) -> Result<Value> {
        match self {
            SerializeMap::Map { map, .. } => Ok(Value::Object(map)),
            SerializeMap::Token { raw, out } => out.ok_or_else(|| if raw { invalid_raw_value() } else { invalid_number() }),
        }
    }
}
impl serde::ser::SerializeStructVariant for SerializeStructVariant {
    type Ok = Value;
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
        self.map.insert(Key::from(key), value.serialize(Serializer)?);
        Ok(())
    }
    fn end(self) -> Result<Value> {
        let mut object = Map::with_capacity(1);
        object.insert(Key::from(self.name), Value::Object(self.map));
        Ok(Value::Object(object))
    }
}

/// A map's key, as the text serde_json writes for it.
struct MapKeySerializer;

/// The text serde_json writes for a float key: the float's own shortest text (an `f32`'s is not its widened
/// `f64`'s), whatever the build holds numbers as. serde_json writes that text for a finite float value.
fn float_key<T: Serialize>(value: &T, finite: bool) -> Result<Key> {
    if !finite {
        return Err(float_key_must_be_finite());
    }
    serde_json::to_string(value).map(Key::from)
}

impl serde::Serializer for MapKeySerializer {
    type Ok = Key;
    type Error = Error;
    type SerializeSeq = Impossible<Key, Error>;
    type SerializeTuple = Impossible<Key, Error>;
    type SerializeTupleStruct = Impossible<Key, Error>;
    type SerializeTupleVariant = Impossible<Key, Error>;
    type SerializeMap = Impossible<Key, Error>;
    type SerializeStruct = Impossible<Key, Error>;
    type SerializeStructVariant = Impossible<Key, Error>;

    fn serialize_unit_variant(self, _name: &'static str, _index: u32, variant: &'static str) -> Result<Key> {
        Ok(Key::from(variant))
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, value: &T) -> Result<Key> {
        value.serialize(self)
    }
    fn serialize_bool(self, value: bool) -> Result<Key> {
        Ok(Key::from(if value { "true" } else { "false" }))
    }
    fn serialize_i8(self, value: i8) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_i16(self, value: i16) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_i32(self, value: i32) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_i64(self, value: i64) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_i128(self, value: i128) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_u8(self, value: u8) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_u16(self, value: u16) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_u32(self, value: u32) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_u64(self, value: u64) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_u128(self, value: u128) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_f32(self, value: f32) -> Result<Key> {
        float_key(&value, value.is_finite())
    }
    fn serialize_f64(self, value: f64) -> Result<Key> {
        float_key(&value, value.is_finite())
    }
    fn serialize_char(self, value: char) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
    fn serialize_str(self, value: &str) -> Result<Key> {
        Ok(Key::from(value))
    }
    fn serialize_bytes(self, _value: &[u8]) -> Result<Key> {
        Err(key_must_be_a_string())
    }
    fn serialize_unit(self) -> Result<Key> {
        Err(key_must_be_a_string())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Key> {
        Err(key_must_be_a_string())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Key> {
        Err(key_must_be_a_string())
    }
    fn serialize_none(self) -> Result<Key> {
        Err(key_must_be_a_string())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<Key> {
        Err(key_must_be_a_string())
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(key_must_be_a_string())
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(key_must_be_a_string())
    }
    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct> {
        Err(key_must_be_a_string())
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(key_must_be_a_string())
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(key_must_be_a_string())
    }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(key_must_be_a_string())
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(key_must_be_a_string())
    }
    fn collect_str<T: ?Sized + std::fmt::Display>(self, value: &T) -> Result<Key> {
        Ok(Key::from(value.to_string()))
    }
}

/// The one field of serde_json's number token or raw text: its text, read as the number it spells or as the
/// JSON it is.
struct TokenText {
    raw: bool,
}
impl TokenText {
    fn refused(&self) -> Error {
        if self.raw { invalid_raw_value() } else { invalid_number() }
    }
}

impl serde::Serializer for TokenText {
    type Ok = Value;
    type Error = Error;
    type SerializeSeq = Impossible<Value, Error>;
    type SerializeTuple = Impossible<Value, Error>;
    type SerializeTupleStruct = Impossible<Value, Error>;
    type SerializeTupleVariant = Impossible<Value, Error>;
    type SerializeMap = Impossible<Value, Error>;
    type SerializeStruct = Impossible<Value, Error>;
    type SerializeStructVariant = Impossible<Value, Error>;

    fn serialize_str(self, value: &str) -> Result<Value> {
        if self.raw {
            // Read as serde_json reads it (its own value, which reads its reserved keys), then converted: raw
            // text is the one input whose reading is serde_json's to define.
            serde_json::from_str::<serde_json::Value>(value).map(Value::from)
        } else {
            match Number::from_token(value) {
                Some(number) => Ok(Value::Number(number)),
                // Not a number this module reads: serde_json's own reading of the text decides, error included.
                None => value.parse::<serde_json::Number>().map(|number| Value::Number(Number::from(&number))),
            }
        }
    }
    fn serialize_bool(self, _value: bool) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_i8(self, _value: i8) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_i16(self, _value: i16) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_i32(self, _value: i32) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_i64(self, _value: i64) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_u8(self, _value: u8) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_u16(self, _value: u16) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_u32(self, _value: u32) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_u64(self, _value: u64) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_f32(self, _value: f32) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_f64(self, _value: f64) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_char(self, _value: char) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_bytes(self, _value: &[u8]) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_unit(self) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_unit_variant(self, _name: &'static str, _index: u32, _variant: &'static str) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, _value: &T) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _value: &T,
    ) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_none(self) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_some<T: ?Sized + Serialize>(self, _value: &T) -> Result<Value> {
        Err(self.refused())
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq> {
        Err(self.refused())
    }
    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple> {
        Err(self.refused())
    }
    fn serialize_tuple_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeTupleStruct> {
        Err(self.refused())
    }
    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant> {
        Err(self.refused())
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap> {
        Err(self.refused())
    }
    fn serialize_struct(self, _name: &'static str, _len: usize) -> Result<Self::SerializeStruct> {
        Err(self.refused())
    }
    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant> {
        Err(self.refused())
    }
}
