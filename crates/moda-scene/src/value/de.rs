//! A borrowed [`Value`] as a serde `Deserializer`, so a typed record (a paint, an effect) reads straight from a
//! node's fields: `Fill::deserialize(&value)`, as `&serde_json::Value` allowed. Numbers arrive as numbers.

use super::{Key, Map, Number, Value, N};
use serde::de::{
    DeserializeSeed, Deserializer, EnumAccess, Error as _, IntoDeserializer, MapAccess, SeqAccess, Unexpected,
    VariantAccess, Visitor,
};
use serde::forward_to_deserialize_any;

/// What reading a typed record from a [`Value`] fails with.
pub type Error = serde::de::value::Error;

fn visit_number<'de, V: Visitor<'de>>(n: &'de Number, visitor: V) -> Result<V::Value, Error> {
    match &n.0 {
        N::PosInt(u) => visitor.visit_u64(*u),
        N::NegInt(i) => visitor.visit_i64(*i),
        N::Float(f) => visitor.visit_f64(*f),
        // As serde_json hands on a number it keeps as text: a one-entry map under its number marker. A value
        // that carries any JSON (a `Value` field, a tagged record's buffered content) keeps the token; an
        // integer field refuses the map, as it refused the number. A float field asks through
        // `deserialize_f64` / `deserialize_f32` and never comes here.
        N::OutOfRange(token) => visitor.visit_map(NumberToken { token: Some(token) }),
    }
}

impl<'de> Deserializer<'de> for &'de Value {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Value::Null => visitor.visit_unit(),
            Value::Bool(b) => visitor.visit_bool(*b),
            Value::Number(n) => visit_number(n, visitor),
            Value::String(text) => visitor.visit_borrowed_str(text),
            Value::Array(list) => {
                let mut access = SeqRef { items: list.iter() };
                let value = visitor.visit_seq(&mut access)?;
                match access.items.len() {
                    0 => Ok(value),
                    _ => Err(Error::invalid_length(list.len(), &"fewer elements in array")),
                }
            }
            Value::Object(map) => {
                let mut access = MapRef { entries: map.iter(), value: None };
                let value = visitor.visit_map(&mut access)?;
                match access.entries.len() {
                    0 => Ok(value),
                    _ => Err(Error::invalid_length(map.len(), &"fewer elements in map")),
                }
            }
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Value::Null => visitor.visit_none(),
            _ => visitor.visit_some(self),
        }
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        match self {
            Value::String(variant) => visitor.visit_enum(EnumRef { variant, value: None }),
            Value::Object(map) => {
                let mut entries = map.iter();
                let (Some((variant, value)), None) = (entries.next(), entries.next()) else {
                    return Err(Error::invalid_value(Unexpected::Map, &"map with a single key"));
                };
                visitor.visit_enum(EnumRef { variant: variant.as_str(), value: Some(value) })
            }
            other => Err(Error::invalid_type(unexpected(other), &"string or map")),
        }
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(self, _name: &'static str, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    /// A field that asks for a float reads an out-of-range token as the float its text parses to (an infinity),
    /// as serde_json's number does for a requested float; every other number goes the usual way.
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Value::Number(Number(N::OutOfRange(token))) => match token.parse::<f64>() {
                Ok(f) => visitor.visit_f64(f),
                Err(_) => Err(Error::custom("number out of range")),
            },
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Value::Number(Number(N::OutOfRange(token))) => match token.parse::<f32>() {
                Ok(f) => visitor.visit_f32(f),
                Err(_) => Err(Error::custom("number out of range")),
            },
            _ => self.deserialize_any(visitor),
        }
    }

    /// A field nothing reads is skipped without being interpreted, whatever it holds.
    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_unit()
    }

    forward_to_deserialize_any! {
        bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 char str string bytes byte_buf unit unit_struct seq tuple
        tuple_struct map struct identifier
    }
}

/// An out-of-range number token as serde_json's number protocol: `{ "$serde_json::private::Number": token }`.
struct NumberToken<'de> {
    token: Option<&'de str>,
}

impl<'de> MapAccess<'de> for NumberToken<'de> {
    type Error = Error;
    fn next_key_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>, Error> {
        if self.token.is_none() {
            return Ok(None);
        }
        let key: serde::de::value::BorrowedStrDeserializer<'de, Error> =
            serde::de::value::BorrowedStrDeserializer::new(super::NUMBER_TOKEN);
        seed.deserialize(key).map(Some)
    }
    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value, Error> {
        match self.token.take() {
            Some(token) => {
                let token: serde::de::value::BorrowedStrDeserializer<'de, Error> =
                    serde::de::value::BorrowedStrDeserializer::new(token);
                seed.deserialize(token)
            }
            None => Err(Error::custom("value is missing")),
        }
    }
}

impl<'de> IntoDeserializer<'de, Error> for &'de Value {
    type Deserializer = Self;
    fn into_deserializer(self) -> Self {
        self
    }
}

fn unexpected(value: &Value) -> Unexpected<'_> {
    match value {
        Value::Null => Unexpected::Unit,
        Value::Bool(b) => Unexpected::Bool(*b),
        Value::Number(n) => match &n.0 {
            N::PosInt(u) => Unexpected::Unsigned(*u),
            N::NegInt(i) => Unexpected::Signed(*i),
            N::Float(f) => Unexpected::Float(*f),
            N::OutOfRange(_) => Unexpected::Other("number out of range"),
        },
        Value::String(text) => Unexpected::Str(text),
        Value::Array(_) => Unexpected::Seq,
        Value::Object(_) => Unexpected::Map,
    }
}

struct SeqRef<'de> {
    items: std::slice::Iter<'de, Value>,
}

impl<'de> SeqAccess<'de> for SeqRef<'de> {
    type Error = Error;
    fn next_element_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>, Error> {
        self.items.next().map(|item| seed.deserialize(item)).transpose()
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.items.len())
    }
}

struct MapRef<'de> {
    entries: super::Iter<'de>,
    value: Option<&'de Value>,
}

impl<'de> MapAccess<'de> for MapRef<'de> {
    type Error = Error;
    fn next_key_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>, Error> {
        match self.entries.next() {
            Some((key, value)) => {
                self.value = Some(value);
                seed.deserialize(KeyRef(key)).map(Some)
            }
            None => Ok(None),
        }
    }
    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value, Error> {
        match self.value.take() {
            Some(value) => seed.deserialize(value),
            None => Err(Error::custom("value is missing")),
        }
    }
    fn size_hint(&self) -> Option<usize> {
        Some(self.entries.len())
    }
}

/// An object key: a string (a number-typed key parses from it, as serde_json's does).
struct KeyRef<'de>(&'de Key);

macro_rules! key_as_number {
    ($($method:ident => $visit:ident,)*) => {$(
        fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
            match self.0.as_str().parse() {
                Ok(n) => visitor.$visit(n),
                Err(_) => visitor.visit_borrowed_str(self.0.as_str()),
            }
        }
    )*};
}

impl<'de> Deserializer<'de> for KeyRef<'de> {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_borrowed_str(self.0.as_str())
    }

    key_as_number! {
        deserialize_i8 => visit_i8, deserialize_i16 => visit_i16, deserialize_i32 => visit_i32,
        deserialize_i64 => visit_i64, deserialize_u8 => visit_u8, deserialize_u16 => visit_u16,
        deserialize_u32 => visit_u32, deserialize_u64 => visit_u64, deserialize_f32 => visit_f32,
        deserialize_f64 => visit_f64,
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_some(self)
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(self, _name: &'static str, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error> {
        visitor.visit_enum(EnumRef { variant: self.0.as_str(), value: None })
    }

    forward_to_deserialize_any! {
        bool i128 u128 char str string bytes byte_buf unit unit_struct seq tuple tuple_struct map struct identifier
        ignored_any
    }
}

struct EnumRef<'de> {
    variant: &'de str,
    value: Option<&'de Value>,
}

impl<'de> EnumAccess<'de> for EnumRef<'de> {
    type Error = Error;
    type Variant = VariantRef<'de>;
    fn variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<(T::Value, VariantRef<'de>), Error> {
        let variant: serde::de::value::BorrowedStrDeserializer<'de, Error> =
            serde::de::value::BorrowedStrDeserializer::new(self.variant);
        seed.deserialize(variant).map(|v| (v, VariantRef(self.value)))
    }
}

struct VariantRef<'de>(Option<&'de Value>);

impl<'de> VariantAccess<'de> for VariantRef<'de> {
    type Error = Error;
    fn unit_variant(self) -> Result<(), Error> {
        match self.0 {
            None | Some(Value::Null) => Ok(()),
            Some(other) => Err(Error::invalid_type(unexpected(other), &"unit variant")),
        }
    }
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(self, seed: T) -> Result<T::Value, Error> {
        match self.0 {
            Some(value) => seed.deserialize(value),
            None => Err(Error::invalid_type(Unexpected::UnitVariant, &"newtype variant")),
        }
    }
    fn tuple_variant<V: Visitor<'de>>(self, _len: usize, visitor: V) -> Result<V::Value, Error> {
        match self.0 {
            Some(value @ Value::Array(_)) => value.deserialize_any(visitor),
            Some(other) => Err(Error::invalid_type(unexpected(other), &"tuple variant")),
            None => Err(Error::invalid_type(Unexpected::UnitVariant, &"tuple variant")),
        }
    }
    fn struct_variant<V: Visitor<'de>>(self, _fields: &'static [&'static str], visitor: V) -> Result<V::Value, Error> {
        match self.0 {
            Some(value @ (Value::Object(_) | Value::Array(_))) => value.deserialize_any(visitor),
            Some(other) => Err(Error::invalid_type(unexpected(other), &"struct variant")),
            None => Err(Error::invalid_type(Unexpected::UnitVariant, &"struct variant")),
        }
    }
}

/// A [`Map`] reads as its object.
impl<'de> IntoDeserializer<'de, Error> for &'de Map {
    type Deserializer = serde::de::value::MapAccessDeserializer<MapRefOwned<'de>>;
    fn into_deserializer(self) -> Self::Deserializer {
        serde::de::value::MapAccessDeserializer::new(MapRefOwned(MapRef { entries: self.iter(), value: None }))
    }
}

/// [`MapRef`] by value, for [`serde::de::value::MapAccessDeserializer`].
pub struct MapRefOwned<'de>(MapRef<'de>);

impl<'de> MapAccess<'de> for MapRefOwned<'de> {
    type Error = Error;
    fn next_key_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<Option<T::Value>, Error> {
        self.0.next_key_seed(seed)
    }
    fn next_value_seed<T: DeserializeSeed<'de>>(&mut self, seed: T) -> Result<T::Value, Error> {
        self.0.next_value_seed(seed)
    }
    fn size_hint(&self) -> Option<usize> {
        self.0.size_hint()
    }
}
