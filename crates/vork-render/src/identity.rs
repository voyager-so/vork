//! The identity of a typed value: its serialized content, written once into a
//! hasher as the value is walked. No JSON text and no `Value` tree is built.
//!
//! The content is what the value's `Serialize` writes, in JSON's data model:
//! - a field the type skips is absent; `None`, unit and a non-finite float are null;
//!   `Some(x)` and a newtype are `x`; an enum variant is written as JSON writes it;
//! - a float is its bits, which name one short decimal spelling each (`-0.0` is
//!   not `0.0`); an integer is its value, whatever its width; an exact number
//!   token is its text;
//! - a map is its entries by key, whatever order it held them in, and a key
//!   written twice keeps its last value, as it does when JSON text is read; a
//!   struct's fields are in the order its type declares; an array keeps its order.
//!
//! The contract: two values of one type that are written as the same bytes have
//! the same sorted JSON text, so an identity never stands for different content.
//! The other way round holds wherever the type fixes how the content is written.
//! The bytes also name that: an `f32` and an `f64`, an integer and an exact
//! token, a struct and a map with its keys, or a text and an integer map key are
//! written differently, so a type that can write one JSON text in two such ways
//! (an untagged choice) gets two identities for it. Raw JSON text is refused.
//!
//! A map holds its keys and its short values; a longer value goes into a hasher
//! of its own and the map holds that digest, so a value is read once however
//! deep it sits and nothing the size of a subtree is held. The bytes are an identity only, not a file or protocol format. Files,
//! replies and identities that leave the process use vork-json's sorted text.
use serde::ser::{self, Serialize};
use serde_json::{Error, Result};
use std::io::Write;

/// Write the content of `value` into `writer`, usually a hasher.
pub(crate) fn write<W: Write, T: ?Sized + Serialize>(writer: W, value: &T) -> Result<()> {
    // Most values are a few bytes; the hasher takes them in blocks.
    let mut encoder = Encoder { writer: std::io::BufWriter::with_capacity(1 << 16, writer) };
    value.serialize(&mut encoder)?;
    encoder.writer.flush().map_err(Error::io)
}
/// The content of `value` as one digest.
pub(crate) fn digest<T: ?Sized + Serialize>(value: &T) -> Result<blake3::Hash> {
    let mut hash = blake3::Hasher::new();
    write(&mut hash, value)?;
    Ok(hash.finalize())
}

// Every value starts with one of these and delimits itself: scalars have a fixed
// size, a string carries its length, and a closing bracket starts no value.
const NULL: u8 = b'n';
const TRUE: u8 = b'T';
const FALSE: u8 = b'F';
const UNSIGNED: u8 = b'u';
const NEGATIVE: u8 = b'i';
const WIDE_UNSIGNED: u8 = b'U';
const WIDE_NEGATIVE: u8 = b'I';
const F32: u8 = b'f';
const F64: u8 = b'd';
const STRING: u8 = b's';
const ARRAY: u8 = b'[';
const ARRAY_END: u8 = b']';
const OBJECT: u8 = b'{';
const OBJECT_END: u8 = b'}';
const MAP: u8 = b'm';
const DIGEST: u8 = b'h';
const EXACT: u8 = b'N';
// How serde_json writes an exact number token and raw JSON text: a struct of this
// name with one field of this name, holding the text.
const NUMBER_TOKEN: &str = "$serde_json::private::Number";
const RAW_TOKEN: &str = "$serde_json::private::RawValue";

struct Encoder<W> {
    writer: W,
}
impl<W: Write> Encoder<W> {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.writer.write_all(bytes).map_err(Error::io)
    }
    fn tagged(&mut self, tag: u8, bytes: &[u8]) -> Result<()> {
        self.put(&[tag])?;
        self.put(bytes)
    }
    fn unsigned(&mut self, value: u64) -> Result<()> {
        self.tagged(UNSIGNED, &value.to_le_bytes())
    }
    fn signed(&mut self, value: i64) -> Result<()> {
        match u64::try_from(value) {
            Ok(value) => self.unsigned(value),
            Err(_) => self.tagged(NEGATIVE, &value.to_le_bytes()),
        }
    }
    fn text(&mut self, value: &str) -> Result<()> {
        self.tagged(STRING, &(value.len() as u64).to_le_bytes())?;
        self.put(value.as_bytes())
    }
    /// `{"variant":` of an enum variant that carries data.
    fn variant(&mut self, variant: &str) -> Result<()> {
        self.put(&[OBJECT])?;
        self.text(variant)
    }
}

impl<'a, W: Write> ser::Serializer for &'a mut Encoder<W> {
    type Ok = ();
    type Error = Error;
    type SerializeSeq = Items<'a, W>;
    type SerializeTuple = Items<'a, W>;
    type SerializeTupleStruct = Items<'a, W>;
    type SerializeTupleVariant = Items<'a, W>;
    type SerializeMap = Entries<'a, W>;
    type SerializeStruct = Fields<'a, W>;
    type SerializeStructVariant = Fields<'a, W>;

    fn serialize_bool(self, value: bool) -> Result<()> {
        self.put(&[if value { TRUE } else { FALSE }])
    }
    fn serialize_i8(self, value: i8) -> Result<()> {
        self.signed(value.into())
    }
    fn serialize_i16(self, value: i16) -> Result<()> {
        self.signed(value.into())
    }
    fn serialize_i32(self, value: i32) -> Result<()> {
        self.signed(value.into())
    }
    fn serialize_i64(self, value: i64) -> Result<()> {
        self.signed(value)
    }
    fn serialize_i128(self, value: i128) -> Result<()> {
        match i64::try_from(value) {
            Ok(value) => self.signed(value),
            Err(_) => match u128::try_from(value) {
                Ok(value) => self.serialize_u128(value),
                Err(_) => self.tagged(WIDE_NEGATIVE, &value.to_le_bytes()),
            },
        }
    }
    fn serialize_u8(self, value: u8) -> Result<()> {
        self.unsigned(value.into())
    }
    fn serialize_u16(self, value: u16) -> Result<()> {
        self.unsigned(value.into())
    }
    fn serialize_u32(self, value: u32) -> Result<()> {
        self.unsigned(value.into())
    }
    fn serialize_u64(self, value: u64) -> Result<()> {
        self.unsigned(value)
    }
    fn serialize_u128(self, value: u128) -> Result<()> {
        match u64::try_from(value) {
            Ok(value) => self.unsigned(value),
            Err(_) => self.tagged(WIDE_UNSIGNED, &value.to_le_bytes()),
        }
    }
    fn serialize_f32(self, value: f32) -> Result<()> {
        if !value.is_finite() {
            return self.serialize_unit();
        }
        self.tagged(F32, &value.to_bits().to_le_bytes())
    }
    fn serialize_f64(self, value: f64) -> Result<()> {
        if !value.is_finite() {
            return self.serialize_unit();
        }
        self.tagged(F64, &value.to_bits().to_le_bytes())
    }
    fn serialize_char(self, value: char) -> Result<()> {
        self.text(value.encode_utf8(&mut [0; 4]))
    }
    fn serialize_str(self, value: &str) -> Result<()> {
        self.text(value)
    }
    fn serialize_bytes(self, value: &[u8]) -> Result<()> {
        self.put(&[ARRAY])?;
        for byte in value {
            self.unsigned((*byte).into())?;
        }
        self.put(&[ARRAY_END])
    }
    fn serialize_none(self) -> Result<()> {
        self.serialize_unit()
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<()> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<()> {
        self.put(&[NULL])
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<()> {
        self.serialize_unit()
    }
    fn serialize_unit_variant(self, _name: &'static str, _index: u32, variant: &'static str) -> Result<()> {
        self.text(variant)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(self, _name: &'static str, value: &T) -> Result<()> {
        value.serialize(self)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(self, _name: &'static str, _index: u32, variant: &'static str, value: &T) -> Result<()> {
        self.variant(variant)?;
        value.serialize(&mut *self)?;
        self.put(&[OBJECT_END])
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<Items<'a, W>> {
        self.put(&[ARRAY])?;
        Ok(Items { encoder: self, variant: false })
    }
    fn serialize_tuple(self, len: usize) -> Result<Items<'a, W>> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<Items<'a, W>> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(self, _name: &'static str, _index: u32, variant: &'static str, _len: usize) -> Result<Items<'a, W>> {
        self.variant(variant)?;
        self.put(&[ARRAY])?;
        Ok(Items { encoder: self, variant: true })
    }
    fn serialize_map(self, _len: Option<usize>) -> Result<Entries<'a, W>> {
        Ok(Entries { encoder: self, held: Vec::new(), entries: Vec::new() })
    }
    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<Fields<'a, W>> {
        if name == RAW_TOKEN {
            return Err(<Error as ser::Error>::custom("raw JSON text has no typed identity"));
        }
        // An exact number is a scalar: the tag, then its token as the one field writes it.
        if name == NUMBER_TOKEN {
            self.put(&[EXACT])?;
            return Ok(Fields { encoder: self, closes: 0, exact: true });
        }
        self.put(&[OBJECT])?;
        Ok(Fields { encoder: self, closes: 1, exact: false })
    }
    fn serialize_struct_variant(self, _name: &'static str, _index: u32, variant: &'static str, _len: usize) -> Result<Fields<'a, W>> {
        self.variant(variant)?;
        self.put(&[OBJECT])?;
        Ok(Fields { encoder: self, closes: 2, exact: false })
    }
}

/// An array's items, in order. A tuple variant also closes the object that names it.
struct Items<'a, W> {
    encoder: &'a mut Encoder<W>,
    variant: bool,
}
impl<W: Write> Items<'_, W> {
    fn item<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        value.serialize(&mut *self.encoder)
    }
    fn close(self) -> Result<()> {
        self.encoder.put(&[ARRAY_END])?;
        if self.variant {
            self.encoder.put(&[OBJECT_END])?;
        }
        Ok(())
    }
}
impl<W: Write> ser::SerializeSeq for Items<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.item(value)
    }
    fn end(self) -> Result<()> {
        self.close()
    }
}
impl<W: Write> ser::SerializeTuple for Items<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.item(value)
    }
    fn end(self) -> Result<()> {
        self.close()
    }
}
impl<W: Write> ser::SerializeTupleStruct for Items<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.item(value)
    }
    fn end(self) -> Result<()> {
        self.close()
    }
}
impl<W: Write> ser::SerializeTupleVariant for Items<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        self.item(value)
    }
    fn end(self) -> Result<()> {
        self.close()
    }
}

/// A struct's fields, in the order its type declares them; a skipped field writes
/// nothing. A struct variant also closes the object that names it, and an exact
/// number is its token alone.
struct Fields<'a, W> {
    encoder: &'a mut Encoder<W>,
    closes: u8,
    exact: bool,
}
impl<W: Write> Fields<'_, W> {
    fn field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
        if !self.exact {
            self.encoder.text(key)?;
        }
        value.serialize(&mut *self.encoder)
    }
    fn close(self) -> Result<()> {
        for _ in 0..self.closes {
            self.encoder.put(&[OBJECT_END])?;
        }
        Ok(())
    }
}
impl<W: Write> ser::SerializeStruct for Fields<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
        self.field(key, value)
    }
    fn end(self) -> Result<()> {
        self.close()
    }
}
impl<W: Write> ser::SerializeStructVariant for Fields<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_field<T: ?Sized + Serialize>(&mut self, key: &'static str, value: &T) -> Result<()> {
        self.field(key, value)
    }
    fn end(self) -> Result<()> {
        self.close()
    }
}

/// A map's entries, written end to end in `held`: each key, then its value as it
/// is written while that is short, or the value's digest once it is not (`Value`).
/// So a map holds a bounded number of bytes for each entry and a subtree is never
/// copied into its parents. When the map closes it writes its entries in the order
/// of their keys (a key delimits itself, so the order of the bytes is an order of
/// the keys). Of entries with one key the last written stays, as when JSON text
/// with a repeated key is read.
struct Entries<'a, W> {
    encoder: &'a mut Encoder<W>,
    held: Vec<u8>,
    /// Each entry's place in `held`: where its key starts, where its value starts, and where that ends.
    entries: Vec<[usize; 3]>,
}
impl<W: Write> ser::SerializeMap for Entries<'_, W> {
    type Ok = ();
    type Error = Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        let start = self.held.len();
        key.serialize(&mut Encoder { writer: &mut self.held })?;
        self.entries.push([start, self.held.len(), self.held.len()]);
        Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        let entry = self.entries.last_mut().ok_or_else(|| <Error as ser::Error>::custom("a map value without its key"))?;
        let mut encoder = Encoder { writer: Value { held: &mut self.held, start: entry[1], hasher: None } };
        value.serialize(&mut encoder)?;
        if let Some(hasher) = encoder.writer.hasher {
            self.held.push(DIGEST);
            self.held.extend_from_slice(hasher.finalize().as_bytes());
        }
        entry[2] = self.held.len();
        Ok(())
    }
    fn end(self) -> Result<()> {
        let Self { encoder, held, mut entries } = self;
        // A stable order: entries with one key stay in the order they were written.
        entries.sort_by(|a, b| held[a[0]..a[1]].cmp(&held[b[0]..b[1]]));
        encoder.put(&[MAP])?;
        for (index, entry) in entries.iter().enumerate() {
            let key = &held[entry[0]..entry[1]];
            if entries.get(index + 1).is_some_and(|next| &held[next[0]..next[1]] == key) {
                continue;
            }
            encoder.put(&held[entry[0]..entry[2]])?;
        }
        encoder.put(&[OBJECT_END])
    }
}

/// The most bytes of one value a map holds as written. A longer value is hashed.
const HELD: usize = 64;

/// Where a map entry's value is written: at the end of the map's `held` bytes
/// while it is at most `HELD` long, and into a hasher of its own from the write
/// that takes it past that. Which of the two is a matter of the value's length
/// alone, so one value is always written one way.
struct Value<'a> {
    held: &'a mut Vec<u8>,
    start: usize,
    hasher: Option<Box<blake3::Hasher>>,
}
impl Write for Value<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.hasher.is_none() && self.held.len() - self.start + bytes.len() > HELD {
            let mut hasher = Box::new(blake3::Hasher::new());
            hasher.update(&self.held[self.start..]);
            self.held.truncate(self.start);
            self.hasher = Some(hasher);
        }
        match &mut self.hasher {
            Some(hasher) => {
                hasher.update(bytes);
            }
            None => self.held.extend_from_slice(bytes),
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::write;
    use serde_json::{Value, from_str, to_vec};
    use serde::Serialize;
    use std::collections::BTreeMap;

    fn content<T: ?Sized + Serialize>(value: &T) -> Vec<u8> {
        let mut bytes = Vec::new();
        write(&mut bytes, value).unwrap();
        bytes
    }

    #[derive(Serialize)]
    enum Kind {
        Unit,
        Newtype(u8),
        Tuple(u8, u8),
        Struct { a: u8 },
    }
    #[derive(Serialize)]
    struct Wrapped(f32);
    #[derive(Serialize)]
    struct Shape {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        skipped: Option<u32>,
        nullable: Option<u32>,
        #[serde(skip)]
        #[allow(dead_code)]
        origin: u32,
        values: Vec<f32>,
        table: BTreeMap<String, Kind>,
        wrapped: Wrapped,
        #[serde(flatten)]
        rest: BTreeMap<String, Value>,
    }
    fn shape() -> Shape {
        Shape {
            name: "a".into(),
            skipped: None,
            nullable: None,
            origin: 0,
            values: vec![0.1, -0.0, 3.],
            table: BTreeMap::from([("k".to_owned(), Kind::Struct { a: 1 }), ("l".to_owned(), Kind::Unit)]),
            wrapped: Wrapped(0.5),
            rest: BTreeMap::from([("z".to_owned(), from_str(r#"{"b":[1,1.0,"1"],"a":null}"#).unwrap())]),
        }
    }

    /// For values of one type, the content is the same exactly when the sorted JSON is.
    #[test]
    fn typed_values_are_the_same_exactly_when_their_sorted_json_is() {
        let edits: Vec<fn(&mut Shape)> = vec![
            |_| {},
            |s| s.origin = 9,
            |s| s.name = "b".into(),
            |s| s.skipped = Some(0),
            |s| s.nullable = Some(0),
            |s| s.values[1] = 0.0,
            |s| s.values[0] = f32::NAN,
            |s| s.values[0] = f32::INFINITY,
            |s| s.values.swap(0, 2),
            |s| s.values.push(3.),
            |s| { s.table.insert("k".into(), Kind::Newtype(1)); },
            |s| { s.table.insert("k".into(), Kind::Tuple(1, 1)); },
            |s| { s.table.insert("k".into(), Kind::Struct { a: 2 }); },
            |s| { s.table.insert("l".into(), Kind::Newtype(0)); },
            |s| { s.table.remove("l"); },
            |s| s.wrapped = Wrapped(0.25),
            |s| { s.rest.insert("z".into(), from_str(r#"{"a":null,"b":[1,1.0,"1"]}"#).unwrap()); },
            |s| { s.rest.insert("z".into(), from_str(r#"{"a":null,"b":[1.0,1,"1"]}"#).unwrap()); },
            |s| { s.rest.insert("z".into(), from_str(r#"{"a":null,"b":[1,1.0,1]}"#).unwrap()); },
            |s| { s.rest.insert("y".into(), Value::Null); },
        ];
        let shapes: Vec<Shape> = edits.iter().map(|edit| { let mut s = shape(); edit(&mut s); s }).collect();
        let mut same = 0;
        for (i, a) in shapes.iter().enumerate() {
            for (j, b) in shapes.iter().enumerate() {
                let json = to_vec(a).unwrap() == to_vec(b).unwrap();
                assert_eq!(content(a) == content(b), json, "edits {i} and {j}");
                same += usize::from(json && i != j);
            }
        }
        // The unchanged value, the skipped field and the reordered object are one content (six ordered pairs);
        // the two non-finite floats are another (two).
        assert_eq!(same, 8);
    }

    #[test]
    fn a_map_is_its_entries_whatever_order_it_held_them_in() {
        let a: Value = from_str(r#"{"z":[{"b":1,"a":2},3],"a":{"z":0,"a":1}}"#).unwrap();
        let b: Value = from_str(r#"{"a":{"a":1,"z":0},"z":[{"a":2,"b":1},3]}"#).unwrap();
        let c: Value = from_str(r#"{"a":{"a":1,"z":0},"z":[3,{"a":2,"b":1}]}"#).unwrap();
        assert_eq!(content(&a), content(&b));
        assert_ne!(content(&a), content(&c));
        // An entry is its key and its value: moving a value to another key is a change.
        let d: Value = from_str(r#"{"a":1,"b":2}"#).unwrap();
        let e: Value = from_str(r#"{"a":2,"b":1}"#).unwrap();
        assert_ne!(content(&d), content(&e));
    }

    #[test]
    fn values_delimit_themselves() {
        assert_ne!(content(&("ab", "c")), content(&("a", "bc")));
        assert_ne!(content(&(vec![1u8], vec![2u8, 3])), content(&(vec![1u8, 2], vec![3u8])));
        assert_ne!(content(&vec![vec![0u8; 0]; 2]), content(&vec![vec![vec![0u8; 0]; 1]; 1]));
        assert_ne!(content(&""), content(&Option::<&str>::None));
        assert_ne!(content(&0u8), content(&false));
    }

    #[test]
    fn scalars_follow_the_json_they_would_be() {
        assert_eq!(content(&Some(1u8)), content(&1u64));
        assert_eq!(content(&1i8), content(&1u128));
        assert_ne!(content(&-1i8), content(&u64::MAX));
        assert_eq!(content(&-1i8), content(&-1i128));
        assert_ne!(content(&u128::MAX), content(&u64::MAX));
        assert_ne!(content(&i128::MIN), content(&i64::MIN));
        assert_ne!(content(&0.0f32), content(&-0.0f32));
        assert_ne!(content(&0.1f32), content(&f32::from_bits(0.1f32.to_bits() + 1)));
        assert_eq!(content(&f64::NAN), content(&()));
        assert_eq!(content(&'a'), content("a"));
        assert_eq!(content(&Kind::Unit), content("Unit"));
        #[derive(Serialize)]
        struct Newtype {
            #[serde(rename = "Newtype")]
            value: u8,
        }
        #[derive(Serialize)]
        struct Tuple {
            #[serde(rename = "Tuple")]
            value: [u8; 2],
        }
        #[derive(Serialize)]
        struct Struct {
            #[serde(rename = "Struct")]
            value: Inner,
        }
        #[derive(Serialize)]
        struct Inner {
            a: u8,
        }
        assert_eq!(content(&Kind::Newtype(1)), content(&Newtype { value: 1 }));
        assert_eq!(content(&Kind::Tuple(1, 2)), content(&Tuple { value: [1, 2] }));
        assert_eq!(content(&Kind::Struct { a: 1 }), content(&Struct { value: Inner { a: 1 } }));
        // Exact number tokens stay apart, as they do in sorted text.
        let one: Value = from_str("1").unwrap();
        let one_point: Value = from_str("1.0").unwrap();
        assert_ne!(content(&one), content(&one_point));
    }

    /// A key written twice keeps its last value, as the sorted text does once it is read back.
    #[test]
    fn a_repeated_key_keeps_its_last_value() {
        #[derive(Serialize)]
        struct Repeats {
            a: u8,
            #[serde(flatten)]
            rest: BTreeMap<String, u8>,
        }
        let repeats = |a: u8, rest: &[(&str, u8)]| Repeats { a, rest: rest.iter().map(|(key, value)| ((*key).to_owned(), *value)).collect() };
        let values = [
            repeats(1, &[("a", 2)]),
            repeats(2, &[("a", 1)]),
            repeats(9, &[("a", 2)]),
            repeats(2, &[]),
            repeats(2, &[("b", 1)]),
        ];
        for (i, a) in values.iter().enumerate() {
            for (j, b) in values.iter().enumerate() {
                assert_eq!(content(a) == content(b), to_vec(a).unwrap() == to_vec(b).unwrap(), "values {i} and {j}");
            }
        }
        assert_ne!(content(&values[0]), content(&values[1]));
        assert_eq!(content(&values[0]), content(&values[2]));
        assert_eq!(content(&values[0]), content(&values[3]));
    }

    #[test]
    fn an_exact_number_is_not_the_object_serde_json_writes_it_through() {
        let number: Value = from_str("1").unwrap();
        let mut object = serde_json::Map::new();
        object.insert(super::NUMBER_TOKEN.to_owned(), Value::String("1".to_owned()));
        assert_ne!(content(&number), content(&Value::Object(object)));
        assert_ne!(content(&number), content("1"));
    }

    /// A value under many maps is its own content at every depth, and is read once: no map holds its bytes.
    #[test]
    fn a_value_under_nested_maps_is_content_at_every_depth() {
        let nested = |leaf: &str, depth: usize| {
            let mut value: Value = from_str(leaf).unwrap();
            for level in 0..depth {
                let mut object = serde_json::Map::new();
                object.insert("children".to_owned(), Value::Array(vec![value, Value::from(level)]));
                object.insert("name".to_owned(), Value::from("g"));
                value = Value::Object(object);
            }
            value
        };
        let leaf = r#"{"d":"M0 0H20V20H0Z","fill":"red"}"#;
        assert_eq!(content(&nested(leaf, 100)), content(&nested(r#"{"fill":"red","d":"M0 0H20V20H0Z"}"#, 100)));
        assert_ne!(content(&nested(leaf, 100)), content(&nested(r#"{"d":"M0 0H20V20H0Z","fill":"lime"}"#, 100)));
        assert_ne!(content(&nested(leaf, 100)), content(&nested(leaf, 99)));
        // A map writes its keys and a digest for a value that is not short, whatever the value holds.
        let small = content(&nested(leaf, 2)).len();
        assert_eq!(content(&nested(leaf, 100)).len(), small);
        // A value is held or hashed by its length alone: either side of the bound, a change is a change.
        for length in [super::HELD - 12, super::HELD - 9, super::HELD - 8, super::HELD + 4] {
            let text = |fill: char| BTreeMap::from([("k", fill.to_string().repeat(length))]);
            assert_eq!(content(&text('a')), content(&text('a')));
            assert_ne!(content(&text('a')), content(&text('b')));
            assert_ne!(content(&text('a')), content(&BTreeMap::from([("k", "a".repeat(length + 1))])));
        }
    }
}
