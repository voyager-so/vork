//! Typed scene values: what a node's fields are held as.
//!
//! [`Value`], [`Number`] and [`Map`] have the shape and the methods of `serde_json`'s types of the same names, so
//! the scene layer and the engine read fields exactly as they did. The storage is what differs:
//!
//! - a [`Number`] is a number (`u64`, `i64` or `f64`), never its decimal text. A build that turns on serde_json's
//!   `arbitrary_precision` anywhere in the dependency graph (vork does, for exact clocks) makes every
//!   `serde_json::Number` a heap string that is parsed on each read; these are parsed once, at the boundary.
//! - a [`Map`]'s keys are [`Key`]s: a known field name is an enum value ([`K`]), compared as an integer and held
//!   without a heap allocation. Any other name is shared text.
//!
//! JSON stays the importer and the exporter: `Value` implements serde's `Serialize` and `Deserialize`, converts
//! from and to `serde_json::Value`, and this module re-exports serde_json's reading and writing functions, so
//! `value::from_str`, `value::to_string`, `value::from_value`, `value::to_value` and `value::json!` are what
//! their serde_json namesakes were.
//!
//! Numbers read the same as before the change. An integer token is an integer; any other token is the `f64`
//! nearest its decimal text (`str::parse`, correctly rounded). [`Number::from_f32_shortest`] is the one boundary
//! kept on purpose: an `f32` written into a document was the decimal text serde_json writes for it, read back
//! as `f64`. A token no finite `f64` holds reads as no number, as it did.

pub mod de;
mod keys;
pub mod ser;

pub use keys::K;
pub use serde_json::{
    from_reader, from_slice, from_str, to_string, to_string_pretty, to_vec, to_vec_pretty, to_writer, to_writer_pretty,
    Error, Result,
};
#[doc(hidden)]
pub use serde_json as __serde_json;

use indexmap::IndexMap;
use rustc_hash::FxBuildHasher;
use serde::de::{DeserializeOwned, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::borrow::{Borrow, Cow};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// serde_json's `json!`, giving a typed [`Value`]: the same grammar (JSON with Rust expressions as values and
/// parenthesised expressions as keys), built as typed values directly. An interpolated expression is written
/// by [`to_value`].
#[macro_export]
#[doc(hidden)]
macro_rules! __moda_value_json {
    // Arrays: `[$($elems:expr),*]` collects the elements read so far; the rest is munched one value at a time.
    (@array [$($elems:expr,)*]) => {
        ::std::vec![$($elems,)*]
    };
    (@array [$($elems:expr),*]) => {
        ::std::vec![$($elems),*]
    };
    (@array [$($elems:expr,)*] null $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!(null)] $($rest)*)
    };
    (@array [$($elems:expr,)*] true $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!(true)] $($rest)*)
    };
    (@array [$($elems:expr,)*] false $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!(false)] $($rest)*)
    };
    (@array [$($elems:expr,)*] [$($array:tt)*] $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!([$($array)*])] $($rest)*)
    };
    (@array [$($elems:expr,)*] {$($map:tt)*} $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!({$($map)*})] $($rest)*)
    };
    (@array [$($elems:expr,)*] $next:expr, $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!($next),] $($rest)*)
    };
    (@array [$($elems:expr,)*] $last:expr) => {
        $crate::__moda_value_json!(@array [$($elems,)* $crate::__moda_value_json!($last)])
    };
    (@array [$($elems:expr),*] , $($rest:tt)*) => {
        $crate::__moda_value_json!(@array [$($elems,)*] $($rest)*)
    };

    // Objects: `($($key:tt)*)` collects a key's tokens up to its colon; `[$($key:tt)+] ($value:expr)` is an
    // entry ready to insert.
    (@object $object:ident () () ()) => {};
    (@object $object:ident [$($key:tt)+] ($value:expr) , $($rest:tt)*) => {
        let _ = $object.insert($crate::value::__json_key!($($key)+), $value);
        $crate::__moda_value_json!(@object $object () ($($rest)*) ($($rest)*));
    };
    (@object $object:ident [$($key:tt)+] ($value:expr)) => {
        let _ = $object.insert($crate::value::__json_key!($($key)+), $value);
    };
    (@object $object:ident ($($key:tt)+) (: null $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!(null)) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: true $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!(true)) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: false $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!(false)) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: [$($array:tt)*] $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!([$($array)*])) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: {$($map:tt)*} $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!({$($map)*})) $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: $value:expr , $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!($value)) , $($rest)*);
    };
    (@object $object:ident ($($key:tt)+) (: $value:expr) $copy:tt) => {
        $crate::__moda_value_json!(@object $object [$($key)+] ($crate::__moda_value_json!($value)));
    };
    // A key written as a parenthesised expression.
    (@object $object:ident () (($key:expr) : $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object ($key) (: $($rest)*) (: $($rest)*));
    };
    // One more token of the key.
    (@object $object:ident ($($key:tt)*) ($tt:tt $($rest:tt)*) $copy:tt) => {
        $crate::__moda_value_json!(@object $object ($($key)* $tt) ($($rest)*) ($($rest)*));
    };

    (null) => {
        $crate::value::Value::Null
    };
    (true) => {
        $crate::value::Value::Bool(true)
    };
    (false) => {
        $crate::value::Value::Bool(false)
    };
    ([]) => {
        $crate::value::Value::Array(::std::vec::Vec::new())
    };
    ([ $($tt:tt)+ ]) => {
        $crate::value::Value::Array($crate::__moda_value_json!(@array [] $($tt)+))
    };
    ({}) => {
        $crate::value::Value::Object($crate::value::Map::new())
    };
    ({ $($tt:tt)+ }) => {
        $crate::value::Value::Object({
            let mut object = $crate::value::Map::new();
            $crate::__moda_value_json!(@object object () ($($tt)+) ($($tt)+));
            object
        })
    };
    ($other:expr) => {
        $crate::value::to_value(&$other).unwrap()
    };
}
pub use crate::__moda_value_json as json;

/// A `json!` object key: anything serde_json's `json!` takes, which is anything that converts into a `String`.
/// A key written as text (every literal) becomes a [`Key`] without that `String`.
#[macro_export]
#[doc(hidden)]
macro_rules! __moda_value_json_key {
    ($($key:tt)+) => {{
        #[allow(unused_imports)]
        use $crate::value::JsonKeyFromString as _;
        $crate::value::JsonKey($($key)+).key()
    }};
}
#[doc(hidden)]
pub use crate::__moda_value_json_key as __json_key;

/// What `json!` wraps an object key in to choose its conversion: the method on text itself is found before
/// the one every `Into<String>` has.
#[doc(hidden)]
pub struct JsonKey<T>(pub T);
impl JsonKey<&str> {
    pub fn key(self) -> Key {
        Key::new(self.0)
    }
}
#[doc(hidden)]
pub trait JsonKeyFromString {
    fn key(self) -> Key;
}
impl<T: Into<String>> JsonKeyFromString for JsonKey<T> {
    fn key(self) -> Key {
        Key::from(self.0.into())
    }
}

/// `serde_json::to_value`, giving a typed [`Value`]: written straight into one ([`ser::Serializer`]), by
/// serde_json's rules.
pub fn to_value<T: Serialize>(value: T) -> Result<Value> {
    value.serialize(ser::Serializer)
}

/// `serde_json::from_value`, from a typed [`Value`]: the record reads the value's numbers as numbers.
pub fn from_value<T: DeserializeOwned>(value: Value) -> Result<T> {
    from_value_ref(&value)
}

/// [`from_value`] of a borrowed value.
pub fn from_value_ref<T: DeserializeOwned>(value: &Value) -> Result<T> {
    T::deserialize(value).map_err(|error| <Error as serde::de::Error>::custom(error))
}

// ---------------------------------------------------------------------------
// Key
// ---------------------------------------------------------------------------

/// A field name: a known one ([`K`]) or any other text. It reads as a `str` everywhere a `String` key did.
///
/// The representation is private: a name on the known list is always held as the enum value (there is no way to
/// build a key that holds a known name as text), so two keys are equal exactly when their representations are.
#[derive(Clone)]
pub struct Key(KeyRepr);

#[derive(Clone)]
enum KeyRepr {
    Known(K),
    Other(Arc<str>),
}

impl Key {
    pub fn new(text: &str) -> Key {
        match K::lookup(text) {
            Some(known) => Key(KeyRepr::Known(known)),
            None => Key(KeyRepr::Other(Arc::from(text))),
        }
    }

    pub fn as_str(&self) -> &str {
        match &self.0 {
            KeyRepr::Known(known) => known.as_str(),
            KeyRepr::Other(text) => text,
        }
    }

    /// The known name this is, if it is one.
    pub fn known(&self) -> Option<K> {
        match &self.0 {
            KeyRepr::Known(known) => Some(*known),
            KeyRepr::Other(_) => None,
        }
    }
}

impl std::ops::Deref for Key {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}
impl Borrow<str> for Key {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}
impl AsRef<str> for Key {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl AsRef<str> for K {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}
impl PartialEq for Key {
    fn eq(&self, other: &Key) -> bool {
        match (&self.0, &other.0) {
            (KeyRepr::Known(a), KeyRepr::Known(b)) => a == b,
            (KeyRepr::Other(a), KeyRepr::Other(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for Key {}
/// As the text hashes, so a map of keys answers a `str` lookup.
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Key) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Key) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl PartialEq<str> for Key {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl PartialEq<&str> for Key {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}
impl PartialEq<String> for Key {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}
impl PartialEq<Key> for str {
    fn eq(&self, other: &Key) -> bool {
        self == other.as_str()
    }
}
impl PartialEq<Key> for &str {
    fn eq(&self, other: &Key) -> bool {
        *self == other.as_str()
    }
}
impl PartialEq<Key> for String {
    fn eq(&self, other: &Key) -> bool {
        self.as_str() == other.as_str()
    }
}
impl PartialEq<K> for Key {
    fn eq(&self, other: &K) -> bool {
        matches!(&self.0, KeyRepr::Known(known) if known == other)
    }
}
impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
impl fmt::Debug for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}
impl From<K> for Key {
    fn from(known: K) -> Key {
        Key(KeyRepr::Known(known))
    }
}
impl From<&str> for Key {
    fn from(text: &str) -> Key {
        Key::new(text)
    }
}
impl From<String> for Key {
    fn from(text: String) -> Key {
        match K::lookup(&text) {
            Some(known) => Key(KeyRepr::Known(known)),
            None => Key(KeyRepr::Other(Arc::from(text))),
        }
    }
}
impl From<&String> for Key {
    fn from(text: &String) -> Key {
        Key::new(text)
    }
}
impl From<&Key> for Key {
    fn from(key: &Key) -> Key {
        key.clone()
    }
}
impl From<Cow<'_, str>> for Key {
    fn from(text: Cow<'_, str>) -> Key {
        Key::new(&text)
    }
}
impl From<Key> for String {
    fn from(key: Key) -> String {
        key.as_str().to_string()
    }
}
impl From<&Key> for String {
    fn from(key: &Key) -> String {
        key.as_str().to_string()
    }
}
impl Serialize for Key {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}
impl<'de> Deserialize<'de> for Key {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Key, D::Error> {
        struct KeyVisitor;
        impl Visitor<'_> for KeyVisitor {
            type Value = Key;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string key")
            }
            fn visit_str<E: serde::de::Error>(self, text: &str) -> std::result::Result<Key, E> {
                Ok(Key::new(text))
            }
            fn visit_string<E: serde::de::Error>(self, text: String) -> std::result::Result<Key, E> {
                Ok(Key::from(text))
            }
        }
        deserializer.deserialize_str(KeyVisitor)
    }
}

// ---------------------------------------------------------------------------
// Number
// ---------------------------------------------------------------------------

/// A JSON number, held as a number: an integer token as `u64` or `i64`, anything else as `f64`.
///
/// A token whose value no finite `f64` holds (`1e400`) is kept as its text: it reads as no number at all
/// (`as_f64` is `None`, so a reader's default applies, as it did when numbers were text) and writes back as the
/// token it was.
#[derive(Clone)]
pub struct Number(N);

#[derive(Clone)]
enum N {
    PosInt(u64),
    /// Always negative.
    NegInt(i64),
    /// Always finite.
    Float(f64),
    /// A JSON number token outside `f64`'s range.
    OutOfRange(Arc<str>),
}

/// The map key serde_json's `arbitrary_precision` wraps a number token in when it hands one to a generic visitor.
const NUMBER_TOKEN: &str = "$serde_json::private::Number";

/// Whether this build's serde_json keeps number tokens as text and so reserves [`NUMBER_TOKEN`] as an object's
/// first key (`arbitrary_precision`, a feature any crate in the build can turn on). Where it does not, the name
/// is an ordinary key, here as in serde_json.
fn number_marker_reserved() -> bool {
    static RESERVED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *RESERVED.get_or_init(|| {
        matches!(
            serde_json::from_str::<serde_json::Value>(r#"{"$serde_json::private::Number":"1"}"#),
            Ok(serde_json::Value::Number(_))
        )
    })
}

/// Whether `token` is a JSON number (RFC 8259): `-`? digits, an optional fraction, an optional exponent.
fn is_json_number(token: &str) -> bool {
    let bytes = token.as_bytes();
    let mut at = usize::from(bytes.first() == Some(&b'-'));
    let digits = |at: &mut usize| {
        let start = *at;
        while bytes.get(*at).is_some_and(u8::is_ascii_digit) {
            *at += 1;
        }
        *at > start
    };
    let int_start = at;
    if !digits(&mut at) || (bytes[int_start] == b'0' && at - int_start > 1) {
        return false;
    }
    if bytes.get(at) == Some(&b'.') {
        at += 1;
        if !digits(&mut at) {
            return false;
        }
    }
    if matches!(bytes.get(at), Some(b'e' | b'E')) {
        at += 1;
        if matches!(bytes.get(at), Some(b'+' | b'-')) {
            at += 1;
        }
        if !digits(&mut at) {
            return false;
        }
    }
    at == bytes.len()
}

impl Number {
    /// The number a JSON number token spells: an integer when the token is one, else the nearest `f64`. `None`
    /// for text that is not a JSON number (`NaN`, `inf`, `+1`, `01`), and for a token outside the range of `f64` in a
    /// build whose serde_json cannot hold one.
    pub fn from_token(token: &str) -> Option<Number> {
        if !is_json_number(token) {
            return None;
        }
        if let Ok(u) = token.parse::<u64>() {
            return Some(Number(N::PosInt(u)));
        }
        if let Ok(i) = token.parse::<i64>() {
            // `-0` is negative zero, as a float parse reads it.
            return Some(if i < 0 { Number(N::NegInt(i)) } else { Number(N::Float(-0.0)) });
        }
        match token.parse::<f64>() {
            Ok(f) if f.is_finite() => Some(Number(N::Float(f))),
            // Kept only where serde_json can hold the token too (a build that keeps number tokens as text):
            // a number this build could neither import nor write back is not constructed.
            // The text kept is serde_json's spelling of the token (it writes `1e400` as `1e+400`), so the same
            // number is the same text however it arrived.
            _ => token.parse::<serde_json::Number>().ok().map(|n| Number(N::OutOfRange(Arc::from(n.to_string())))),
        }
    }

    /// A finite float; `None` for NaN and the infinities (they are not JSON numbers).
    pub fn from_f64(f: f64) -> Option<Number> {
        f.is_finite().then_some(Number(N::Float(f)))
    }

    /// The `f64` an `f32` reads back as after being written to a document: what `serde_json::Value::from(f32)`
    /// reads as in this build. Where serde_json keeps number tokens as text (vork's build), that is the `f64`
    /// nearest the token serde_json's own formatter writes for the `f32`, which is not the widened `f32`. The
    /// formatter must be serde_json's: shortest round-trip text is not unique, and another formatter breaks a tie
    /// the other way (`2^-12` as an `f32` is `0.00024414062` here and `0.00024414063` from Rust's `Display`), which
    /// would move a placement by one unit in the last place. `None` for a non-finite value.
    pub fn from_f32_shortest(f: f32) -> Option<Number> {
        match serde_json::Value::from(f) {
            serde_json::Value::Number(n) => Some(Number::from(&n)),
            _ => None,
        }
    }

    /// The number as an `f64`; `None` for a token outside `f64`'s range.
    pub fn as_f64(&self) -> Option<f64> {
        match &self.0 {
            N::PosInt(u) => Some(*u as f64),
            N::NegInt(i) => Some(*i as f64),
            N::Float(f) => Some(*f),
            N::OutOfRange(_) => None,
        }
    }

    /// The number narrowed to `f32`, when that is finite. An integer narrows from the integer itself, not through
    /// `f64` (two roundings can land on the other side of an `f32` midpoint). A float narrows from the stored
    /// `f64`: a decimal token that sits between an `f32` midpoint and its nearest `f64` narrows as that `f64`
    /// does, which is the one place this differs from parsing the token as an `f32`.
    pub fn as_f32(&self) -> Option<f32> {
        let narrowed = match &self.0 {
            N::PosInt(u) => *u as f32,
            N::NegInt(i) => *i as f32,
            N::Float(f) => *f as f32,
            N::OutOfRange(_) => return None,
        };
        narrowed.is_finite().then_some(narrowed)
    }

    pub fn as_u64(&self) -> Option<u64> {
        match &self.0 {
            N::PosInt(u) => Some(*u),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match &self.0 {
            N::PosInt(u) => i64::try_from(*u).ok(),
            N::NegInt(i) => Some(*i),
            N::Float(_) | N::OutOfRange(_) => None,
        }
    }

    /// Whether [`Number::as_f64`] gives the number as stored (a float that is not an integer token). An
    /// out-of-range token is none of `is_f64`, `is_i64`, `is_u64`.
    pub fn is_f64(&self) -> bool {
        matches!(self.0, N::Float(_))
    }

    pub fn is_u64(&self) -> bool {
        matches!(self.0, N::PosInt(_))
    }

    pub fn is_i64(&self) -> bool {
        match &self.0 {
            N::PosInt(u) => *u <= i64::MAX as u64,
            N::NegInt(_) => true,
            N::Float(_) | N::OutOfRange(_) => false,
        }
    }

    /// As serde_json holds it. Always `Some` for a number built through this type: an out-of-range token exists
    /// only where serde_json can hold it.
    pub fn to_json(&self) -> Option<serde_json::Number> {
        match &self.0 {
            N::PosInt(u) => Some((*u).into()),
            N::NegInt(i) => Some((*i).into()),
            N::Float(f) => serde_json::Number::from_f64(*f),
            N::OutOfRange(token) => token.parse::<serde_json::Number>().ok(),
        }
    }
}

impl From<&serde_json::Number> for Number {
    fn from(n: &serde_json::Number) -> Number {
        if let Some(u) = n.as_u64() {
            Number(N::PosInt(u))
        } else if let Some(i) = n.as_i64() {
            if i < 0 {
                Number(N::NegInt(i))
            } else {
                Number(N::Float(-0.0))
            }
        } else {
            match n.as_f64() {
                Some(f) if f.is_finite() => Number(N::Float(f)),
                // A token outside f64's range (only a build that keeps tokens as text holds one).
                _ => Number(N::OutOfRange(Arc::from(n.to_string()))),
            }
        }
    }
}
impl From<serde_json::Number> for Number {
    fn from(n: serde_json::Number) -> Number {
        Number::from(&n)
    }
}

/// As serde_json compares numbers it holds as numbers: an integer never equals a float. Out-of-range tokens are
/// equal when their text is.
impl PartialEq for Number {
    fn eq(&self, other: &Number) -> bool {
        match (&self.0, &other.0) {
            (N::PosInt(a), N::PosInt(b)) => a == b,
            (N::NegInt(a), N::NegInt(b)) => a == b,
            (N::Float(a), N::Float(b)) => a == b,
            (N::OutOfRange(a), N::OutOfRange(b)) => a == b,
            _ => false,
        }
    }
}
impl Eq for Number {}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.0, self.to_json()) {
            (N::OutOfRange(token), _) => f.write_str(token),
            (_, Some(n)) => fmt::Display::fmt(&n, f),
            (_, None) => f.write_str("null"),
        }
    }
}
impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Number({self})")
    }
}

macro_rules! number_from_unsigned {
    ($($ty:ty)*) => {$(
        impl From<$ty> for Number {
            fn from(u: $ty) -> Number {
                Number(N::PosInt(u as u64))
            }
        }
    )*};
}
macro_rules! number_from_signed {
    ($($ty:ty)*) => {$(
        impl From<$ty> for Number {
            fn from(i: $ty) -> Number {
                if i < 0 {
                    Number(N::NegInt(i as i64))
                } else {
                    Number(N::PosInt(i as u64))
                }
            }
        }
    )*};
}
number_from_unsigned!(u8 u16 u32 u64 usize);
number_from_signed!(i8 i16 i32 i64 isize);

impl Serialize for Number {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match &self.0 {
            N::PosInt(u) => serializer.serialize_u64(*u),
            N::NegInt(i) => serializer.serialize_i64(*i),
            N::Float(f) => serializer.serialize_f64(*f),
            // As the token it was, through serde_json's own number.
            N::OutOfRange(token) => match token.parse::<serde_json::Number>() {
                Ok(n) => n.serialize(serializer),
                Err(_) => Err(serde::ser::Error::custom("a number outside the range this build can write")),
            },
        }
    }
}

/// Reads the text of an `arbitrary_precision` number token.
struct NumberTokenText;
impl<'de> DeserializeSeed<'de> for NumberTokenText {
    type Value = Number;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> std::result::Result<Number, D::Error> {
        struct TextVisitor;
        impl Visitor<'_> for TextVisitor {
            type Value = Number;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a number token")
            }
            fn visit_str<E: serde::de::Error>(self, text: &str) -> std::result::Result<Number, E> {
                Number::from_token(text).ok_or_else(|| E::custom("not a JSON number"))
            }
        }
        deserializer.deserialize_str(TextVisitor)
    }
}

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Number, D::Error> {
        struct NumberVisitor;
        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = Number;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON number")
            }
            fn visit_i64<E: serde::de::Error>(self, i: i64) -> std::result::Result<Number, E> {
                Ok(i.into())
            }
            fn visit_u64<E: serde::de::Error>(self, u: u64) -> std::result::Result<Number, E> {
                Ok(u.into())
            }
            fn visit_f64<E: serde::de::Error>(self, f: f64) -> std::result::Result<Number, E> {
                Number::from_f64(f).ok_or_else(|| E::custom("not a JSON number"))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> std::result::Result<Number, A::Error> {
                match access.next_key_seed(KeyClassifier)? {
                    Some(KeyClass::Number) => access.next_value_seed(NumberTokenText),
                    _ => Err(serde::de::Error::custom("expected a JSON number")),
                }
            }
        }
        deserializer.deserialize_any(NumberVisitor)
    }
}

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------

/// A scene value. The variants and methods are `serde_json::Value`'s.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Value {
    #[default]
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(Map),
}

static NULL: Value = Value::Null;

impl Value {
    pub fn get<I: Index>(&self, index: I) -> Option<&Value> {
        index.index_into(self)
    }

    pub fn get_mut<I: Index>(&mut self, index: I) -> Option<&mut Value> {
        index.index_into_mut(self)
    }

    pub fn is_object(&self) -> bool {
        matches!(self, Value::Object(_))
    }
    pub fn as_object(&self) -> Option<&Map> {
        match self {
            Value::Object(map) => Some(map),
            _ => None,
        }
    }
    pub fn as_object_mut(&mut self) -> Option<&mut Map> {
        match self {
            Value::Object(map) => Some(map),
            _ => None,
        }
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Value::Array(_))
    }
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(list) => Some(list),
            _ => None,
        }
    }
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(list) => Some(list),
            _ => None,
        }
    }

    pub fn is_string(&self) -> bool {
        matches!(self, Value::String(_))
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(text) => Some(text),
            _ => None,
        }
    }

    pub fn is_number(&self) -> bool {
        matches!(self, Value::Number(_))
    }
    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }
    pub fn is_i64(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_i64())
    }
    pub fn is_u64(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_u64())
    }
    pub fn is_f64(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_f64())
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(n) => n.as_i64(),
            _ => None,
        }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(n) => n.as_u64(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }

    pub fn is_boolean(&self) -> bool {
        matches!(self, Value::Bool(_))
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
    pub fn as_null(&self) -> Option<()> {
        match self {
            Value::Null => Some(()),
            _ => None,
        }
    }

    /// Look up by JSON Pointer (RFC 6901).
    pub fn pointer(&self, pointer: &str) -> Option<&Value> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        pointer.split('/').skip(1).map(|x| x.replace("~1", "/").replace("~0", "~")).try_fold(self, |target, token| {
            match target {
                Value::Object(map) => map.get(&token),
                Value::Array(list) => parse_index(&token).and_then(|x| list.get(x)),
                _ => None,
            }
        })
    }

    /// [`Value::pointer`], mutably.
    pub fn pointer_mut(&mut self, pointer: &str) -> Option<&mut Value> {
        if pointer.is_empty() {
            return Some(self);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        pointer.split('/').skip(1).map(|x| x.replace("~1", "/").replace("~0", "~")).try_fold(self, |target, token| {
            match target {
                Value::Object(map) => map.get_mut(&token),
                Value::Array(list) => parse_index(&token).and_then(move |x| list.get_mut(x)),
                _ => None,
            }
        })
    }

    /// Take the value, leaving `Null`.
    pub fn take(&mut self) -> Value {
        std::mem::replace(self, Value::Null)
    }

    /// Sort every object's keys, recursively.
    pub fn sort_all_objects(&mut self) {
        match self {
            Value::Object(map) => {
                map.sort_keys();
                map.values_mut().for_each(Value::sort_all_objects);
            }
            Value::Array(list) => list.iter_mut().for_each(Value::sort_all_objects),
            _ => {}
        }
    }

    /// The value as serde_json holds it.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(*b),
            Value::Number(n) => n.to_json().map_or(serde_json::Value::Null, serde_json::Value::Number),
            Value::String(text) => serde_json::Value::String(text.clone()),
            Value::Array(list) => serde_json::Value::Array(list.iter().map(Value::to_json).collect()),
            Value::Object(map) => serde_json::Value::Object(map.to_json()),
        }
    }
}

fn parse_index(text: &str) -> Option<usize> {
    if text.starts_with('+') || (text.starts_with('0') && text.len() != 1) {
        return None;
    }
    text.parse().ok()
}

impl From<serde_json::Value> for Value {
    fn from(value: serde_json::Value) -> Value {
        match value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => Value::Number(Number::from(&n)),
            serde_json::Value::String(text) => Value::String(text),
            serde_json::Value::Array(list) => Value::Array(list.into_iter().map(Value::from).collect()),
            serde_json::Value::Object(map) => Value::Object(Map::from(map)),
        }
    }
}
impl From<&serde_json::Value> for Value {
    fn from(value: &serde_json::Value) -> Value {
        match value {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(*b),
            serde_json::Value::Number(n) => Value::Number(Number::from(n)),
            serde_json::Value::String(text) => Value::String(text.clone()),
            serde_json::Value::Array(list) => Value::Array(list.iter().map(Value::from).collect()),
            serde_json::Value::Object(map) => {
                Value::Object(map.iter().map(|(key, value)| (Key::new(key), Value::from(value))).collect())
            }
        }
    }
}
impl From<Value> for serde_json::Value {
    fn from(value: Value) -> serde_json::Value {
        match value {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(b),
            Value::Number(n) => n.to_json().map_or(serde_json::Value::Null, serde_json::Value::Number),
            Value::String(text) => serde_json::Value::String(text),
            Value::Array(list) => serde_json::Value::Array(list.into_iter().map(serde_json::Value::from).collect()),
            Value::Object(map) => serde_json::Value::Object(
                map.into_iter().map(|(key, value)| (String::from(key), serde_json::Value::from(value))).collect(),
            ),
        }
    }
}
impl From<&Value> for serde_json::Value {
    fn from(value: &Value) -> serde_json::Value {
        value.to_json()
    }
}

/// Compact JSON; `{:#}` is pretty-printed.
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = if f.alternate() { serde_json::to_string_pretty(self) } else { serde_json::to_string(self) };
        f.write_str(&text.map_err(|_| fmt::Error)?)
    }
}

macro_rules! value_from_integer {
    ($($ty:ty)*) => {$(
        impl From<$ty> for Value {
            fn from(n: $ty) -> Value {
                Value::Number(n.into())
            }
        }
    )*};
}
value_from_integer!(u8 u16 u32 u64 usize i8 i16 i32 i64 isize);

/// As `serde_json::Value::from(f32)` reads back in this build (see [`Number::from_f32_shortest`]).
impl From<f32> for Value {
    fn from(f: f32) -> Value {
        Value::from(serde_json::Value::from(f))
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Value {
        Number::from_f64(f).map_or(Value::Null, Value::Number)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Value {
        Value::Bool(b)
    }
}
impl From<String> for Value {
    fn from(text: String) -> Value {
        Value::String(text)
    }
}
impl From<&str> for Value {
    fn from(text: &str) -> Value {
        Value::String(text.to_string())
    }
}
impl From<&String> for Value {
    fn from(text: &String) -> Value {
        Value::String(text.clone())
    }
}
impl From<Cow<'_, str>> for Value {
    fn from(text: Cow<'_, str>) -> Value {
        Value::String(text.into_owned())
    }
}
impl From<Number> for Value {
    fn from(n: Number) -> Value {
        Value::Number(n)
    }
}
impl From<Map> for Value {
    fn from(map: Map) -> Value {
        Value::Object(map)
    }
}
impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(list: Vec<T>) -> Value {
        Value::Array(list.into_iter().map(Into::into).collect())
    }
}
impl<T: Clone + Into<Value>> From<&[T]> for Value {
    fn from(list: &[T]) -> Value {
        Value::Array(list.iter().cloned().map(Into::into).collect())
    }
}
impl<T: Into<Value>, const N: usize> From<[T; N]> for Value {
    fn from(list: [T; N]) -> Value {
        Value::Array(list.into_iter().map(Into::into).collect())
    }
}
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(value: Option<T>) -> Value {
        value.map_or(Value::Null, Into::into)
    }
}
impl From<()> for Value {
    fn from((): ()) -> Value {
        Value::Null
    }
}
impl<T: Into<Value>> FromIterator<T> for Value {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Value {
        Value::Array(iter.into_iter().map(Into::into).collect())
    }
}
impl<Q: Into<Key>, V: Into<Value>> FromIterator<(Q, V)> for Value {
    fn from_iter<I: IntoIterator<Item = (Q, V)>>(iter: I) -> Value {
        Value::Object(iter.into_iter().map(|(key, value)| (key.into(), value.into())).collect())
    }
}

impl PartialEq<str> for Value {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == Some(other)
    }
}
impl PartialEq<&str> for Value {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == Some(*other)
    }
}
impl PartialEq<String> for Value {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == Some(other.as_str())
    }
}
impl PartialEq<Value> for str {
    fn eq(&self, other: &Value) -> bool {
        other.as_str() == Some(self)
    }
}
impl PartialEq<Value> for &str {
    fn eq(&self, other: &Value) -> bool {
        other.as_str() == Some(*self)
    }
}
impl PartialEq<Value> for String {
    fn eq(&self, other: &Value) -> bool {
        other.as_str() == Some(self.as_str())
    }
}
impl PartialEq<bool> for Value {
    fn eq(&self, other: &bool) -> bool {
        self.as_bool() == Some(*other)
    }
}
impl PartialEq<Value> for bool {
    fn eq(&self, other: &Value) -> bool {
        other.as_bool() == Some(*self)
    }
}

macro_rules! value_eq_number {
    ($read:ident as $wide:ty: $($ty:ty)*) => {$(
        impl PartialEq<$ty> for Value {
            fn eq(&self, other: &$ty) -> bool {
                self.$read() == Some(*other as $wide)
            }
        }
        impl PartialEq<Value> for $ty {
            fn eq(&self, other: &Value) -> bool {
                other.$read() == Some(*self as $wide)
            }
        }
        impl PartialEq<$ty> for &Value {
            fn eq(&self, other: &$ty) -> bool {
                self.$read() == Some(*other as $wide)
            }
        }
        impl PartialEq<$ty> for &mut Value {
            fn eq(&self, other: &$ty) -> bool {
                self.$read() == Some(*other as $wide)
            }
        }
    )*};
}
value_eq_number!(as_i64 as i64: i8 i16 i32 i64 isize);
value_eq_number!(as_u64 as u64: u8 u16 u32 u64 usize);
value_eq_number!(as_f64 as f64: f64);

/// The value as an `f32`, when it is a number whose narrowing is finite ([`Number::as_f32`]).
fn as_finite_f32(value: &Value) -> Option<f32> {
    value.as_number().and_then(Number::as_f32)
}
impl PartialEq<f32> for Value {
    fn eq(&self, other: &f32) -> bool {
        as_finite_f32(self) == Some(*other)
    }
}
impl PartialEq<Value> for f32 {
    fn eq(&self, other: &Value) -> bool {
        as_finite_f32(other) == Some(*self)
    }
}
impl PartialEq<f32> for &Value {
    fn eq(&self, other: &f32) -> bool {
        as_finite_f32(self) == Some(*other)
    }
}

// --- indexing --------------------------------------------------------------

mod private {
    pub trait Sealed {}
    impl Sealed for usize {}
    impl Sealed for str {}
    impl Sealed for String {}
    impl Sealed for super::Key {}
    impl Sealed for super::K {}
    impl<T: ?Sized + Sealed> Sealed for &T {}
}

/// What can index a [`Value`]: a position in an array, or a key of an object.
pub trait Index: private::Sealed {
    #[doc(hidden)]
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value>;
    #[doc(hidden)]
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value>;
    #[doc(hidden)]
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value;
}

impl Index for usize {
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value> {
        match value {
            Value::Array(list) => list.get(*self),
            _ => None,
        }
    }
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value> {
        match value {
            Value::Array(list) => list.get_mut(*self),
            _ => None,
        }
    }
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value {
        match value {
            Value::Array(list) => {
                let len = list.len();
                list.get_mut(*self)
                    .unwrap_or_else(|| panic!("cannot access index {self} of JSON array of length {len}"))
            }
            _ => panic!("cannot access index {self} of a JSON value that is not an array"),
        }
    }
}

impl Index for str {
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value> {
        match value {
            Value::Object(map) => map.get(self),
            _ => None,
        }
    }
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value> {
        match value {
            Value::Object(map) => map.get_mut(self),
            _ => None,
        }
    }
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value {
        if let Value::Null = value {
            *value = Value::Object(Map::new());
        }
        match value {
            Value::Object(map) => map.entry(self).or_insert(Value::Null),
            _ => panic!("cannot access key {self:?} in a JSON value that is not an object"),
        }
    }
}

impl Index for String {
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value> {
        self.as_str().index_into(value)
    }
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value> {
        self.as_str().index_into_mut(value)
    }
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value {
        self.as_str().index_or_insert(value)
    }
}

impl Index for Key {
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value> {
        self.as_str().index_into(value)
    }
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value> {
        self.as_str().index_into_mut(value)
    }
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value {
        self.as_str().index_or_insert(value)
    }
}

impl Index for K {
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value> {
        match value {
            Value::Object(map) => map.get_known(*self),
            _ => None,
        }
    }
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value> {
        self.as_str().index_into_mut(value)
    }
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value {
        self.as_str().index_or_insert(value)
    }
}

impl<T: ?Sized + Index> Index for &T {
    fn index_into<'v>(&self, value: &'v Value) -> Option<&'v Value> {
        (**self).index_into(value)
    }
    fn index_into_mut<'v>(&self, value: &'v mut Value) -> Option<&'v mut Value> {
        (**self).index_into_mut(value)
    }
    fn index_or_insert<'v>(&self, value: &'v mut Value) -> &'v mut Value {
        (**self).index_or_insert(value)
    }
}

/// `value["key"]` and `value[0]`: `Null` when absent, as serde_json indexes.
impl<I: Index> std::ops::Index<I> for Value {
    type Output = Value;
    fn index(&self, index: I) -> &Value {
        index.index_into(self).unwrap_or(&NULL)
    }
}
/// `value["key"] = ...` inserts into an object (a `Null` becomes one); an array position must exist.
impl<I: Index> std::ops::IndexMut<I> for Value {
    fn index_mut(&mut self, index: I) -> &mut Value {
        index.index_or_insert(self)
    }
}

// --- serde -----------------------------------------------------------------

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Value::Null => serializer.serialize_unit(),
            Value::Bool(b) => serializer.serialize_bool(*b),
            Value::Number(n) => n.serialize(serializer),
            Value::String(text) => serializer.serialize_str(text),
            Value::Array(list) => serializer.collect_seq(list),
            Value::Object(map) => map.serialize(serializer),
        }
    }
}

/// An object's first key: a field name, or the marker of an `arbitrary_precision` number token.
enum KeyClass {
    Number,
    Key(Key),
}
struct KeyClassifier;
impl<'de> DeserializeSeed<'de> for KeyClassifier {
    type Value = KeyClass;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> std::result::Result<KeyClass, D::Error> {
        deserializer.deserialize_str(self)
    }
}
impl Visitor<'_> for KeyClassifier {
    type Value = KeyClass;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a string key")
    }
    fn visit_str<E: serde::de::Error>(self, text: &str) -> std::result::Result<KeyClass, E> {
        Ok(if text == NUMBER_TOKEN && number_marker_reserved() { KeyClass::Number } else { KeyClass::Key(Key::new(text)) })
    }
    fn visit_string<E: serde::de::Error>(self, text: String) -> std::result::Result<KeyClass, E> {
        Ok(if text == NUMBER_TOKEN && number_marker_reserved() { KeyClass::Number } else { KeyClass::Key(Key::from(text)) })
    }
}

struct ValueVisitor;
impl<'de> Visitor<'de> for ValueVisitor {
    type Value = Value;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any valid JSON value")
    }
    fn visit_bool<E>(self, b: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(b))
    }
    fn visit_i64<E>(self, i: i64) -> std::result::Result<Value, E> {
        Ok(Value::Number(i.into()))
    }
    fn visit_u64<E>(self, u: u64) -> std::result::Result<Value, E> {
        Ok(Value::Number(u.into()))
    }
    fn visit_f64<E>(self, f: f64) -> std::result::Result<Value, E> {
        Ok(Value::from(f))
    }
    fn visit_str<E>(self, text: &str) -> std::result::Result<Value, E> {
        Ok(Value::String(text.to_string()))
    }
    fn visit_string<E>(self, text: String) -> std::result::Result<Value, E> {
        Ok(Value::String(text))
    }
    fn visit_none<E>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> std::result::Result<Value, D::Error> {
        Deserialize::deserialize(deserializer)
    }
    fn visit_unit<E>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut access: A) -> std::result::Result<Value, A::Error> {
        let mut list = Vec::with_capacity(access.size_hint().unwrap_or(0).min(4096));
        while let Some(item) = access.next_element()? {
            list.push(item);
        }
        Ok(Value::Array(list))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> std::result::Result<Value, A::Error> {
        match access.next_key_seed(KeyClassifier)? {
            None => Ok(Value::Object(Map::new())),
            Some(KeyClass::Number) => Ok(Value::Number(access.next_value_seed(NumberTokenText)?)),
            Some(KeyClass::Key(first)) => {
                let mut map = Map::with_capacity(access.size_hint().unwrap_or(0).min(4096));
                map.insert(first, access.next_value()?);
                while let Some(key) = access.next_key::<Key>()? {
                    map.insert(key, access.next_value()?);
                }
                Ok(Value::Object(map))
            }
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Value, D::Error> {
        deserializer.deserialize_any(ValueVisitor)
    }
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

/// An object: keys in insertion order, as serde_json's `preserve_order` map keeps them.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Map {
    map: IndexMap<Key, Value, FxBuildHasher>,
}

/// Up to this many keys a lookup scans in order instead of hashing (every node, most paints).
const SCAN_MAX: usize = 24;

pub type Iter<'a> = indexmap::map::Iter<'a, Key, Value>;
pub type IterMut<'a> = indexmap::map::IterMut<'a, Key, Value>;
pub type IntoIter = indexmap::map::IntoIter<Key, Value>;
pub type Keys<'a> = indexmap::map::Keys<'a, Key, Value>;
pub type Values<'a> = indexmap::map::Values<'a, Key, Value>;
pub type ValuesMut<'a> = indexmap::map::ValuesMut<'a, Key, Value>;

impl Map {
    pub fn new() -> Map {
        Map::default()
    }

    pub fn with_capacity(capacity: usize) -> Map {
        Map { map: IndexMap::with_capacity_and_hasher(capacity, FxBuildHasher) }
    }

    pub fn clear(&mut self) {
        self.map.clear()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn position(&self, key: &str) -> Option<usize> {
        if self.map.len() <= SCAN_MAX {
            self.map.keys().position(|k| k.as_str() == key)
        } else {
            self.map.get_index_of(key)
        }
    }

    pub fn get<Q: ?Sized + AsRef<str>>(&self, key: &Q) -> Option<&Value> {
        let at = self.position(key.as_ref())?;
        self.map.get_index(at).map(|(_, value)| value)
    }

    /// [`Map::get`] of a known name: the scan compares integers.
    pub fn get_known(&self, key: K) -> Option<&Value> {
        if self.map.len() <= SCAN_MAX {
            self.map.iter().find(|(k, _)| matches!(&k.0, KeyRepr::Known(known) if *known == key)).map(|(_, value)| value)
        } else {
            self.map.get(key.as_str())
        }
    }

    pub fn get_mut<Q: ?Sized + AsRef<str>>(&mut self, key: &Q) -> Option<&mut Value> {
        let at = self.position(key.as_ref())?;
        self.map.get_index_mut(at).map(|(_, value)| value)
    }

    pub fn get_key_value<Q: ?Sized + AsRef<str>>(&self, key: &Q) -> Option<(&Key, &Value)> {
        let at = self.position(key.as_ref())?;
        self.map.get_index(at)
    }

    pub fn contains_key<Q: ?Sized + AsRef<str>>(&self, key: &Q) -> bool {
        self.position(key.as_ref()).is_some()
    }

    /// Insert, or replace the value in place (the key keeps its position). Returns the value replaced.
    pub fn insert(&mut self, key: Key, value: Value) -> Option<Value> {
        self.map.insert(key, value)
    }

    /// Remove a key by moving the last entry into its place, as serde_json's ordered map removes.
    pub fn remove<Q: ?Sized + AsRef<str>>(&mut self, key: &Q) -> Option<Value> {
        self.map.swap_remove(key.as_ref())
    }

    pub fn swap_remove<Q: ?Sized + AsRef<str>>(&mut self, key: &Q) -> Option<Value> {
        self.map.swap_remove(key.as_ref())
    }

    /// Remove a key, keeping the order of the rest.
    pub fn shift_remove<Q: ?Sized + AsRef<str>>(&mut self, key: &Q) -> Option<Value> {
        self.map.shift_remove(key.as_ref())
    }

    pub fn remove_entry<Q: ?Sized + AsRef<str>>(&mut self, key: &Q) -> Option<(Key, Value)> {
        self.map.swap_remove_entry(key.as_ref())
    }

    pub fn append(&mut self, other: &mut Map) {
        for (key, value) in other.map.drain(..) {
            self.map.insert(key, value);
        }
    }

    pub fn entry<S: Into<Key>>(&mut self, key: S) -> Entry<'_> {
        match self.map.entry(key.into()) {
            indexmap::map::Entry::Occupied(entry) => Entry::Occupied(OccupiedEntry(entry)),
            indexmap::map::Entry::Vacant(entry) => Entry::Vacant(VacantEntry(entry)),
        }
    }

    pub fn retain<F: FnMut(&Key, &mut Value) -> bool>(&mut self, keep: F) {
        self.map.retain(keep)
    }

    pub fn sort_keys(&mut self) {
        self.map.sort_keys()
    }

    pub fn iter(&self) -> Iter<'_> {
        self.map.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_> {
        self.map.iter_mut()
    }

    pub fn keys(&self) -> Keys<'_> {
        self.map.keys()
    }

    pub fn values(&self) -> Values<'_> {
        self.map.values()
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_> {
        self.map.values_mut()
    }

    /// The map as serde_json holds it.
    pub fn to_json(&self) -> serde_json::Map<String, serde_json::Value> {
        self.map.iter().map(|(key, value)| (key.as_str().to_string(), value.to_json())).collect()
    }
}

impl From<serde_json::Map<String, serde_json::Value>> for Map {
    fn from(map: serde_json::Map<String, serde_json::Value>) -> Map {
        map.into_iter().map(|(key, value)| (Key::from(key), Value::from(value))).collect()
    }
}
impl From<Map> for serde_json::Map<String, serde_json::Value> {
    fn from(map: Map) -> serde_json::Map<String, serde_json::Value> {
        map.into_iter().map(|(key, value)| (String::from(key), serde_json::Value::from(value))).collect()
    }
}

impl fmt::Debug for Map {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.map.iter()).finish()
    }
}

impl<Q: ?Sized + AsRef<str>> std::ops::Index<&Q> for Map {
    type Output = Value;
    fn index(&self, key: &Q) -> &Value {
        self.get(key).unwrap_or_else(|| panic!("no entry found for key {:?}", key.as_ref()))
    }
}
impl<Q: ?Sized + AsRef<str>> std::ops::IndexMut<&Q> for Map {
    fn index_mut(&mut self, key: &Q) -> &mut Value {
        self.get_mut(key).unwrap_or_else(|| panic!("no entry found for key {:?}", key.as_ref()))
    }
}

impl IntoIterator for Map {
    type Item = (Key, Value);
    type IntoIter = IntoIter;
    fn into_iter(self) -> IntoIter {
        self.map.into_iter()
    }
}
impl<'a> IntoIterator for &'a Map {
    type Item = (&'a Key, &'a Value);
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.map.iter()
    }
}
impl<'a> IntoIterator for &'a mut Map {
    type Item = (&'a Key, &'a mut Value);
    type IntoIter = IterMut<'a>;
    fn into_iter(self) -> IterMut<'a> {
        self.map.iter_mut()
    }
}

impl FromIterator<(Key, Value)> for Map {
    fn from_iter<I: IntoIterator<Item = (Key, Value)>>(iter: I) -> Map {
        Map { map: iter.into_iter().collect() }
    }
}
impl FromIterator<(String, Value)> for Map {
    fn from_iter<I: IntoIterator<Item = (String, Value)>>(iter: I) -> Map {
        Map { map: iter.into_iter().map(|(key, value)| (Key::from(key), value)).collect() }
    }
}
impl<'a> FromIterator<(&'a str, Value)> for Map {
    fn from_iter<I: IntoIterator<Item = (&'a str, Value)>>(iter: I) -> Map {
        Map { map: iter.into_iter().map(|(key, value)| (Key::new(key), value)).collect() }
    }
}
impl Extend<(Key, Value)> for Map {
    fn extend<I: IntoIterator<Item = (Key, Value)>>(&mut self, iter: I) {
        self.map.extend(iter)
    }
}
impl Extend<(String, Value)> for Map {
    fn extend<I: IntoIterator<Item = (String, Value)>>(&mut self, iter: I) {
        self.map.extend(iter.into_iter().map(|(key, value)| (Key::from(key), value)))
    }
}

impl Serialize for Map {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_map(self.map.iter().map(|(key, value)| (key.as_str(), value)))
    }
}
/// An object, key by key: every key is a field name here, the number-token marker included (only a [`Value`]
/// reads that marker as a number).
impl<'de> Deserialize<'de> for Map {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Map, D::Error> {
        struct MapVisitor;
        impl<'de> Visitor<'de> for MapVisitor {
            type Value = Map;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_unit<E>(self) -> std::result::Result<Map, E> {
                Ok(Map::new())
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> std::result::Result<Map, A::Error> {
                let mut map = Map::with_capacity(access.size_hint().unwrap_or(0).min(4096));
                while let Some(key) = access.next_key::<Key>()? {
                    map.insert(key, access.next_value()?);
                }
                Ok(map)
            }
        }
        deserializer.deserialize_map(MapVisitor)
    }
}

/// A view of one key of a [`Map`], present or not.
pub enum Entry<'a> {
    Vacant(VacantEntry<'a>),
    Occupied(OccupiedEntry<'a>),
}
pub struct VacantEntry<'a>(indexmap::map::VacantEntry<'a, Key, Value>);
pub struct OccupiedEntry<'a>(indexmap::map::OccupiedEntry<'a, Key, Value>);

impl<'a> Entry<'a> {
    pub fn key(&self) -> &Key {
        match self {
            Entry::Vacant(entry) => entry.key(),
            Entry::Occupied(entry) => entry.key(),
        }
    }

    pub fn or_insert(self, default: Value) -> &'a mut Value {
        match self {
            Entry::Vacant(entry) => entry.insert(default),
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }

    pub fn or_insert_with<F: FnOnce() -> Value>(self, default: F) -> &'a mut Value {
        match self {
            Entry::Vacant(entry) => entry.insert(default()),
            Entry::Occupied(entry) => entry.into_mut(),
        }
    }

    pub fn and_modify<F: FnOnce(&mut Value)>(self, change: F) -> Entry<'a> {
        match self {
            Entry::Occupied(mut entry) => {
                change(entry.get_mut());
                Entry::Occupied(entry)
            }
            vacant => vacant,
        }
    }
}

impl<'a> VacantEntry<'a> {
    pub fn key(&self) -> &Key {
        self.0.key()
    }
    pub fn insert(self, value: Value) -> &'a mut Value {
        self.0.insert(value)
    }
}

impl<'a> OccupiedEntry<'a> {
    pub fn key(&self) -> &Key {
        self.0.key()
    }
    pub fn get(&self) -> &Value {
        self.0.get()
    }
    pub fn get_mut(&mut self) -> &mut Value {
        self.0.get_mut()
    }
    pub fn into_mut(self) -> &'a mut Value {
        self.0.into_mut()
    }
    pub fn insert(&mut self, value: Value) -> Value {
        self.0.insert(value)
    }
    /// Remove by moving the last entry into this one's place ([`Map::remove`]).
    pub fn remove(self) -> Value {
        self.0.swap_remove()
    }
    pub fn shift_remove(self) -> Value {
        self.0.shift_remove()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether this build's serde_json keeps number tokens as text (`arbitrary_precision`).
    fn tokens_are_text() -> bool {
        serde_json::Value::from(0.1f32).to_string() == "0.1"
    }

    #[test]
    fn known_keys_are_unique_and_round_trip() {
        let mut seen = std::collections::HashSet::new();
        for &known in K::ALL {
            assert!(seen.insert(known.as_str()), "{} is listed twice", known.as_str());
            assert_eq!(K::lookup(known.as_str()), Some(known));
            assert_eq!(Key::new(known.as_str()).known(), Some(known));
            assert_eq!(Key::from(known.as_str().to_string()).known(), Some(known));
            assert_eq!(Key::from(std::borrow::Cow::Borrowed(known.as_str())).known(), Some(known));
        }
        assert!(Key::new("not a field").known().is_none());
        assert_eq!(Key::new("x"), Key::from(K::X));
        assert_ne!(Key::new("x"), Key::new("y"));
        assert_eq!(Key::new("custom"), Key::new("custom"));
        assert!(Key::new("x") == "x" && Key::new("custom") == "custom");
    }

    #[test]
    fn number_tokens_read_as_they_did() {
        for (token, f) in [("1", 1.0), ("-7", -7.0), ("0.1", 0.1), ("1e2", 100.0), ("1.2300", 1.23),
            ("29.970000000000001", 29.970000000000001), ("0.84551240822557006", 0.84551240822557006),
            ("18446744073709551615", 18446744073709551615.0), ("18446744073709551616", 18446744073709551616.0)]
        {
            let n = Number::from_token(token).unwrap();
            assert_eq!(n.as_f64().unwrap().to_bits(), f64::to_bits(f), "{token}");
            assert_eq!(n.as_f64().unwrap().to_bits(), token.parse::<f64>().unwrap().to_bits(), "{token}");
        }
        assert_eq!(Number::from_token("1").unwrap().as_u64(), Some(1));
        assert_eq!(Number::from_token("-7").unwrap().as_i64(), Some(-7));
        assert_eq!(Number::from_token("1.0").unwrap().as_u64(), None);
        assert_eq!(Number::from_token("1.0").unwrap().as_i64(), None);
        assert!(Number::from_token("-0").unwrap().as_f64().unwrap().is_sign_negative());
        assert_ne!(Number::from_token("1").unwrap(), Number::from_token("1.0").unwrap());
        assert_eq!(Number::from_token("1.0").unwrap(), Number::from_token("1.00").unwrap());
        assert!(Number::from_token("x").is_none());
    }

    #[test]
    fn parsed_documents_equal_converted_documents() {
        let text = r#"{"id":"n1","type":"path","x":1,"y":-2.5,"opacity":0.30000000000000004,"big":18446744073709551615,
            "neg":-9223372036854775808,"tiny":1e-7,"points":[{"type":"move","x":0,"y":0.1}],"flag":true,"none":null,
            "nested":{"a":[1,2.0,"three",{"b":null}]}}"#;
        let parsed: Value = from_str(text).unwrap();
        let converted = Value::from(serde_json::from_str::<serde_json::Value>(text).unwrap());
        assert_eq!(parsed, converted);
        assert_eq!(parsed["x"].as_u64(), Some(1));
        assert_eq!(parsed["y"].as_f64(), Some(-2.5));
        assert_eq!(parsed["opacity"].as_f64(), Some(0.30000000000000004));
        assert_eq!(parsed["big"].as_u64(), Some(u64::MAX));
        assert_eq!(parsed["neg"].as_i64(), Some(i64::MIN));
        assert_eq!(parsed["points"][0]["y"].as_f64(), Some(0.1));
        assert!(parsed["absent"].is_null() && parsed["nested"]["a"][3]["b"].is_null());
        assert_eq!(parsed["nested"]["a"][2], "three");
        assert_eq!(parsed.pointer("/nested/a/1").and_then(Value::as_f64), Some(2.0));
        let keys: Vec<&str> = parsed.as_object().unwrap().keys().map(Key::as_str).collect();
        assert_eq!(keys, ["id", "type", "x", "y", "opacity", "big", "neg", "tiny", "points", "flag", "none", "nested"]);
        // Out through serde_json and back is the same value.
        assert_eq!(Value::from(parsed.to_json()), parsed);
        assert_eq!(from_str::<Value>(&to_string(&parsed).unwrap()).unwrap(), parsed);
        assert_eq!(parsed.to_string(), to_string(&parsed.to_json()).unwrap());
    }

    #[test]
    fn f32_values_read_back_as_serde_json_writes_them() {
        let mut bits = 0x9E37_79B9u32;
        // 2^-12 has two equally close shortest decimals; formatters break the tie differently.
        let tie = f32::from_bits(0x3980_0000);
        let mut samples = vec![0.0f32, -0.0, 1.0, 0.1, 0.3, 1.0e-7, 16777216.0, 1.0e21, f32::MIN_POSITIVE, f32::MAX, f32::MIN,
            f32::from_bits(1), 123456.79, 0.333_333_34, tie, -tie];
        samples.extend((0..32).flat_map(|e| [2f32.powi(-e), -(2f32.powi(-e)), 2f32.powi(e)]));
        for _ in 0..20_000 {
            bits = bits.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            samples.push(f32::from_bits(bits));
        }
        for f in samples.into_iter().filter(|f| f.is_finite()) {
            let read = Number::from_f32_shortest(f).unwrap().as_f64().unwrap();
            assert_eq!((read as f32).to_bits(), f.to_bits(), "{f:?}");
            // The boundary itself: what a document holding this f32 reads back as.
            assert_eq!(read.to_bits(), Value::from(f).as_f64().unwrap().to_bits(), "{f:?}");
            assert_eq!(read.to_bits(), json!(f).as_f64().unwrap().to_bits(), "{f:?}");
            if tokens_are_text() {
                let token = serde_json::Value::from(f).to_string();
                assert_eq!(read.to_bits(), token.parse::<f64>().unwrap().to_bits(), "{f:?} written as {token}");
            } else {
                assert_eq!(read.to_bits(), f64::from(f).to_bits(), "{f:?}");
            }
        }
        assert!(Number::from_f32_shortest(f32::NAN).is_none() && Number::from_f32_shortest(f32::INFINITY).is_none());
    }

    #[test]
    fn out_of_range_and_invalid_number_tokens() {
        for text in ["NaN", "inf", "-inf", "+1", "01", "1.", ".5", "1e", "-", "", "0x10", "1 "] {
            assert!(Number::from_token(text).is_none(), "{text:?}");
        }
        for text in ["0", "-0", "10", "1.5", "-1.5e-3", "1E+2", "0.0"] {
            assert!(Number::from_token(text).is_some(), "{text:?}");
        }
        // Every number says what it is, and what it says it is can be read.
        for text in ["0", "-0", "7", "-7", "1.5", "1e400", "-1e400", "18446744073709551616"] {
            if let Some(n) = Number::from_token(text) {
                assert!(!n.is_f64() || n.as_f64().is_some(), "{text}");
                assert!(!n.is_u64() || n.as_u64().is_some(), "{text}");
                assert!(!n.is_i64() || n.as_i64().is_some(), "{text}");
                // Whatever is constructed converts and writes; nothing becomes null.
                assert!(n.to_json().is_some(), "{text}");
                assert!(Value::Number(n.clone()).to_json().is_number(), "{text}");
                assert!(to_string(&n).is_ok(), "{text}");
            }
        }
        // A token no finite f64 holds exists only where serde_json can hold it too.
        assert_eq!(Number::from_token("1e400").is_some(), tokens_are_text());
        if tokens_are_text() {
            // It reads as no number (a reader's default applies) and keeps its text.
            let big = Number::from_token("1e400").unwrap();
            assert_eq!((big.as_f64(), big.as_u64(), big.as_i64()), (None, None, None));
            assert!(!big.is_f64() && !big.is_u64() && !big.is_i64());
            // It writes back as serde_json writes the token.
            let written = serde_json::from_str::<serde_json::Value>("1e400").unwrap().to_string();
            assert_eq!(big.to_string(), written);
            assert_eq!(to_string(&big).unwrap(), written);
            assert_eq!(big, Number::from_token("1e400").unwrap());
            assert_ne!(big, Number::from_token("-1e400").unwrap());
            // Parsing and converting agree, and the token survives a round trip.
            let parsed: Value = from_str(r#"{"rotation":1e400,"opacity":-1e400}"#).unwrap();
            let converted = Value::from(serde_json::from_str::<serde_json::Value>(r#"{"rotation":1e400,"opacity":-1e400}"#).unwrap());
            assert_eq!(parsed, converted);
            assert_eq!(parsed["rotation"].as_f64().unwrap_or(0.0), 0.0);
            assert_eq!(parsed["opacity"].as_f64().unwrap_or(1.0), 1.0);
            assert!(parsed["rotation"].is_number());
            let through_serde_json = serde_json::to_string(&serde_json::from_str::<serde_json::Value>(r#"{"rotation":1e400,"opacity":-1e400}"#).unwrap()).unwrap();
            assert_eq!(to_string(&parsed).unwrap(), through_serde_json);
            assert_eq!(Value::from(parsed.to_json()), parsed);
        }
    }

    #[test]
    fn a_known_name_is_one_key_however_it_is_built() {
        // Below and above the scan limit: every way of building the key finds the one entry.
        for extra in [0usize, 40] {
            let mut map = Map::new();
            for i in 0..extra {
                map.insert(format!("k{i}").into(), json!(i));
            }
            map.insert(Key::from("x".to_string()), json!(1));
            map.insert(Key::from(K::X), json!(2));
            map.insert(Key::new("x"), json!(3));
            *map.entry("x").or_insert(Value::Null) = json!(4);
            assert_eq!(map.len(), extra + 1);
            assert_eq!(map.get("x"), Some(&json!(4)));
            assert_eq!(map.get_known(K::X), Some(&json!(4)));
            assert_eq!(map.keys().filter(|key| *key == "x").count(), 1);
            assert_eq!(to_string(&map).unwrap().matches("\"x\"").count(), 1);
        }
    }

    #[test]
    fn f32_equality_is_of_finite_values() {
        assert!(Value::from(0.5f64) == 0.5f32 && 0.5f32 == Value::from(0.5f64));
        assert!(Value::from(1e100f64) != f32::INFINITY && Value::from(-1e100f64) != f32::NEG_INFINITY);
        assert!(f32::INFINITY != Value::from(1e100f64));
        assert!(Value::Null != 0.0f32 && json!("1") != 1.0f32);
        // An integer one above an f32 midpoint narrows up from the integer, though its f64 is the midpoint
        // itself (which would tie to even, down). Both directions, both signs, as serde_json compares.
        let above = 9_223_372_586_610_589_697u64; // 2^63 + 2^39 + 1
        assert!(Value::from(above) == above as f32 && above as f32 == Value::from(above));
        assert!(Value::from(above) != 9_223_372_036_854_775_808u64 as f32);
        assert_eq!(Value::from(above) == above as f32, serde_json::Value::from(above) == above as f32);
        let below = -4_611_686_293_305_294_849i64; // -(2^62 + 2^38 + 1)
        assert!(Value::from(below) == below as f32 && below as f32 == Value::from(below));
        assert!(Value::from(below) != -4_611_686_018_427_387_904i64 as f32);
        assert_eq!(Value::from(below) == below as f32, serde_json::Value::from(below) == below as f32);
    }

    #[test]
    fn the_number_marker_is_reserved_only_where_serde_json_reserves_it() {
        // Parsing agrees with converting what serde_json parses, in either build.
        for text in [r#"{"$serde_json::private::Number":"1"}"#, r#"{"metadata":{"$serde_json::private::Number":"1"}}"#] {
            let parsed: Value = from_str(text).unwrap();
            assert_eq!(parsed, Value::from(serde_json::from_str::<serde_json::Value>(text).unwrap()), "{text}");
        }
        assert_eq!(number_marker_reserved(), tokens_are_text());
    }

    #[test]
    fn a_map_keeps_the_number_marker_as_an_ordinary_key() {
        for text in [
            r#"{"$serde_json::private::Number":"1"}"#,
            r#"{"$serde_json::private::Number":"text","b":2}"#,
            r#"{"a":1,"$serde_json::private::Number":"1"}"#,
        ] {
            let map: Map = from_str(text).unwrap();
            assert!(map.get(NUMBER_TOKEN).is_some_and(Value::is_string), "{text}");
            assert_eq!(to_string(&map).unwrap(), text);
        }
    }

    #[test]
    fn maps_keep_insertion_order_and_serde_json_removal() {
        let mut map = Map::new();
        for key in ["a", "b", "c", "d"] {
            assert!(map.insert(key.into(), json!(key)).is_none());
        }
        assert_eq!(map.insert("b".into(), json!(2)), Some(json!("b")));
        assert_eq!(map.keys().map(Key::as_str).collect::<Vec<_>>(), ["a", "b", "c", "d"]);
        assert_eq!(map.remove("a"), Some(json!("a")));
        assert_eq!(map.keys().map(Key::as_str).collect::<Vec<_>>(), ["d", "b", "c"]);
        assert_eq!(map.shift_remove("d"), Some(json!("d")));
        assert_eq!(map.keys().map(Key::as_str).collect::<Vec<_>>(), ["b", "c"]);
        *map.entry("e").or_insert(Value::Null) = json!(5);
        map.entry("e").and_modify(|v| *v = json!(6));
        assert_eq!(map["e"], 6);
        map.retain(|key, _| key != "b");
        assert!(!map.contains_key("b") && map.contains_key(&"c".to_string()) && map.get_known(K::C).is_some());
        // Past the scan limit lookups hash, and answer the same.
        let mut big = Map::new();
        for i in 0..200 {
            big.insert(format!("k{i}").into(), json!(i));
        }
        big.insert("x".into(), json!("known"));
        assert_eq!(big.get("k150"), Some(&json!(150)));
        assert_eq!(big.get_known(K::X), Some(&json!("known")));
        assert_eq!(big.remove("k0"), Some(json!(0)));
        assert_eq!(big.keys().next().map(Key::as_str), Some("x"));
        assert!(big.get("k0").is_none());
        // Equal whatever the order, as serde_json's maps are.
        let a: Value = from_str(r#"{"x":1,"y":2}"#).unwrap();
        let b: Value = from_str(r#"{"y":2,"x":1}"#).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn typed_structs_read_through_from_value() {
        #[derive(serde::Deserialize, serde::Serialize, PartialEq, Debug)]
        struct Stop {
            offset: f32,
            count: u32,
            name: Option<String>,
            extra: Value,
        }
        let value = json!({"offset": 0.1, "count": 3, "name": null, "extra": {"k": [1, 2.5]}});
        let stop: Stop = from_value(value.clone()).unwrap();
        assert_eq!(stop, Stop { offset: 0.1, count: 3, name: None, extra: json!({"k": [1, 2.5]}) });
        assert_eq!(from_value_ref::<Stop>(&value).unwrap(), stop);
        assert_eq!(to_value(&stop).unwrap()["extra"]["k"][1], 2.5);
        assert!(from_value::<Stop>(json!({"offset": 0.1, "count": 3.0, "extra": null})).is_err());
    }

    #[test]
    fn records_keep_reading_around_out_of_range_numbers() {
        if !tokens_are_text() {
            return; // this build's serde_json cannot hold such a token at all
        }
        #[derive(serde::Deserialize, PartialEq, Debug)]
        struct Layer {
            id: String,
            opacity: Option<f64>,
            #[serde(default)]
            effects: Option<Vec<Value>>,
        }
        #[derive(serde::Deserialize, PartialEq, Debug)]
        #[serde(tag = "type", rename_all = "lowercase")]
        enum Paint {
            Solid { color: String, #[serde(default)] params: Value },
        }
        // An unknown field is skipped whatever it holds, from JSON text and from a converted serde_json value.
        let text = r#"{"id":"fill-0","opacity":0.5,"extra":1e400,"more":{"deep":[1e400,{"x":-1e400}]}}"#;
        for value in [from_str::<Value>(text).unwrap(), Value::from(serde_json::from_str::<serde_json::Value>(text).unwrap())] {
            let layer: Layer = from_value_ref(&value).unwrap();
            assert_eq!(layer, Layer { id: "fill-0".into(), opacity: Some(0.5), effects: None });
            assert_eq!(Layer::deserialize(&value).unwrap(), layer);
        }
        // A field that carries any value keeps the token, also through a tagged record's buffered content.
        let value: Value = from_str(r#"{"id":"f","effects":[{"type":"future","amount":1e400}]}"#).unwrap();
        let layer: Layer = from_value_ref(&value).unwrap();
        assert_eq!(layer.effects.as_ref().unwrap()[0], value["effects"][0]);
        let through_serde_json = serde_json::to_string(&serde_json::from_str::<serde_json::Value>(r#"{"type":"future","amount":1e400}"#).unwrap()).unwrap();
        assert_eq!(to_string(&layer.effects.unwrap()[0]).unwrap(), through_serde_json);
        let value: Value = from_str(r##"{"type":"solid","color":"#f00","params":{"big":1e400}}"##).unwrap();
        let Paint::Solid { color, params } = from_value_ref(&value).unwrap();
        assert_eq!((color.as_str(), &params), ("#f00", &value["params"]));
        // A float field reads the token as the float its text parses to, as it did through serde_json's own
        // borrowed reader: the record stays (a fill with such an opacity is clamped downstream, not dropped).
        let text = r#"{"id":"f","opacity":1e400}"#;
        for value in [from_str::<Value>(text).unwrap(), Value::from(serde_json::from_str::<serde_json::Value>(text).unwrap())] {
            let layer: Layer = from_value_ref(&value).unwrap();
            assert_eq!(layer.opacity, Some(f64::INFINITY));
            let old: Layer = Layer::deserialize(&serde_json::from_str::<serde_json::Value>(text).unwrap()).unwrap();
            assert_eq!(layer.opacity, old.opacity, "the reader this one replaces");
        }
        let value: Value = from_str(r#"{"id":"f","opacity":-1e400}"#).unwrap();
        assert_eq!(from_value_ref::<Layer>(&value).unwrap().opacity, Some(f64::NEG_INFINITY));
        // An integer field refuses it.
        #[derive(serde::Deserialize, Debug)]
        struct Count {
            #[allow(dead_code)]
            n: u32,
        }
        assert!(from_value_ref::<Count>(&from_str::<Value>(r#"{"n":1e400}"#).unwrap()).is_err());
        // A whole value round-trips.
        let value: Value = from_str(text).unwrap();
        assert_eq!(from_value_ref::<Value>(&value).unwrap(), value);
    }

    #[test]
    fn index_assignment_builds_objects() {
        let mut value = Value::Null;
        value["a"]["b"] = json!(1);
        value["list"] = json!([1, 2]);
        value["list"][1] = json!("two");
        assert_eq!(value, json!({"a": {"b": 1}, "list": [1, "two"]}));
        assert_eq!(value["list"].take(), json!([1, "two"]));
        assert!(value["list"].is_null());
    }

    /// `json!` and `to_value` build typed values directly. They must give what the route they replace gave: the
    /// value written into a `serde_json::Value` (numbers as serde_json holds them in this build) and converted.
    #[test]
    fn direct_values_equal_the_serde_json_route() {
        #[derive(serde::Serialize)]
        enum Shape {
            Unit,
            Newtype(f32),
            Tuple(u8, i64),
            Struct { x: f64, name: &'static str },
        }
        #[derive(serde::Serialize)]
        struct Record {
            flag: bool,
            small: f32,
            wide: f64,
            signed: i64,
            unsigned: u64,
            text: String,
            none: Option<u8>,
            some: Option<f32>,
            unit: (),
            list: Vec<f32>,
            pair: (u8, f64),
            shapes: Vec<Shape>,
            keyed: std::collections::BTreeMap<i32, f64>,
            json: serde_json::Value,
            typed: Value,
        }
        let floats = [0.0f32, -0.0, 0.1, 1.0, 16.0, 0.000_244_140_62, 1e-7, 3.402_823_5e38, f32::MIN_POSITIVE, f32::NAN, f32::INFINITY];
        let wide = [0.0f64, -0.0, 0.1, 3.0, 1e20, 1e-320, 0.845_512_408_225_570_06, f64::MAX, f64::NAN, f64::NEG_INFINITY];
        for (i, (&small, &wide)) in floats.iter().zip(wide.iter().cycle()).enumerate() {
            let record = Record {
                flag: i % 2 == 0,
                small,
                wide,
                signed: -(i as i64),
                unsigned: u64::MAX - i as u64,
                text: format!("t{i}"),
                none: None,
                some: Some(small),
                unit: (),
                list: floats.to_vec(),
                pair: (i as u8, wide),
                shapes: vec![Shape::Unit, Shape::Newtype(small), Shape::Tuple(7, -7), Shape::Struct { x: wide, name: "n" }],
                keyed: [(-1, wide), (2, 0.5)].into_iter().collect(),
                json: serde_json::from_str(r#"{"a":[1,-2,3.0,1e-7,"s",null,true],"b":{"c":0.1}}"#).unwrap(),
                typed: from_str(r#"{"n":[0,-0,-0.0,5,5.0,1e21],"id":"k"}"#).unwrap(),
            };
            let old = Value::from(serde_json::to_value(&record).unwrap());
            assert_eq!(to_value(&record).unwrap(), old, "record {i}");
            assert_eq!(to_value(&record).unwrap().to_string(), old.to_string(), "record {i}");
            let (x, name) = (wide, "key");
            let old = Value::from(serde_json::json!({
                "id": "n", "x": x, "small": small, (name): [1, 2.5, small, null, true, {"deep": [x]}], "empty": {}, "none": [],
                "record": record, "dup": 1, "dup": 2,
            }));
            let new = json!({
                "id": "n", "x": x, "small": small, (name): [1, 2.5, small, null, true, {"deep": [x]}], "empty": {}, "none": [],
                "record": record, "dup": 1, "dup": 2,
            });
            assert_eq!(new, old, "literal {i}");
            assert_eq!(new.to_string(), old.to_string(), "literal {i}");
            assert_eq!(new.as_object().unwrap().keys().map(Key::as_str).collect::<Vec<_>>(),
                old.as_object().unwrap().keys().map(Key::as_str).collect::<Vec<_>>());
        }
        // Numbers are the same number, in kind and in bits.
        fn bits(value: &Value) -> Vec<(u8, u64)> {
            match value {
                Value::Number(n) if n.is_u64() => vec![(0, n.as_u64().unwrap())],
                Value::Number(n) if n.is_i64() => vec![(1, n.as_i64().unwrap() as u64)],
                Value::Number(n) => vec![(2, n.as_f64().map_or(u64::MAX, f64::to_bits))],
                Value::Array(list) => list.iter().flat_map(bits).collect(),
                Value::Object(map) => map.values().flat_map(bits).collect(),
                _ => vec![(3, 0)],
            }
        }
        for &small in &floats {
            for &x in &wide {
                let old = Value::from(serde_json::json!({"a": small, "b": x, "c": [small, x, 7, -7, 7.0], "d": {"e": -0.0}}));
                let new = json!({"a": small, "b": x, "c": [small, x, 7, -7, 7.0], "d": {"e": -0.0}});
                assert_eq!(bits(&new), bits(&old), "{small} {x}");
                assert_eq!(bits(&to_value(small).unwrap()), bits(&Value::from(serde_json::to_value(small).unwrap())));
                assert_eq!(bits(&to_value(x).unwrap()), bits(&Value::from(serde_json::to_value(x).unwrap())));
            }
        }
        // Keys are whatever converts into a `String`, as serde_json's `json!` takes them.
        struct Named;
        impl From<Named> for String {
            fn from(_: Named) -> String {
                "named".into()
            }
        }
        let boxed: Box<str> = "boxed".into();
        let owned = String::from("owned");
        let new = json!({"id": 1, ('x'): 2, (boxed.clone()): 3, (owned.clone()): 4, (&owned): 5, (Named): 6});
        let old = Value::from(serde_json::json!({"id": 1, ('x'): 2, (boxed.clone()): 3, (owned.clone()): 4, (&owned): 5, (Named): 6}));
        assert_eq!(new, old);
        assert_eq!(new.as_object().unwrap().keys().map(Key::as_str).collect::<Vec<_>>(), ["id", "x", "boxed", "owned", "named"]);
        assert_eq!(new["owned"], 5);
        // Float keys are the float's own text; an integer wider than 64 bits and serde_json's number struct name
        // follow what this build's serde_json does with them.
        let old = |value: &dyn Fn() -> serde_json::Result<serde_json::Value>| value().ok().map(Value::from);
        let keyed = std::collections::BTreeMap::from([(1u8, 0.5f32)]);
        assert_eq!(to_value(&keyed).ok(), old(&|| serde_json::to_value(&keyed)));
        struct FloatKeys;
        impl Serialize for FloatKeys {
            fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(2))?;
                map.serialize_entry(&0.1f32, &1)?;
                map.serialize_entry(&0.1f64, &2)?;
                map.end()
            }
        }
        assert_eq!(to_value(FloatKeys).ok(), old(&|| serde_json::to_value(FloatKeys)));
        assert_eq!(to_value(FloatKeys).unwrap().as_object().unwrap().keys().map(Key::as_str).collect::<Vec<_>>(), ["0.1"]);
        for wide in [u128::from(u64::MAX), u128::from(u64::MAX) + 1, u128::MAX] {
            assert_eq!(to_value(wide).ok(), old(&|| serde_json::to_value(wide)), "{wide}");
        }
        for wide in [i128::from(i64::MIN), i128::from(i64::MIN) - 1, i128::MIN, i128::from(u64::MAX), i128::MAX] {
            assert_eq!(to_value(wide).ok(), old(&|| serde_json::to_value(wide)), "{wide}");
        }
        #[derive(serde::Serialize)]
        #[serde(rename = "$serde_json::private::Number")]
        struct Marked {
            x: u8,
        }
        #[derive(serde::Serialize)]
        #[serde(rename = "$serde_json::private::Number")]
        struct Token {
            #[serde(rename = "$serde_json::private::Number")]
            text: &'static str,
        }
        assert_eq!(to_value(Marked { x: 1 }).ok(), old(&|| serde_json::to_value(Marked { x: 1 })));
        assert_eq!(to_value(Token { text: "1.5" }).ok(), old(&|| serde_json::to_value(Token { text: "1.5" })));
        // A refusal is serde_json's error for it, category included.
        let category = |error: Error| (error.is_syntax(), error.is_data(), error.to_string());
        let bad_key = std::collections::BTreeMap::from([(vec![1u8], 1u8)]);
        assert_eq!(category(to_value(&bad_key).unwrap_err()), category(serde_json::to_value(&bad_key).unwrap_err()));
        if let Err(theirs) = serde_json::to_value(u128::MAX) {
            assert_eq!(category(to_value(u128::MAX).unwrap_err()), category(theirs));
        }
        if let Err(theirs) = serde_json::to_value(Marked { x: 1 }) {
            assert_eq!(category(to_value(Marked { x: 1 }).unwrap_err()), category(theirs));
        }
        // serde_json's raw-text struct is read as it reads it in this build.
        #[derive(serde::Serialize)]
        #[serde(rename = "$serde_json::private::RawValue")]
        struct Raw {
            #[serde(rename = "$serde_json::private::RawValue")]
            text: &'static str,
        }
        for text in ["1.5", "[1,2]", r#"{"a":[true,null,"s"]}"#] {
            assert_eq!(to_value(Raw { text }).ok(), old(&|| serde_json::to_value(Raw { text })), "{text}");
            let new = json!({"points": Raw { text }, "list": [Raw { text }]});
            assert_eq!(new, Value::from(serde_json::json!({"points": Raw { text }, "list": [Raw { text }]})), "{text}");
        }
        // Raw text that itself holds serde_json's reserved keys, and number tokens that are not numbers.
        let located = |error: Error| (error.classify() as u8, error.line(), error.column(), error.to_string());
        for text in [r#"{"$serde_json::private::RawValue":"[1,2]"}"#, r#"[{"$serde_json::private::RawValue":"nope"}]"#, r#"{"$serde_json::private::Number":"7"}"#] {
            match (to_value(Raw { text }), serde_json::to_value(Raw { text })) {
                (Ok(ours), Ok(theirs)) => assert_eq!(ours, Value::from(theirs), "{text}"),
                (Err(ours), Err(theirs)) => assert_eq!(located(ours), located(theirs), "{text}"),
                (ours, theirs) => panic!("{text}: {ours:?} against {theirs:?}"),
            }
        }
        for text in ["", "1.", "abc", "01", "1e400", "-"] {
            match (to_value(Token { text }), serde_json::to_value(Token { text })) {
                (Ok(ours), Ok(theirs)) => assert_eq!(ours, Value::from(theirs), "{text:?}"),
                (Err(ours), Err(theirs)) => assert_eq!(located(ours), located(theirs), "{text:?}"),
                (ours, theirs) => panic!("{text:?}: {ours:?} against {theirs:?}"),
            }
        }
        assert_eq!(json!(null), Value::Null);
        assert_eq!(json!([]), Value::Array(Vec::new()));
        assert_eq!(json!([1, [2, 3], {"a": false},]), Value::from(serde_json::json!([1, [2, 3], {"a": false}])));
        assert!(to_value(std::collections::BTreeMap::from([(vec![1u8], 1u8)])).is_err());
    }
}
