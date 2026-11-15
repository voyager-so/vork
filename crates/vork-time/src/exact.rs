//! Arithmetic for consumed time-valued channels and discrete timing decisions.
//! Ordinary numeric animation does not construct these values.
use anyhow::{Result, ensure};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

pub type Rational = num_rational::BigRational;

/// Decode authored decimal seconds without passing through binary floating point.
pub fn decimal(text: &str) -> Result<Rational> {
    if let Some((n, d)) = text.split_once('/') {
        let n: BigInt = n.parse()?;
        let d: BigInt = d.parse()?;
        ensure!(!d.is_zero(), "time denominator must be nonzero");
        return Ok(Rational::new(n, d));
    }
    let (mantissa, exponent) = text.split_once(['e', 'E']).unwrap_or((text, "0"));
    let exponent: BigInt = exponent.parse()?;
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    ensure!(!whole.is_empty() && fraction.bytes().all(|b| b.is_ascii_digit()), "invalid decimal time");
    let digits: BigInt = format!("{whole}{fraction}").parse()?;
    if digits.is_zero() { return Ok(Rational::zero()); }
    let scale = BigInt::from(fraction.len()) - exponent;
    let mut magnitude = scale.abs();
    let mut power = BigInt::one();
    let mut factor = BigInt::from(10);
    while !magnitude.is_zero() {
        if magnitude.is_odd() { power *= &factor; }
        magnitude >>= 1;
        if !magnitude.is_zero() { factor = &factor * &factor; }
    }
    Ok(if !scale.is_negative() { Rational::new(digits, power) } else { Rational::from_integer(digits * power) })
}

/// Numeric controls and decoded buffers mean their stored IEEE value.
pub fn ieee(value: f32) -> Result<Rational> {
    ensure!(value.is_finite(), "numeric animation must be finite");
    Ok(Rational::from_float(value).expect("finite IEEE value"))
}

/// Direct local polynomial arithmetic in positive epsilon. A point has only a
/// constant coefficient; an incoming left limit starts as t - epsilon. Keeping
/// coefficients through composition handles reversal, extrema and cancellation.
/// There is no expression tree and no chosen numerical epsilon.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Germ {
    coefficients: Vec<Rational>,
}

impl std::ops::Add for Germ {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { Germ::add(&self, &rhs) }
}
impl std::ops::Sub for Germ {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { Germ::sub(&self, &rhs) }
}
impl std::ops::Mul for Germ {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { Germ::mul(&self, &rhs) }
}

impl Germ {
    pub fn point(value: Rational) -> Self { Self { coefficients: vec![value] } }
    pub fn left(value: Rational) -> Self {
        Self { coefficients: vec![value, -Rational::one()] }
    }
    pub fn right(value: Rational) -> Self {
        Self { coefficients: vec![value, Rational::one()] }
    }
    pub fn value(&self) -> &Rational { &self.coefficients[0] }
    /// Local rate with respect to the incoming positive epsilon (sample time).
    pub fn rate(&self) -> Rational {
        self.coefficients
            .get(1)
            .cloned()
            .unwrap_or_else(Rational::zero)
    }
    /// Continuous time for media sampling; discrete decisions retain the exact germ.
    pub fn seconds(&self) -> Result<f64> {
        self.value()
            .to_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| anyhow::anyhow!("time exceeds the media sample range"))
    }
    /// Unambiguous identity for retained consumers, including every approached-side coefficient.
    pub fn append_key(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.coefficients.len() as u64).to_le_bytes());
        for coefficient in &self.coefficients {
            for integer in [coefficient.numer(), coefficient.denom()] {
                let bytes = integer.to_signed_bytes_le();
                out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
                out.extend_from_slice(&bytes);
            }
        }
    }

    fn normalized(mut coefficients: Vec<Rational>) -> Self {
        while coefficients.len() > 1 && coefficients.last().unwrap().is_zero() { coefficients.pop(); }
        Self { coefficients }
    }
    pub fn add(&self, rhs: &Self) -> Self {
        let mut result = self.coefficients.clone();
        result.resize(result.len().max(rhs.coefficients.len()), Rational::zero());
        for (out, value) in result.iter_mut().zip(&rhs.coefficients) { *out += value; }
        Self::normalized(result)
    }
    pub fn sub(&self, rhs: &Self) -> Self { self.add(&rhs.scale(&-Rational::one())) }
    pub fn scale(&self, rhs: &Rational) -> Self {
        Self::normalized(self.coefficients.iter().map(|value| value * rhs).collect())
    }
    pub fn mul(&self, rhs: &Self) -> Self {
        let mut result = vec![Rational::zero(); self.coefficients.len() + rhs.coefficients.len() - 1];
        for (i, a) in self.coefficients.iter().enumerate() {
            for (j, b) in rhs.coefficients.iter().enumerate() { result[i+j] += a * b; }
        }
        Self::normalized(result)
    }
    /// Compare the approached values; constant equality is inclusive.
    pub fn compare(&self, rhs: &Self) -> Ordering {
        for i in 0..self.coefficients.len().max(rhs.coefficients.len()) {
            let zero = Rational::zero();
            let order = self.coefficients.get(i).unwrap_or(&zero).cmp(rhs.coefficients.get(i).unwrap_or(&zero));
            if order != Ordering::Equal { return order; }
        }
        Ordering::Equal
    }
    pub fn compare_point(&self, rhs: &Rational) -> Ordering { self.compare(&Self::point(rhs.clone())) }
    pub fn floor(&self) -> BigInt {
        let value = self.value();
        let mut floor = value.numer().div_floor(value.denom());
        if value.is_integer() && self.compare_point(value) == Ordering::Less { floor -= 1; }
        floor
    }
    pub fn floor_clamped(&self, lower: i64, upper: i64) -> Result<i64> {
        ensure!(lower <= upper, "invalid time index bounds");
        Ok(self.floor().clamp(lower.into(), upper.into()).to_i64().unwrap())
    }
    /// `∫_0^ε` of this germ as a polynomial in ε: the constant term vanishes and each
    /// coefficient moves one power up, divided by its new power. The side of an
    /// integrated rate survives an isolated zero of the rate this way.
    pub fn integrate_epsilon(&self) -> Self {
        let mut out = vec![Rational::zero()];
        for (k, c) in self.coefficients.iter().enumerate() { out.push(c / Rational::from_integer((k as i64 + 1).into())); }
        Self::normalized(out)
    }
    pub fn euclidean_loop(&self, start: &Rational, end: &Rational) -> Result<Self> {
        let period = end - start;
        ensure!(period.is_positive(), "loop period must be positive");
        let offset = self.sub(&Self::point(start.clone()));
        let cycle = offset.scale(&period.recip()).floor();
        Ok(self.sub(&Self::point(period * Rational::from_integer(cycle))))
    }
    pub fn numeric(&self) -> Result<f32> {
        // ToPrimitive may pass through f64. Correct against exact IEEE cell
        // midpoints so an intermediate rounding cannot move a decision's input.
        let number=self.value();
        let mut value=number.to_f32().unwrap_or(if number.is_negative(){f32::NEG_INFINITY}else{f32::INFINITY});
        if !value.is_finite() {value=if number.is_negative(){-f32::MAX}else{f32::MAX};}
        loop {
            let lower=value.next_down();
            let upper=value.next_up();
            let midpoint=|a:f32,b:f32| -> Result<Rational> {Ok((ieee(a)?+ieee(b)?)/Rational::from_integer(2.into()))};
            if lower.is_finite() {
                let boundary=midpoint(lower,value)?;
                if number<&boundary || (number==&boundary && value.to_bits()&1!=0) {value=lower;continue}
            } else {
                let boundary=ieee(value)?-(ieee(upper)?-ieee(value)?)/Rational::from_integer(2.into());
                ensure!(number>&boundary,"time exceeds finite numeric evaluation range");
            }
            if upper.is_finite() {
                let boundary=midpoint(value,upper)?;
                if number>&boundary || (number==&boundary && value.to_bits()&1!=0) {value=upper;continue}
            } else {
                let boundary=ieee(value)?+(ieee(value)?-ieee(lower)?)/Rational::from_integer(2.into());
                ensure!(number<&boundary,"time exceeds finite numeric evaluation range");
            }
            return Ok(value)
        }
    }
}

/// Exact values at equally spaced instants: a polynomial in the integer `k` with rational
/// coefficients, a line `value + k * step` in the common case. Each value is read exactly as
/// `Germ::seconds` and `Germ::numeric` read it: in machine integers while its numerator and
/// denominator are exact in an f64, and through the same rational otherwise.
#[derive(Clone, Debug)]
pub struct Progression {
    /// Coefficients of the powers of `k`, lowest first.
    coefficients: Vec<Rational>,
    /// Their numerators over one common denominator.
    fixed: Option<(Vec<i128>, i128)>,
    /// A line's value numerator, step numerator and denominator, held inline: the common
    /// case is read at every output sample with one multiplication and one addition.
    line: Option<(i128, i128, i128)>,
}
impl Progression {
    /// Integers up to here are exact in an f64, so their IEEE quotient is correctly rounded.
    const EXACT: i128 = 1 << f64::MANTISSA_DIGITS;
    pub fn new(value: Rational, step: Rational) -> Self {
        Self::polynomial(vec![value, step])
    }
    /// The polynomial of degree below `values.len()` that takes `values[k]` at `k = 0, 1, ...`:
    /// Newton's forward differences, collected into powers of `k`.
    pub fn through(values: &[Rational]) -> Self {
        let mut differences = values.to_vec();
        let mut coefficients = vec![Rational::zero(); values.len()];
        // The binomial `k choose i` in powers of `k`.
        let mut binomial = vec![Rational::one()];
        for i in 0..values.len() {
            // Once the differences vanish the polynomial is complete.
            if differences.iter().all(Zero::is_zero) { break; }
            for (power, part) in binomial.iter().enumerate() {
                coefficients[power] += part * &differences[0];
            }
            for j in 0..differences.len() - 1 {
                differences[j] = &differences[j + 1] - &differences[j];
            }
            differences.pop();
            // `k choose i+1` is `k choose i` times `(k - i) / (i + 1)`.
            let at = Rational::from_integer(BigInt::from(i));
            let next = Rational::from_integer(BigInt::from(i + 1));
            let mut grown = vec![Rational::zero(); binomial.len() + 1];
            for (power, part) in binomial.iter().enumerate() {
                grown[power + 1] += part / &next;
                grown[power] -= part * &at / &next;
            }
            binomial = grown;
        }
        Self::polynomial(coefficients)
    }
    fn polynomial(mut coefficients: Vec<Rational>) -> Self {
        while coefficients.last().is_some_and(Zero::is_zero) { coefficients.pop(); }
        let denom = coefficients.iter().fold(BigInt::one(), |denom, c| denom.lcm(c.denom()));
        let fixed = (|| {
            let numers = coefficients
                .iter()
                .map(|c| (c.numer() * (&denom / c.denom())).to_i128())
                .collect::<Option<Vec<_>>>()?;
            Some((numers, denom.to_i128()?))
        })();
        let line = match &fixed {
            Some((numers, denom)) if numers.len() <= 2 => Some((numers.first().copied().unwrap_or(0), numers.get(1).copied().unwrap_or(0), *denom)),
            _ => None,
        };
        Self { coefficients, fixed, line }
    }
    pub fn exact(&self, k: i64) -> Rational {
        let k = Rational::from_integer(k.into());
        self.coefficients.iter().rev().fold(Rational::zero(), |value, c| value * &k + c)
    }
    /// The value at `k` as one fraction of machine integers, when it fits.
    fn fraction(&self, k: i64) -> Option<(i128, i128)> {
        if let Some((value, step, denom)) = self.line {
            return Some((step.checked_mul(k.into())?.checked_add(value)?, denom));
        }
        let (numers, denom) = self.fixed.as_ref()?;
        // Horner from the leading coefficient.
        let mut numers = numers.iter().rev();
        let mut numer = *numers.next()?;
        for c in numers {
            numer = numer.checked_mul(k.into())?.checked_add(*c)?;
        }
        Some((numer, *denom))
    }
    /// The same rational `exact(k)` is, for the readers that take one.
    fn rational(&self, k: i64) -> Rational {
        match self.fraction(k) {
            Some((numer, denom)) => Rational::new(numer.into(), denom.into()),
            None => self.exact(k),
        }
    }
    /// Whether a progression taking `value` at some integer `k` can be read in machine
    /// integers at all: the coefficients' common denominator is a multiple of the denominator
    /// of every such value, so a value past the long division's limit rules it out.
    pub fn admits(value: &Rational) -> bool { value.denom().bits() <= 126 }
    /// Whether `seconds(k)` and `numeric(k)` are read in machine integers, never through big
    /// rationals, for every `k` in `0..=last`. Three facts prove it for the whole interval:
    /// the coefficients share a machine denominator; it is below the long division's limit
    /// of 2^126; and the sum of every coefficient's magnitude times its power of `last` fits
    /// a machine integer, which bounds every intermediate of the evaluation at every such
    /// `k`. A nonzero value then lies between 2^-126 and 2^127 in magnitude, inside the
    /// normal range of both float widths. The answer can only turn false as `last` grows.
    pub fn fits_through(&self, last: i64) -> bool {
        let Some((numers, denom)) = &self.fixed else { return false };
        if last < 0 || denom >> 126 != 0 { return false; }
        let mut power = 1_i128;
        let mut sum = Some(0_i128);
        for (i, c) in numers.iter().enumerate() {
            if i > 0 {
                let Some(next) = power.checked_mul(last.max(1).into()) else { return false };
                power = next;
            }
            sum = c.checked_abs().and_then(|c| c.checked_mul(power)).and_then(|term| sum?.checked_add(term));
        }
        sum.is_some()
    }
    /// Whether the value at `k` is exactly zero.
    pub fn is_zero(&self, k: i64) -> bool {
        match self.fraction(k) {
            Some((numer, _)) => numer == 0,
            None => self.exact(k).is_zero(),
        }
    }
    /// The value at `k` as its correctly rounded f64, in machine integers.
    fn quotient(&self, k: i64) -> Option<f64> {
        let (numer, denom) = self.fraction(k)?;
        if numer.unsigned_abs() <= Self::EXACT as u128 && denom <= Self::EXACT { return Some(numer as f64 / denom as f64); }
        if numer == 0 { return Some(0.); }
        let (mantissa, power) = Self::divide(numer, denom, f64::MANTISSA_DIGITS)?;
        if !(f64::MIN_EXP - 1..f64::MAX_EXP).contains(&power) { return None; }
        let value = f64::from_bits(((power + 1023) as u64) << 52 | (mantissa & ((1 << 52) - 1)));
        Some(if numer < 0 { -value } else { value })
    }
    /// The value at `k` as its correctly rounded f32, in machine integers.
    fn single(&self, k: i64) -> Option<f32> {
        let (numer, denom) = self.fraction(k)?;
        if numer == 0 { return Some(0.); }
        // The common case: both integers are exact in an f64, so their IEEE quotient is the
        // fraction's correctly rounded f64. Rounding that to f32 is the fraction's own rounding
        // unless the quotient sits exactly on an f32 midpoint (midpoints are f64 values, so
        // the quotient and the fraction lie on one side of every other one); within the
        // normal f32 range a midpoint has only the first of the 29 dropped bits set. A
        // midpoint, like every larger fraction, is decided by the long division below.
        if numer.unsigned_abs() <= Self::EXACT as u128 && denom <= Self::EXACT {
            let wide = numer as f64 / denom as f64;
            let value = wide as f32;
            if value.is_finite() && wide.abs() >= f64::from(f32::MIN_POSITIVE) && wide.to_bits() & 0x1fff_ffff != 0x1000_0000 {
                return Some(value);
            }
        }
        let (mantissa, power) = Self::divide(numer, denom, f32::MANTISSA_DIGITS)?;
        if !(f32::MIN_EXP - 1..f32::MAX_EXP).contains(&power) { return None; }
        let value = f32::from_bits(((power + 127) as u32) << 23 | (mantissa as u32 & ((1 << 23) - 1)));
        Some(if numer < 0 { -value } else { value })
    }
    /// `|numer / denom|` (`numer != 0`, `denom > 0`) rounded to `digits` significant bits, ties
    /// to even, by long division: the quotient's leading bits, a rounding bit and whether
    /// anything remains. Returns the mantissa, in `[2^(digits-1), 2^digits)`, and the power of
    /// two of its leading bit.
    fn divide(numer: i128, denom: i128, digits: u32) -> Option<(u64, i32)> {
        let (a, b) = (numer.unsigned_abs(), denom.unsigned_abs());
        // A remainder is doubled below and must stay in range.
        if a == 0 || denom <= 0 || b >> 126 != 0 { return None; }
        // The value is `(q + r / b) * 2^exponent` throughout.
        let (mut q, mut r, mut exponent) = (a / b, a % b, 0_i32);
        while q == 0 {
            r <<= 1;
            exponent -= 1;
            if r >= b { r -= b; q = 1; }
        }
        let bits = 128 - q.leading_zeros();
        let mut rest = false;
        if bits > digits + 1 {
            let dropped = bits - digits - 1;
            rest = q & ((1_u128 << dropped) - 1) != 0;
            q >>= dropped;
            exponent += dropped as i32;
        } else {
            for _ in bits..digits + 1 {
                q <<= 1;
                r <<= 1;
                exponent -= 1;
                if r >= b { r -= b; q |= 1; }
            }
        }
        rest |= r != 0;
        // `digits + 1` bits: the mantissa and the rounding bit.
        let mut mantissa = (q >> 1) as u64;
        exponent += 1;
        if q & 1 == 1 && (rest || mantissa & 1 == 1) {
            mantissa += 1;
            if mantissa == 1 << digits { mantissa >>= 1; exponent += 1; }
        }
        Some((mantissa, exponent + digits as i32 - 1))
    }
    /// `Germ::point(self.exact(k)).seconds()`: the rational's correctly rounded f64.
    pub fn seconds(&self, k: i64) -> Result<f64> {
        match self.quotient(k) {
            Some(seconds) => Ok(seconds),
            None => Germ::point(self.rational(k)).seconds(),
        }
    }
    /// `Germ::point(self.exact(k)).numeric()`: the rational's correctly rounded f32.
    pub fn numeric(&self, k: i64) -> Result<f32> {
        match self.single(k) {
            Some(value) => Ok(value),
            None => Germ::point(self.rational(k)).numeric(),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn retained_instant_keys_preserve_subframes_and_approached_sides() -> Result<()> {
        let key=|instant:&Germ| {let mut key=Vec::new();instant.append_key(&mut key);key};
        let zero=Germ::point(decimal("0")?);
        let subframe=Germ::point(decimal("1/300")?);
        assert_ne!(key(&zero),key(&subframe));
        let value=decimal("1")?;
        let point=Germ::point(value.clone());
        let left=Germ::left(value.clone());
        let right=Germ::right(value);
        assert_ne!(key(&point),key(&left));
        assert_ne!(key(&left),key(&right));
        assert_ne!(key(&point),key(&right));
        assert_eq!(key(&point),key(&Germ::point(decimal("2/2")?)));
        // Nonlinear clock composition can carry a higher-order approach even
        // when both its value and its first derivative agree with a point.
        let epsilon=right.sub(&point);
        assert_ne!(key(&point),key(&point.add(&epsilon.mul(&epsilon))));
        Ok(())
    }
    use super::*;
    fn q(text: &str) -> Rational { decimal(text).unwrap() }
    #[test]
    fn numeric_lowering_resolves_exact_midpoints_without_double_rounding() -> Result<()> {
        let tiny=q("1/10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000");
        for low in [0.,f32::MIN_POSITIVE,0.1,1.,-1.,1e20] {
            let high=low.next_up();
            let midpoint=(ieee(low)?+ieee(high)?)/q("2");
            assert_eq!(Germ::point(&midpoint-&tiny).numeric()?,low);
            assert_eq!(Germ::point(&midpoint+&tiny).numeric()?,high);
            assert_eq!(Germ::point(midpoint.clone()).numeric()?,if low.to_bits()&1==0 {low}else{high});
            assert_eq!(Germ::left(midpoint.clone()).numeric_side()?,low);
            assert_eq!(Germ::right(midpoint).numeric_side()?,high);
        }
        Ok(())
    }
    #[test]
    fn progressions_read_each_value_as_its_germ_does() -> Result<()> {
        let tie = (ieee(1.)? + ieee(1_f32.next_up())?) / q("2");
        let cases = [
            (q("0"), q("1/48000")),
            (q("-7/3"), q("147/160")),
            (q("1/3"), q("-1/44100")),
            (q("0.1"), ieee(0.1)? / q("48000")),
            (tie.clone(), q("0")),
            (&tie - q("1/48000"), q("1/48000")),
            (q("1e-50"), q("1e-50")),
            (q("1e30"), q("1/3")),
            (q("123456789012345678901234567890/7"), q("1/48000")),
        ];
        for (value, step) in cases {
            let line = Progression::new(value, step);
            for k in [0, 1, 2, 47_999, 48_000, 143_999, 10_000_000_019] {
                let germ = Germ::point(line.exact(k));
                assert_eq!(line.seconds(k)?.to_bits(), germ.seconds()?.to_bits(), "{line:?} at {k}");
                assert_eq!(line.numeric(k)?.to_bits(), germ.numeric()?.to_bits(), "{line:?} at {k}");
            }
        }
        Ok(())
    }
    #[test]
    fn a_progression_through_values_is_their_polynomial() -> Result<()> {
        // f(k) = 1/3 - 5k/7 + k^2/48000^2 - k^3/11 + 2k^4, from its first five values.
        let f = |k: i64| {
            let k = Rational::from_integer(k.into());
            q("1/3") - q("5/7") * &k + q("1/2304000000") * &k * &k - q("1/11") * &k * &k * &k
                + q("2") * &k * &k * &k * &k
        };
        let values: Vec<_> = (0..5).map(f).collect();
        let curve = Progression::through(&values);
        for k in [0, 1, 4, 5, 97, 48_000, 3_000_000_000] {
            assert_eq!(curve.exact(k), f(k), "at {k}");
            let germ = Germ::point(f(k));
            assert_eq!(curve.seconds(k)?.to_bits(), germ.seconds()?.to_bits(), "at {k}");
        }
        // More values than the degree needs change nothing; one value is a constant.
        let line: Vec<_> = (0..4).map(|k| q("1/3") + q("1/48000") * Rational::from_integer(k.into())).collect();
        assert_eq!(Progression::through(&line).exact(99), Progression::new(q("1/3"), q("1/48000")).exact(99));
        assert_eq!(Progression::through(&[q("7/5")]).exact(12), q("7/5"));
        // Only the powers the values need are kept.
        assert_eq!(Progression::through(&line).coefficients.len(), 2);
        assert_eq!(Progression::through(&vec![q("0"); 9]).exact(5), q("0"));
        assert!(curve.fits_through(48_000) && !curve.fits_through(3_000_000_000));
        // A denominator past the division's limit is refused even where an end value is zero.
        let steep = Progression::new(q("-1") / Rational::from_integer(BigInt::one() << 111), q("1/48000") / Rational::from_integer(BigInt::one() << 111));
        assert!(steep.is_zero(48_000) && !steep.fits_through(48_000) && !steep.fits_through(0));
        assert!(Progression::admits(&steep.exact(48_000)) && !Progression::admits(&steep.exact(1)) && Progression::admits(&q("1/3")));
        // Long division reads what the rational reads, for every size of fraction.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || { state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); state };
        for round in 0..20_000 {
            let wide = |bits: u32, a: u64, b: u64| ((u128::from(a) << 64 | u128::from(b)) >> (128 - bits)) as i128;
            let (numer_bits, denom_bits) = (1 + (next() % 126) as u32, 1 + (next() % 125) as u32);
            let mut numer = wide(numer_bits, next(), next());
            let denom = wide(denom_bits, next(), next()).max(1);
            if round % 2 == 1 { numer = -numer; }
            // Every tenth fraction sits exactly on or beside a rounding tie.
            if round % 10 == 0 { numer = ((1_i128 << 53) + 1 + (round as i128 % 3)) << (round % 60); }
            let denom = if round % 10 == 0 { 1_i128 << (1 + round % 70) } else { denom };
            let germ = Germ::point(Rational::new(numer.into(), denom.into()));
            let double = Progression::polynomial(vec![Rational::new(numer.into(), denom.into())]);
            // Whatever the size of the fraction, both readers agree with the rational's.
            assert_eq!(double.seconds(0)?.to_bits(), germ.seconds()?.to_bits(), "{numer}/{denom}");
            if numer_bits <= 100 && denom_bits <= 100 || round % 10 == 0 {
                assert_eq!(double.numeric(0)?.to_bits(), germ.numeric()?.to_bits(), "{numer}/{denom}");
            }
            if denom < 1 << 125 {
                assert!(double.fits_through(0), "{numer}/{denom}");
            }
        }
        // The most negative machine integer has no absolute value of its own type; both
        // readers still read it.
        let least = Progression::new(Rational::from_integer(BigInt::from(i128::MIN)), q("0"));
        assert_eq!(least.numeric(0)?, -(2_f32.powi(127)));
        assert_eq!(least.seconds(0)?, -(2_f64.powi(127)));
        // Small fractions take the f64 quotient: on, beside and away from f32 ties they read
        // what the rational reads.
        for round in 0..20_000_i64 {
            let denom = 1 + (next() % (1 << (1 + round % 53))) as i128;
            let mut numer = (next() % (1 << (1 + (round / 53) % 53))) as i128;
            // Every seventh sits on or one step beside a tie between neighbouring f32 values.
            if round % 7 == 0 { numer = (((1 << 24) + 1 + 2 * (next() % 1000) as i128) << (round % 20)) + (round % 3 - 1) as i128; }
            let denom = if round % 7 == 0 { 1 << (round % 40) } else { denom };
            if round % 2 == 1 { numer = -numer; }
            let small = Progression::polynomial(vec![Rational::new(numer.into(), denom.into())]);
            assert_eq!(small.numeric(0)?.to_bits(), Germ::point(small.exact(0)).numeric()?.to_bits(), "{numer}/{denom}");
        }
        for (numer, denom) in [(1_i128, 1_i128 << 52), (3, 1 << 53), (1 << 53, 3), ((1 << 53) - 1, 1 << 53), ((1 << 24) + 1, 1 << 24), ((1 << 25) + 3, 1 << 25)] {
            let small = Progression::polynomial(vec![Rational::new(numer.into(), denom.into())]);
            assert_eq!(small.numeric(0)?.to_bits(), Germ::point(small.exact(0)).numeric()?.to_bits(), "{numer}/{denom}");
        }
        // Every value of this line is an f32 tie; it is still read in machine integers.
        let ties = Progression::new(q("16777217/16777216"), q("1/8388608"));
        assert!(ties.fits_through(47_999));
        for k in [0, 1, 2, 777, 47_999] {
            assert_eq!(ties.numeric(k)?.to_bits(), Germ::point(ties.exact(k)).numeric()?.to_bits(), "at {k}");
        }
        assert_eq!(Progression::divide(1, 3, 53), Some((0x15555555555555, -2)));
        assert_eq!(Progression::divide(-(1 << 60) - 1, 1 << 7, 53), Some((1 << 52, 53)));
        Ok(())
    }
    #[test]
    fn authored_decimals_and_ieee_buffers_have_distinct_meanings() {
        assert_ne!(q("0.1"), ieee(0.1).unwrap());
        assert_eq!(q("29.97"), q("2997/100"));
        assert_eq!(q("-1.25e-2"), q("-1/80"));
        assert!(ieee(1e-20).unwrap().denom() > &BigInt::from(i64::MAX));
    }
    #[test]
    fn reverse_loop_and_history_preserve_the_approached_branch() {
        let left = Germ::left(q("1/2"));
        let reverse = |t: &Germ| Germ::point(q("1")).sub(t);
        let right = reverse(&left);
        assert_eq!(right, Germ::right(q("1/2")));
        assert_eq!(reverse(&right), left);
        assert_eq!(right.sub(&Germ::point(q("1/24"))), Germ::right(q("11/24")));
        assert_eq!(Germ::left(q("1")).euclidean_loop(&q("0"), &q("1")).unwrap(), Germ::left(q("1")));
        assert_eq!(Germ::point(q("1")).euclidean_loop(&q("0"), &q("1")).unwrap(), Germ::point(q("0")));
        assert_eq!(Germ::right(q("1")).euclidean_loop(&q("0"), &q("1")).unwrap(), Germ::right(q("0")));
    }
    #[test]
    fn extrema_and_blend_cancellation_keep_higher_terms() {
        let epsilon = Germ::left(q("0"));
        let square = epsilon.mul(&epsilon);
        assert_eq!(square.compare_point(&q("0")), Ordering::Greater);
        assert_eq!(square.scale(&q("-1")).compare_point(&q("0")), Ordering::Less);
        let a = Germ::left(q("1/2"));
        let b = a.sub(&square);
        assert_eq!(a.compare_point(&q("1/2")), b.compare_point(&q("1/2")));
        assert_eq!(a.sub(&b).compare_point(&q("0")), Ordering::Greater);
        assert_eq!(a.sub(&a).compare_point(&q("0")), Ordering::Equal);
    }
    #[test]
    fn integrating_a_rate_germ_keeps_the_side_through_an_isolated_zero() {
        // rate (p - 6)^2 approached from below 6: ε² integrates to ε³/3, still a side.
        let below = Germ::left(q("6"));
        let rate = below.sub(&Germ::point(q("6")));
        let rate = rate.mul(&rate);
        assert_eq!(rate.compare_point(&q("0")), Ordering::Greater);
        let integral = rate.integrate_epsilon();
        assert_eq!(integral.value(), &q("0"));
        assert_eq!(integral.compare_point(&q("0")), Ordering::Greater);
        assert_eq!(Germ::point(q("5")).integrate_epsilon(), Germ::point(q("0")).add(&Germ::right(q("0")).scale(&q("5"))));
        assert_eq!(Germ::point(q("0")).integrate_epsilon(), Germ::point(q("0")));
    }
    #[test]
    fn derived_values_have_no_fixed_width_bound() {
        let mut value = Germ::point(q("0.1234567"));
        for _ in 0..8 { value = value.mul(&value).mul(&value); }
        assert!(value.value().denom().bits() > 100_000);
        assert_eq!(Germ::point(q("1e100")).floor_clamped(0, 30).unwrap(), 30);
        assert_eq!(Germ::left(q("0")).floor_clamped(-10, 10).unwrap(), -1);
    }
}

impl Germ {
    /// IEEE evaluation on the approached side, for discrete decisions driven by
    /// numeric controls. Ordinary continuous evaluation still uses `numeric`.
    pub fn numeric_side(&self) -> Result<f32> {
        let value=self.numeric()?;
        for (neighbor,order) in [(value.next_down(),Ordering::Less),(value.next_up(),Ordering::Greater)] {
            if neighbor.is_finite() {
                let midpoint=(ieee(value)?+ieee(neighbor)?)/Rational::from_integer(2.into());
                if self.value()==&midpoint && self.compare_point(&midpoint)==order {return Ok(neighbor)}
            }
        }
        Ok(value)
    }
}
