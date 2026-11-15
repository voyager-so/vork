//! `Dyn` values for art / scatter / pattern brushes (brushes.md §4.4): the
//! calligraphic nib's `Dyn`, `InkAt`, `ink_at` and `hash01`
//! (`stroke_outline/nib.rs`) — one implementation — plus the pieces the
//! placers add: `largest`, optional ink and per-copy seeded draws.

use crate::value::Value;

use crate::stroke_model::InkChannels;
pub use crate::stroke_outline::nib::{hash01, Dyn, InkAt, Vary};

/// `normalizeDyn(raw, fallback, lo, hi)`: [`Dyn::parse`]. An inverted range
/// (`max < min`) is kept: the driver runs backwards (Krita / Clip Studio).
pub fn parse_dyn(v: Option<&Value>, fallback: f64, lo: f64, hi: f64) -> Dyn {
    Dyn::parse(v, fallback, lo, hi)
}

/// The largest value `d` can take.
pub fn largest(d: &Dyn) -> f64 {
    match d.vary {
        None => d.value,
        Some(_) => d.value.max(d.min).max(d.max),
    }
}

/// [`crate::stroke_outline::nib::ink_at`] for an optional channel set.
pub fn ink_at(ink: Option<&InkChannels>, t: f64) -> InkAt {
    ink.map_or_else(|| InkAt::at_t(t), |i| crate::stroke_outline::nib::ink_at(i, t))
}

/// A seeded draw for parameter `salt` of item `index` (0 for per-stroke draws).
pub fn draw(seed: u32, salt: u64, index: u64) -> f64 {
    hash01(
        ((seed as u64) << 32)
            ^ salt.wrapping_mul(0x0100_0000_01B3)
            ^ index.wrapping_mul(0x9E37_79B9),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::json;

    #[test]
    fn parse_mirrors_normalize_dyn() {
        assert_eq!(
            parse_dyn(Some(&json!(2.5)), 1.0, 0.0, f64::INFINITY),
            Dyn::fixed(2.5)
        );
        assert_eq!(
            parse_dyn(Some(&json!(-3)), 1.0, 0.0, f64::INFINITY),
            Dyn::fixed(0.0)
        );
        assert_eq!(parse_dyn(None, 1.0, 0.0, f64::INFINITY), Dyn::fixed(1.0));
        assert_eq!(
            parse_dyn(Some(&json!("x")), 1.0, 0.0, f64::INFINITY),
            Dyn::fixed(1.0)
        );
        let d = parse_dyn(
            Some(&json!({"value": 1, "vary": "pressure", "min": 2, "max": 0.5})),
            1.0,
            0.0,
            f64::INFINITY,
        );
        assert_eq!(
            (d.min, d.max, d.vary),
            (2.0, 0.5, Some(Vary::Pressure)),
            "an inverted range is kept, not swapped"
        );
        let d = parse_dyn(Some(&json!({"value": 1, "vary": "bogus"})), 1.0, 0.0, 9.0);
        assert_eq!(d, Dyn::fixed(1.0));
    }

    #[test]
    fn eval_maps_driver_into_range() {
        let d = Dyn {
            value: 1.0,
            vary: Some(Vary::Pressure),
            min: 0.5,
            max: 1.5,
        };
        let ink = InkAt {
            pressure: Some(0.5),
            ..InkAt::default()
        };
        assert_eq!(d.eval(&ink, 0.0), 1.0);
        assert_eq!(
            d.eval(&InkAt::default(), 0.0),
            1.0,
            "missing channel → value"
        );
        let r = Dyn {
            vary: Some(Vary::Random),
            ..d
        };
        assert_eq!(r.eval(&InkAt::default(), 1.0), 1.5);
    }

    #[test]
    fn wheel_drives_its_dyn_and_falls_back_without_the_channel() {
        let d = parse_dyn(
            Some(&json!({"value": 5, "vary": "wheel", "min": 2, "max": 10})),
            1.0,
            0.0,
            f64::INFINITY,
        );
        assert_eq!(d.vary, Some(Vary::Wheel));
        assert!(d.uses_ink());
        let ink = crate::stroke_model::parse_ink_channels(&json!({"t": [0, 1], "wheel": [0, 1]})).unwrap();
        assert_eq!(d.eval(&ink_at(Some(&ink), 0.0), 0.0), 2.0);
        assert_eq!(d.eval(&ink_at(Some(&ink), 0.25), 0.0), 4.0);
        assert_eq!(d.eval(&ink_at(Some(&ink), 1.0), 0.0), 10.0);
        let no_wheel = crate::stroke_model::parse_ink_channels(&json!({"t": [0, 1], "pressure": [0, 1]})).unwrap();
        assert_eq!(d.eval(&ink_at(Some(&no_wheel), 0.5), 0.0), 5.0, "missing channel → value");
        assert_eq!(d.eval(&ink_at(None, 0.5), 0.0), 5.0);
    }

    #[test]
    fn speed_drives_its_dyn_and_fade_follows_the_path_without_channels() {
        let speed = parse_dyn(
            Some(&json!({"value": 5, "vary": "speed", "min": 2, "max": 10})),
            1.0,
            0.0,
            f64::INFINITY,
        );
        assert_eq!(speed.vary, Some(Vary::Speed));
        assert!(speed.uses_ink());
        let ink = crate::stroke_model::parse_ink_channels(&json!({"t": [0, 1], "speed": [0, 1]})).unwrap();
        assert_eq!(speed.eval(&ink_at(Some(&ink), 0.0), 0.0), 10.0, "slow → max");
        assert_eq!(speed.eval(&ink_at(Some(&ink), 1.0), 0.0), 2.0, "fast → min (ink thins when flicked)");
        assert_eq!(speed.eval(&ink_at(None, 0.5), 0.0), 5.0, "no speed channel → value");

        let fade = parse_dyn(
            Some(&json!({"value": 5, "vary": "fade", "min": 2, "max": 10})),
            1.0,
            0.0,
            f64::INFINITY,
        );
        assert_eq!(fade.vary, Some(Vary::Fade));
        assert!(fade.uses_ink(), "evaluated along the path");
        // A mouse stroke (no channels) fades too: start → max, end → min.
        assert_eq!(fade.eval(&ink_at(None, 0.0), 0.0), 10.0);
        assert_eq!(fade.eval(&ink_at(None, 0.5), 0.0), 6.0);
        assert_eq!(fade.eval(&ink_at(None, 1.0), 0.0), 2.0);
        assert_eq!(fade.eval(&ink_at(Some(&ink), 1.0), 0.0), 2.0, "with channels too");
    }

    #[test]
    fn inverted_ranges_run_the_driver_backwards() {
        let inv = |vary: &str| {
            parse_dyn(
                Some(&json!({"value": 5, "vary": vary, "min": 10, "max": 2})),
                1.0,
                0.0,
                f64::INFINITY,
            )
        };
        // Pressure: light → min (10, thick), full → max (2, thin).
        let p = inv("pressure");
        let at = |pressure: f64| InkAt {
            pressure: Some(pressure),
            ..InkAt::default()
        };
        assert_eq!(p.eval(&at(0.0), 0.0), 10.0);
        assert_eq!(p.eval(&at(0.25), 0.0), 8.0);
        assert_eq!(p.eval(&at(1.0), 0.0), 2.0);
        assert_eq!(largest(&p), 10.0, "largest covers an inverted range");

        // Speed: slow → max (2), fast → min (10): a flicked stroke draws thicker.
        let s = inv("speed");
        let ink = crate::stroke_model::parse_ink_channels(&json!({"t": [0, 1], "speed": [0, 1]})).unwrap();
        assert_eq!(s.eval(&ink_at(Some(&ink), 0.0), 0.0), 2.0, "slow → max (thin)");
        assert_eq!(s.eval(&ink_at(Some(&ink), 1.0), 0.0), 10.0, "fast → min (thick)");

        // Fade: start → max (2), end → min (10): the stroke swells toward its end.
        let f = inv("fade");
        assert_eq!(f.eval(&ink_at(None, 0.0), 0.0), 2.0);
        assert_eq!(f.eval(&ink_at(None, 0.5), 0.0), 6.0);
        assert_eq!(f.eval(&ink_at(None, 1.0), 0.0), 10.0);
    }

    #[test]
    fn draws_are_deterministic_and_spread() {
        assert_eq!(draw(7, 1, 3), draw(7, 1, 3));
        assert_ne!(draw(7, 1, 3), draw(7, 1, 4));
        assert_ne!(draw(7, 1, 3), draw(8, 1, 3));
        let mean: f64 = (0..1000).map(|i| draw(1, 2, i)).sum::<f64>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05);
    }
}
