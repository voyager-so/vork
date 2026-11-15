use crate::{Time, constants::GOP_SECONDS};
use anyhow::{Result, ensure};
use std::path::Path;

// CMTimeScale and FFmpeg's track timescale are signed 32-bit integers.
const MAX_NUMERATOR: i64 = i32::MAX as i64;
// MOV muxing rewrites a DTS delta >= INT_MAX; native CMTime has no such limit.
const FFMPEG_MAX_DENOMINATOR: i64 = i32::MAX as i64 - 1;

/// Shared preflight for the movie clock, before any destination is invalidated.
/// A reduced p/q uses timescale p and exactly q ticks per frame.
pub fn validate_video_rate(fps: Time, path: &Path) -> Result<()> {
    validate_rate(
        fps,
        path,
        if cfg!(target_os = "macos") {
            i32::MAX as i64
        } else {
            FFMPEG_MAX_DENOMINATOR
        },
    )
}

pub(crate) fn validate_ffmpeg_rate(fps: Time, path: &Path) -> Result<()> {
    validate_rate(fps, path, FFMPEG_MAX_DENOMINATOR)
}

fn validate_rate(fps: Time, path: &Path, max_denominator: i64) -> Result<()> {
    ensure!(fps > Time::ZERO, "fps must be positive; use fps 24");
    if fps.numerator() > MAX_NUMERATOR || fps.denominator() > max_denominator {
        let format = if path
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("mov"))
        {
            "MOV"
        } else {
            "MP4"
        };
        // Advice also fits FFmpeg fallback and the default keyframe interval.
        // It never changes the authored clock.
        let usable = fps.min(Time::integer(MAX_NUMERATOR / i64::from(GOP_SECONDS)));
        let suggested = nearest(usable, MAX_NUMERATOR, FFMPEG_MAX_DENOMINATOR);
        anyhow::bail!("{format} cannot represent fps {fps}; use fps \"{suggested}\"");
    }
    Ok(())
}

// Continued fractions with bounds on both components. All comparisons use
// integers, including the choice between the last convergent and semiconvergent.
// i64 input × two i32 bounds fits i128. This only supplies explicit user advice;
// it never changes the authored rate or a renderer's clock.
fn nearest(rate: Time, max_num: i64, max_den: i64) -> Time {
    let (num, den) = (i128::from(rate.numerator()), i128::from(rate.denominator()));
    let (mut n, mut d) = (num, den);
    let (mut p0, mut q0, mut p1, mut q1) = (0i128, 1i128, 1i128, 0i128);
    loop {
        let a = n / d;
        let mut bound = i128::MAX;
        if p1 != 0 {
            bound = bound.min((i128::from(max_num) - p0) / p1);
        }
        if q1 != 0 {
            bound = bound.min((i128::from(max_den) - q0) / q1);
        }
        if a > bound {
            let (p, q) = (p0 + bound * p1, q0 + bound * q1);
            let (p, q) = if p1 > 0
                && q1 > 0
                && (p == 0
                    || q == 0
                    || (p1 * den - num * q1).abs() * q <= (p * den - num * q).abs() * q1)
            {
                (p1, q1)
            } else {
                (p, q)
            };
            return Time::new(p as i64, q as i64).expect("bounded positive fraction");
        }
        (p0, q0, p1, q1) = (p1, q1, p0 + a * p1, q0 + a * q1);
        (n, d) = (d, n % d);
        if d == 0 {
            return Time::new(p1 as i64, q1 as i64).expect("bounded fraction");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suggestions_are_nearest_positive_bounded_fractions() {
        for numerator in 1..40 {
            for denominator in 1..40 {
                let rate = Time::new(numerator, denominator).unwrap();
                let got = nearest(rate, 7, 6);
                let error = (got.numerator() * denominator - numerator * got.denominator()).abs();
                for p in 1..=7 {
                    for q in 1..=6 {
                        assert!(
                            error * q
                                <= (p * denominator - numerator * q).abs() * got.denominator(),
                            "{rate}: {got} vs {p}/{q}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn ffmpeg_duration_limit_preserves_native_capacity() {
        let path = Path::new("clip.mp4");
        let edge = Time::new(1, i32::MAX as i64).unwrap();
        validate_rate(edge, path, i32::MAX as i64).unwrap();
        assert!(validate_ffmpeg_rate(edge, path).is_err());
        assert_eq!(
            validate_video_rate(edge, path).is_ok(),
            cfg!(target_os = "macos")
        );
        validate_ffmpeg_rate(Time::new(1, i32::MAX as i64 - 1).unwrap(), path).unwrap();
    }

    #[test]
    fn genuine_limits_name_one_usable_rate() {
        for (rate, suggestion) in [
            ("5.64705898192515", "1113587715/197197819"),
            ("1/2147483648", "1/2147483646"),
            ("1/9223372036854775807", "1/2147483646"),
            ("9223372036854775807", "1073741823"),
        ] {
            let fps = rate.parse().unwrap();
            for extension in ["mp4", "mov"] {
                let path = std::path::PathBuf::from(format!("clip.{extension}"));
                let message = validate_video_rate(fps, &path).unwrap_err().to_string();
                assert_eq!(
                    message,
                    format!(
                        "{} cannot represent fps {fps}; use fps \"{suggestion}\"",
                        extension.to_uppercase()
                    )
                );
                validate_video_rate(suggestion.parse().unwrap(), &path).unwrap();
            }
        }
    }
}
