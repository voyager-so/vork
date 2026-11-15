//! Shared resource admission diagnostic for renderer and encoder.
/// The refusal every device-limit check returns: a typed device failure of a working device,
/// wherever it surfaces.
pub fn device_limit(label: impl std::fmt::Display, requested: impl std::fmt::Display, limit: impl std::fmt::Display, remedy: &str) -> crate::DeviceLimit {
    crate::DeviceLimit(
        format!("{label} exceeds device limit: requested {requested}; limit {limit}; {remedy}; see vork help limits")
    )
}
