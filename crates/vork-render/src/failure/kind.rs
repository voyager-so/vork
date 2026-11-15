//! The failure categories themselves: plain data with no renderer dependency, so a harness that
//! compiles one renderer file on its own (Greenlight's sound probe) builds this file as it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureKind {
    Usage,
    Scene,
    /// GPU work failed: device loss, allocation, a device or renderer limit.
    Device,
    /// A GPU, encoder, denoiser or Slang toolchain could not be set up; `vork setup` reports them.
    Setup,
}
impl std::fmt::Display for FailureKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Usage => "invalid command usage",
            Self::Scene => "invalid scene",
            Self::Device => "device or encoder failure",
            Self::Setup => "setup failure",
        })
    }
}
impl std::error::Error for FailureKind {}
