//! The pixel graph's image layer (docs/design/motion-model.md §10, stage 5).
//!
//! The vector renderer stays 8-bit and gamma-space for designs. A frame whose graph has image nodes (pixel sinks,
//! `out`, or nodes that read `@frame`, `@layer.render` or `@layer.backdrop`) draws those sources into `Rgba16Float`
//! targets with the same blend math ([`crate::Engine::float_capture`]), and they enter the graph as tagged
//! [`Image`]s:
//!
//! - an [`Image`] is a [`Domain`] (a pixel rectangle in the frame's output space, which may be larger than the
//!   frame so blurs and transforms don't clip) plus named [`ChannelGroup`]s of up to four channels (`rgba`,
//!   `depth.z`, `matte.fg`), each one texture;
//! - a colour group carries a [`ColorTag`] and an [`AlphaMode`]; `rgba` is premultiplied; data groups (depth,
//!   vectors) are [`ColorTag::Data`] and stored as `f32`;
//! - textures come from a [`TexturePool`] with a byte budget, so a graph can't take the GPU down with it.
//!
//! Nodes declare which groups they process and pass the others through (Nuke's rule).

pub mod color;
pub mod exec;
pub mod image;
pub mod kernels;
pub mod nodes;
pub mod pass;
pub mod pool;
pub mod sprites;

pub use exec::{Args, ExecStats, Executor, Param, Params, PendingRead};
pub use image::{AlphaMode, ChannelGroup, ColorTag, Domain, Image, Storage, RGBA};
pub use pass::{AgentRun, Download, Kernel, PassInput, Passes};
pub use pool::{PoolStats, PoolTexture, TexturePool, DEFAULT_BUDGET_BYTES};

/// This process renders previews (graph.look, screenshots), not exports: an agent shader past its time budget draws a
/// magenta checker and warns instead of failing the render (moda-render `--preview`, `serve`).
static PREVIEW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// The device is a software adapter (llvmpipe: CI, cloud VMs): agent shaders' time budget is
/// [`SOFTWARE_ADAPTER_BUDGET_FACTOR`] times larger, unless `MODA_SHADER_BUDGET_MS` sets it.
static SOFTWARE_ADAPTER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// How much longer a software adapter may take than a GPU.
pub const SOFTWARE_ADAPTER_BUDGET_FACTOR: f64 = 40.0;

pub fn set_preview(on: bool) {
    PREVIEW.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn is_preview() -> bool {
    PREVIEW.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_software_adapter(on: bool) {
    SOFTWARE_ADAPTER.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Agent shaders' time budget, ms per 1920×1080 output px: `MODA_SHADER_BUDGET_MS`, else
/// [`exec::DEFAULT_SHADER_BUDGET_MS`] (× [`SOFTWARE_ADAPTER_BUDGET_FACTOR`] on a software adapter).
pub fn agent_budget_per_1080p() -> f64 {
    match std::env::var("MODA_SHADER_BUDGET_MS").ok().and_then(|v| v.parse::<f64>().ok()).filter(|v| *v > 0.0) {
        Some(ms) => ms,
        None if SOFTWARE_ADAPTER.load(std::sync::atomic::Ordering::Relaxed) => {
            exec::DEFAULT_SHADER_BUDGET_MS * SOFTWARE_ADAPTER_BUDGET_FACTOR
        }
        None => exec::DEFAULT_SHADER_BUDGET_MS,
    }
}

/// A typed pixel-graph failure (`moda-render` prints `graph <code>: <message>`).
#[derive(Debug, Clone, PartialEq)]
pub enum PixelError {
    /// Allocating would take the pool past its byte budget.
    Budget { needed: u64, in_use: u64, budget: u64 },
    /// A domain larger than the device's texture limit.
    TooLarge { w: u32, h: u32, max: u32 },
    /// A node asked for a channel group the image doesn't have.
    MissingGroup { group: String },
    /// Anything else the executor rejects (bad parameters, a shader that doesn't compile).
    Invalid(String),
    /// An agent shader's typed failure: `shader_invalid` (it doesn't compile on the device) or
    /// `shader_budget_exceeded` (it ran past its time budget). The message starts with the code where it can travel
    /// as plain text (a layer effect's render error).
    Shader { code: &'static str, message: String },
}

impl PixelError {
    pub fn code(&self) -> &'static str {
        match self {
            PixelError::Budget { .. } => "pixel_budget",
            PixelError::TooLarge { .. } => "pixel_too_large",
            PixelError::MissingGroup { .. } => "missing_channel_group",
            PixelError::Invalid(_) => "pixel_invalid",
            PixelError::Shader { code, .. } => code,
        }
    }
}

impl std::fmt::Display for PixelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PixelError::Budget { needed, in_use, budget } => write!(
                f,
                "the pixel graph needs {} MB more GPU memory but {} of its {} MB are in use (fewer or smaller \
                 images, a smaller domain, or MODA_PIXEL_BUDGET_MB)",
                needed.div_ceil(1 << 20),
                in_use.div_ceil(1 << 20),
                budget >> 20
            ),
            PixelError::TooLarge { w, h, max } => {
                write!(f, "an image of {w}×{h} px is past the GPU's {max} px limit (a smaller blur or transform)")
            }
            PixelError::MissingGroup { group } => write!(f, "the image has no `{group}` channels"),
            PixelError::Invalid(m) => f.write_str(m),
            PixelError::Shader { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for PixelError {}
