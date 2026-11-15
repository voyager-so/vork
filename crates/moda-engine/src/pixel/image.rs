//! [`Image`]: a domain and named channel groups, each a pooled texture with a colour tag (motion-model.md §10.1).

use super::pool::PoolTexture;
use super::PixelError;

/// The colour encoding of a channel group's values (§10.1, point 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ColorTag {
    /// sRGB-encoded (the vector renderer's output: gamma-space, Rec.709 primaries).
    Srgb,
    /// The working space: linear light, Rec.709 primaries.
    LinearSrgb,
    /// Display P3: P3 primaries, D65, the sRGB transfer.
    DisplayP3,
    LinearP3,
    /// Rec.2020 primaries with the PQ (SMPTE ST 2084) transfer.
    Rec2020Pq,
    /// Rec.2020 primaries with the HLG (ARIB STD-B67) transfer.
    Rec2020Hlg,
    /// ACEScg: AP1 primaries, linear, D60 white.
    AcesCg,
    /// Not colour: depth, vectors, mattes as data. Never converted.
    Data,
}

impl ColorTag {
    pub const ALL: [ColorTag; 8] = [
        ColorTag::Srgb,
        ColorTag::LinearSrgb,
        ColorTag::DisplayP3,
        ColorTag::LinearP3,
        ColorTag::Rec2020Pq,
        ColorTag::Rec2020Hlg,
        ColorTag::AcesCg,
        ColorTag::Data,
    ];

    /// The tag's name in the text form and in lint messages.
    pub const fn name(self) -> &'static str {
        match self {
            ColorTag::Srgb => "srgb",
            ColorTag::LinearSrgb => "linear-srgb",
            ColorTag::DisplayP3 => "display-p3",
            ColorTag::LinearP3 => "linear-p3",
            ColorTag::Rec2020Pq => "rec2020-pq",
            ColorTag::Rec2020Hlg => "rec2020-hlg",
            ColorTag::AcesCg => "acescg",
            ColorTag::Data => "data",
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.name() == s)
    }

    /// Linear light (blurs, glows, motion blur and grading work here).
    pub const fn is_linear(self) -> bool {
        matches!(self, ColorTag::LinearSrgb | ColorTag::LinearP3 | ColorTag::AcesCg)
    }

    pub const fn is_color(self) -> bool {
        !matches!(self, ColorTag::Data)
    }
}

/// How a group's colour channels relate to its alpha.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlphaMode {
    /// Colour is multiplied by alpha (`rgba` in the graph always is).
    Premultiplied,
    Straight,
    /// No alpha channel, or alpha is 1 everywhere.
    Opaque,
}

/// A group's texel storage (§10.1: `f16` by default, `f32` for data channels).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Storage {
    /// 8-bit RGBA, encoded values (a graph-pixel unit handed to the scene renderer as an image).
    U8,
    F16,
    F32,
}

impl Storage {
    pub const fn format(self) -> wgpu::TextureFormat {
        match self {
            Storage::U8 => wgpu::TextureFormat::Rgba8Unorm,
            Storage::F16 => wgpu::TextureFormat::Rgba16Float,
            Storage::F32 => wgpu::TextureFormat::Rgba32Float,
        }
    }

    pub const fn bytes_per_pixel(self) -> u64 {
        match self {
            Storage::U8 => 4,
            Storage::F16 => 8,
            Storage::F32 => 16,
        }
    }
}

/// A pixel rectangle in the frame's output space: `(0, 0)` is the frame's top-left output pixel. Domains may start
/// at negative coordinates and extend past the frame, so a blur or a transform doesn't clip (§10.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Domain {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Domain {
    pub const EMPTY: Domain = Domain { x: 0, y: 0, w: 0, h: 0 };

    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// The frame itself: `w × h` at the origin.
    pub const fn frame(w: u32, h: u32) -> Self {
        Self { x: 0, y: 0, w, h }
    }

    pub const fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    pub fn right(&self) -> i64 {
        self.x as i64 + self.w as i64
    }

    pub fn bottom(&self) -> i64 {
        self.y as i64 + self.h as i64
    }

    fn from_edges(x0: i64, y0: i64, x1: i64, y1: i64) -> Self {
        if x1 <= x0 || y1 <= y0 {
            return Self::EMPTY;
        }
        let clamp = |v: i64| v.clamp(i32::MIN as i64 / 2, i32::MAX as i64 / 2) as i32;
        let (x0, y0) = (clamp(x0), clamp(y0));
        let (x1, y1) = (clamp(x1), clamp(y1));
        Self {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0) as u32,
            h: (y1 - y0).max(0) as u32,
        }
    }

    /// The smallest domain holding both (an empty one adds nothing).
    pub fn union(&self, o: &Domain) -> Domain {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        Self::from_edges(
            (self.x as i64).min(o.x as i64),
            (self.y as i64).min(o.y as i64),
            self.right().max(o.right()),
            self.bottom().max(o.bottom()),
        )
    }

    pub fn intersect(&self, o: &Domain) -> Domain {
        Self::from_edges(
            (self.x as i64).max(o.x as i64),
            (self.y as i64).max(o.y as i64),
            self.right().min(o.right()),
            self.bottom().min(o.bottom()),
        )
    }

    /// Grown by `dx` / `dy` pixels on each side (a blur's reach). An empty domain stays empty.
    pub fn grow(&self, dx: u32, dy: u32) -> Domain {
        if self.is_empty() {
            return *self;
        }
        Self::from_edges(
            self.x as i64 - dx as i64,
            self.y as i64 - dy as i64,
            self.right() + dx as i64,
            self.bottom() + dy as i64,
        )
    }

    /// The pixel bounds of the region `(x0, y0)–(x1, y1)` in output pixels (fractional edges round outward).
    pub fn covering(x0: f64, y0: f64, x1: f64, y1: f64) -> Domain {
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return Self::EMPTY;
        }
        Self::from_edges(x0.floor() as i64, y0.floor() as i64, x1.ceil() as i64, y1.ceil() as i64)
    }

    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x && py >= self.y && (px as i64) < self.right() && (py as i64) < self.bottom()
    }

    pub fn pixels(&self) -> u64 {
        self.w as u64 * self.h as u64
    }
}

/// The colour group every image has.
pub const RGBA: &str = "rgba";

/// One named group of up to four channels, stored as one texture.
#[derive(Clone, Debug)]
pub struct ChannelGroup {
    /// `rgba`, `depth`, `matte`, `motion`, …
    pub name: String,
    /// Channel names in texel order: `r g b a` for `rgba`, `z` for `depth`, `fg` for `matte`. Up to four.
    pub channels: Vec<String>,
    pub tag: ColorTag,
    pub alpha: AlphaMode,
    pub texture: PoolTexture,
}

impl ChannelGroup {
    pub fn storage(&self) -> Storage {
        self.texture.storage()
    }

    /// The index of `channel` in the group's texels.
    pub fn channel(&self, channel: &str) -> Option<usize> {
        self.channels.iter().position(|c| c == channel)
    }
}

/// A float raster with named channel groups, a domain, and on its colour groups a tag and an alpha mode.
#[derive(Clone, Debug)]
pub struct Image {
    pub domain: Domain,
    /// `rgba` first; every group has the image's domain (and so its texture's size).
    pub groups: Vec<ChannelGroup>,
}

impl Image {
    /// An image with only the premultiplied `rgba` group.
    pub fn rgba(domain: Domain, tag: ColorTag, texture: PoolTexture) -> Self {
        assert_eq!(texture.size(), (domain.w, domain.h), "an image texture must match its domain");
        Self {
            domain,
            groups: vec![ChannelGroup {
                name: RGBA.into(),
                channels: ["r", "g", "b", "a"].map(String::from).to_vec(),
                tag,
                alpha: AlphaMode::Premultiplied,
                texture,
            }],
        }
    }

    pub fn group(&self, name: &str) -> Option<&ChannelGroup> {
        self.groups.iter().find(|g| g.name == name)
    }

    pub fn require(&self, name: &str) -> Result<&ChannelGroup, PixelError> {
        self.group(name).ok_or_else(|| PixelError::MissingGroup { group: name.into() })
    }

    /// The `rgba` group (every image the executor makes has one).
    pub fn color(&self) -> &ChannelGroup {
        self.group(RGBA).expect("an image always has rgba")
    }

    pub fn tag(&self) -> ColorTag {
        self.color().tag
    }

    /// Resolve `group.channel` (`depth.z`, or a bare `r` in `rgba`) to its group and channel index.
    pub fn resolve_channel(&self, spec: &str) -> Result<(&ChannelGroup, usize), PixelError> {
        let (g, c) = spec.split_once('.').unwrap_or((RGBA, spec));
        let group = self.require(g)?;
        let i = group
            .channel(c)
            .ok_or_else(|| PixelError::MissingGroup { group: spec.into() })?;
        Ok((group, i))
    }

    /// Replace (or add) a group, keeping the others: how a node passes through what it doesn't process.
    pub fn with_group(mut self, group: ChannelGroup) -> Self {
        assert_eq!(group.texture.size(), (self.domain.w, self.domain.h), "a channel group must match its image domain");
        match self.groups.iter_mut().find(|g| g.name == group.name) {
            Some(slot) => *slot = group,
            None => self.groups.push(group),
        }
        self
    }

    /// GPU bytes this image holds.
    pub fn bytes(&self) -> u64 {
        self.groups.iter().map(|g| g.texture.bytes()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_round_trip_their_names() {
        for t in ColorTag::ALL {
            assert_eq!(ColorTag::from_name(t.name()), Some(t));
        }
        assert_eq!(ColorTag::from_name("rec709"), None);
        assert!(ColorTag::LinearSrgb.is_linear() && !ColorTag::Srgb.is_linear());
        assert!(!ColorTag::Data.is_color());
    }

    #[test]
    fn domains() {
        let f = Domain::frame(1920, 1080);
        assert_eq!(f.pixels(), 1920 * 1080);
        let g = f.grow(30, 10);
        assert_eq!(g, Domain::new(-30, -10, 1980, 1100));
        assert_eq!(g.intersect(&f), f);
        let a = Domain::new(-5, 5, 10, 10);
        let b = Domain::new(100, -20, 5, 5);
        assert_eq!(a.union(&b), Domain::new(-5, -20, 110, 35));
        assert!(a.intersect(&b).is_empty());
        assert_eq!(Domain::EMPTY.union(&a), a);
        assert!(Domain::EMPTY.grow(4, 4).is_empty());
        assert_eq!(Domain::covering(-0.5, 0.25, 10.1, 3.0), Domain::new(-1, 0, 12, 3));
        assert!(Domain::covering(f64::NAN, 0.0, 1.0, 1.0).is_empty());
        assert!(a.contains(-5, 14) && !a.contains(5, 15));
    }

    #[test]
    fn storage_sizes() {
        // §10.2: f16 1080p is 16.6 MB, 4K is 66 MB.
        assert_eq!(Domain::frame(1920, 1080).pixels() * Storage::F16.bytes_per_pixel(), 16_588_800);
        assert_eq!(Domain::frame(3840, 2160).pixels() * Storage::F16.bytes_per_pixel(), 66_355_200);
        assert_eq!(Storage::F32.format(), wgpu::TextureFormat::Rgba32Float);
    }
}
