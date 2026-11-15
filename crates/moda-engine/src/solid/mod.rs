//! 3D layers (motion-model.md §12.6, phase 2): solids made from a layer — its own outline extruded with a bevel, a
//! primitive fitted to its box, or an imported glTF model — lit and drawn with a depth buffer by
//! [`crate::pixel::Executor::draw_solid`], through the same camera as the 2.5D layers.
//!
//! Coordinates are the layer's own (x right, y down, in px) with z away from the viewer; the host places them in the
//! frame with the layer's 3D matrix. An extrusion's front face lies in the layer's plane (z = 0) and it goes back
//! `depth`; a primitive is centred on the plane.

pub mod env;
pub mod geometry;
pub mod outline;
pub mod render;
pub mod revolve;

pub use geometry::{extrude, primitive, Extrude, Primitive};
pub use outline::{node_shapes, Shape};
pub use render::{EnvironmentLight, Light, LightKind, Material, Shading, SolidItem, SolidPipeline, SolidScene};

use std::sync::Arc;

/// An RGBA8 image (straight alpha, sRGB-encoded) a part samples by its UVs.
#[derive(Clone, Debug, PartialEq)]
pub struct Texture {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Triangles of one colour (and, optionally, one texture).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Part {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    /// Empty, or one per position.
    pub uvs: Vec<[f32; 2]>,
    /// Straight sRGB 0–1; multiplies the texture.
    pub color: [f32; 4],
    pub indices: Vec<u32>,
    pub texture: Option<Arc<Texture>>,
}

impl Part {
    pub fn new(color: [f32; 4]) -> Self {
        Self { color, ..Default::default() }
    }

    /// Add a vertex; returns its index.
    pub fn vertex(&mut self, p: [f64; 3], n: [f64; 3]) -> u32 {
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        let n = if l > 1e-12 { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0, 0.0, -1.0] };
        self.positions.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        self.normals.push([n[0] as f32, n[1] as f32, n[2] as f32]);
        (self.positions.len() - 1) as u32
    }

    pub fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    pub fn quad(&mut self, a: u32, b: u32, c: u32, d: u32) {
        self.indices.extend([a, b, c, a, c, d]);
    }
}

/// A solid: its parts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub parts: Vec<Part>,
}

impl Mesh {
    /// The box of every position, `(min, max)`, or `None` for an empty mesh.
    pub fn bounds(&self) -> Option<([f64; 3], [f64; 3])> {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in self.parts.iter().flat_map(|p| p.positions.iter()) {
            for k in 0..3 {
                lo[k] = lo[k].min(f64::from(p[k]));
                hi[k] = hi[k].max(f64::from(p[k]));
            }
        }
        lo[0].is_finite().then_some((lo, hi))
    }

    pub fn triangles(&self) -> usize {
        self.parts.iter().map(|p| p.indices.len() / 3).sum()
    }
}
