//! Gaussian point geometry, independent of the renderer and its traversal backend.
//!
//! Attribute semantics follow the ratified KHR_gaussian_splatting extension.
//! Imported coefficients remain in the capture's frame; apply its orientation to
//! the object, including the direction used to evaluate spherical harmonics.
use anyhow::{Context, Result, ensure};
use glam::Quat;
use std::io::{BufRead, BufReader, Read};

mod ply;
mod spz;
pub mod shape;

pub const POSITION: &str = "POSITION";
pub const SCALE: &str = "KHR_gaussian_splatting:SCALE";
pub const ROTATION: &str = "KHR_gaussian_splatting:ROTATION";
pub const OPACITY: &str = "KHR_gaussian_splatting:OPACITY";

pub fn sh_attribute(degree: u8, coefficient: u8) -> String {
    format!("KHR_gaussian_splatting:SH_DEGREE_{degree}_COEF_{coefficient}")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Classic,
    /// The training-time 3D smoothing is already baked into scale and opacity.
    Mip,
}

#[derive(Clone, Debug)]
pub struct Metadata {
    pub count: u32,
    pub degree: u8,
    pub filter: Filter,
    /// PLY's capture up vector, before any coordinate conversion.
    pub vertical_axis: Option<[f32; 3]>,
}

impl Metadata {
    pub fn coefficient_count(&self) -> usize {
        (usize::from(self.degree) + 1).pow(2)
    }

    /// Full precision attributes only. The caller must also reserve acceleration,
    /// upload and renderer storage before starting expensive preparation.
    pub fn attribute_bytes(&self) -> u64 {
        u64::from(self.count) * (11 + self.coefficient_count() as u64 * 3) * 4
    }

    pub fn admit(&self, available: u64) -> Result<()> {
        let required = self.attribute_bytes();
        ensure!(required <= available,
            "Gaussian splat attributes need {required} bytes; only {available} bytes available; use a device with more memory or a smaller capture");
        Ok(())
    }
}

/// SoA columns retain every source point and SH coefficient at full precision.
#[derive(Clone, Debug)]
pub struct Cloud {
    pub metadata: Metadata,
    pub positions: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    /// glTF XYZW unit quaternion order.
    pub rotations: Vec<[f32; 4]>,
    pub opacities: Vec<f32>,
    /// One RGB column per coefficient, ordered by degree then ascending m.
    pub harmonics: Vec<Vec<[f32; 3]>>,
}

impl Cloud {
    pub fn shape(&self, index: usize) -> shape::Shape {
        shape::Shape {
            position: self.positions[index].into(),
            scale: self.scales[index].into(),
            rotation: Quat::from_array(self.rotations[index]),
            opacity: self.opacities[index],
        }
    }

    fn allocate(metadata: Metadata, available: u64) -> Result<Self> {
        metadata.admit(available)?;
        fn column<T: Default + Clone>(n: usize) -> Result<Vec<T>> {
            let mut v = Vec::new();
            v.try_reserve_exact(n).context("Gaussian splat attributes do not fit host memory; use a smaller capture")?;
            v.resize(n, T::default());
            Ok(v)
        }
        let n = metadata.count as usize;
        Ok(Self {
            positions: column(n)?, scales: column(n)?, rotations: column(n)?,
            opacities: column(n)?,
            harmonics: (0..metadata.coefficient_count()).map(|_| column(n)).collect::<Result<_>>()?,
            metadata,
        })
    }

    fn validate(&mut self) -> Result<()> {
        for i in 0..self.metadata.count as usize {
            ensure!(self.positions[i].iter().all(|v| v.is_finite()),
                "Gaussian splat {i} has a nonfinite position; re-export the capture");
            ensure!(self.scales[i].iter().all(|v| v.is_finite() && *v > 0.),
                "Gaussian splat {i} has a nonpositive or nonfinite scale; re-export the capture with positive scales");
            ensure!(self.opacities[i].is_finite() && (0.0..=1.0).contains(&self.opacities[i]),
                "Gaussian splat {i} has an invalid opacity; re-export the capture");
            // Normalize after scaling to avoid overflow/underflow in the norm.
            let q = self.rotations[i];
            let largest = q.iter().fold(0.0f32, |a, b| a.max(b.abs()));
            ensure!(q.iter().all(|v| v.is_finite()) && largest > 0.,
                "Gaussian splat {i} has an invalid rotation; re-export the capture");
            self.rotations[i] = Quat::from_array(q.map(|v| v / largest)).normalize().to_array();
            ensure!(self.harmonics.iter().all(|column| column[i].iter().all(|v| v.is_finite())),
                "Gaussian splat {i} has nonfinite spherical harmonics; re-export the capture");
        }
        Ok(())
    }
}

/// Header inspection does not expand point attributes. It enables device
/// admission before decompression of the payload or acceleration construction.
pub fn inspect_ply(reader: impl BufRead) -> Result<Metadata> {
    Ok(ply::Header::read(reader)?.metadata)
}

pub fn read_ply(reader: impl BufRead, available_attribute_bytes: u64) -> Result<Cloud> {
    ply::read(reader, available_attribute_bytes)
}

pub fn inspect_spz(reader: impl Read) -> Result<Metadata> {
    spz::inspect(BufReader::new(reader))
}

pub fn read_spz(reader: impl Read, available_attribute_bytes: u64) -> Result<Cloud> {
    spz::read(BufReader::new(reader), available_attribute_bytes)
}

fn end(reader: &mut impl Read, format: &str) -> Result<()> {
    let mut byte = [0];
    ensure!(reader.read(&mut byte)? == 0,
        "{format} has unsupported trailing data; export a standard Gaussian splat file");
    Ok(())
}
