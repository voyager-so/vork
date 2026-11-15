//! SPZ v2/v3 decoding, from Niantic's documented format (MIT).
//! https://github.com/nianticlabs/spz (src/cc/load-spz.cc and splat-utils.h).
use super::*;
use std::io::Cursor;

fn decoder<R: BufRead>(mut reader: R) -> Result<flate2::bufread::GzDecoder<std::io::Chain<Cursor<[u8;8]>,R>>> {
    let mut prefix=[0u8;8];
    reader.read_exact(&mut prefix).context("SPZ header is truncated; re-export the capture")?;
    if &prefix[..4]==b"NGSP" {
        let version=u32::from_le_bytes(prefix[4..8].try_into().unwrap());
        anyhow::bail!("unsupported SPZ version {version}; export SPZ version 2 or 3, or standard 3DGS PLY");
    }
    ensure!(&prefix[..2]==[0x1f,0x8b],"invalid SPZ compression; export SPZ version 2 or 3, or standard 3DGS PLY");
    Ok(flate2::bufread::GzDecoder::new(Cursor::new(prefix).chain(reader)))
}
pub(crate) fn inspect(reader: impl BufRead) -> Result<Metadata> {
    Ok(Header::read(&mut decoder(reader)?)?.metadata)
}

pub(crate) struct Header {
    pub metadata: Metadata,
    version: u32,
    fractional_bits: u8,
}

impl Header {
    pub fn read(r: &mut impl Read) -> Result<Self> {
        let mut h = [0; 16];
        r.read_exact(&mut h).context("SPZ header is invalid or truncated; re-export the capture")?;
        ensure!(&h[..4] == b"NGSP", "invalid SPZ magic; export a standard SPZ file");
        let version = u32::from_le_bytes(h[4..8].try_into().unwrap());
        ensure!((2..=3).contains(&version), "unsupported SPZ version {version}; export SPZ version 2 or 3, or standard 3DGS PLY");
        let count = u32::from_le_bytes(h[8..12].try_into().unwrap());
        ensure!(count > 0, "SPZ contains no Gaussian splats; export a nonempty capture");
        ensure!(h[12] <= 3, "unsupported SPZ SH degree {}; export degree 0 through 3", h[12]);
        ensure!(h[13] <= 24, "invalid SPZ fractional precision {}; re-export the capture", h[13]);
        ensure!(h[14] & !1 == 0 && h[15] == 0,
            "unsupported SPZ flags or extensions; export standard SPZ without extensions or standard 3DGS PLY");
        Ok(Self { metadata: Metadata {
            count, degree: h[12], filter: if h[14] & 1 != 0 { Filter::Mip } else { Filter::Classic },
            vertical_axis: None,
        }, version, fractional_bits: h[13] })
    }
}

pub(crate) fn read(r: impl BufRead, available: u64) -> Result<Cloud> {
    let mut r = decoder(r)?;
    let header = Header::read(&mut r)?;
    let mut cloud = Cloud::allocate(header.metadata, available)?;
    let scale = 2.0f32.powi(-i32::from(header.fractional_bits));
    let read = |r: &mut dyn Read, b: &mut [u8]| -> Result<()> {
        r.read_exact(b).context("SPZ payload is corrupt or truncated; re-export the capture")
    };
    for p in &mut cloud.positions {
        for c in p {
            let mut b = [0; 3]; read(&mut r, &mut b)?;
            let signed = i32::from_le_bytes([b[0], b[1], b[2], if b[2] & 128 != 0 {255} else {0}]);
            *c = signed as f32 * scale;
        }
    }
    for alpha in &mut cloud.opacities {
        let mut b = [0]; read(&mut r, &mut b)?; *alpha = f32::from(b[0]) / 255.;
    }
    for dc in &mut cloud.harmonics[0] {
        let mut b = [0; 3]; read(&mut r, &mut b)?;
        *dc = b.map(|v| (f32::from(v) / 255. - 0.5) / 0.15);
    }
    for s in &mut cloud.scales {
        let mut b = [0; 3]; read(&mut r, &mut b)?;
        *s = b.map(|v| (f32::from(v) / 16. - 10.).exp());
    }
    for q in &mut cloud.rotations {
        if header.version == 2 {
            let mut b = [0; 3]; read(&mut r, &mut b)?;
            let xyz = b.map(|v| f32::from(v) / 127.5 - 1.);
            *q = [xyz[0], xyz[1], xyz[2], (1. - xyz.iter().map(|v| v*v).sum::<f32>()).max(0.).sqrt()];
        } else {
            let mut b = [0; 4]; read(&mut r, &mut b)?;
            let mut bits = u32::from_le_bytes(b);
            let largest = (bits >> 30) as usize;
            let mut norm = 0.;
            for i in (0..4).rev().filter(|i| *i != largest) {
                q[i] = (bits & 511) as f32 * (std::f32::consts::FRAC_1_SQRT_2 / 511.);
                if bits & 512 != 0 { q[i] = -q[i]; }
                norm += q[i] * q[i]; bits >>= 10;
            }
            ensure!(norm <= 1., "invalid SPZ packed rotation; re-export the capture");
            q[largest] = (1. - norm).sqrt();
        }
    }
    for i in 0..cloud.metadata.count as usize {
        for coefficient in cloud.harmonics.iter_mut().skip(1) {
            let mut b = [0; 3]; read(&mut r, &mut b)?;
            coefficient[i] = b.map(|v| (f32::from(v) - 128.) / 128.);
        }
    }
    // Reading EOF also verifies gzip's checksum and declared uncompressed size.
    end(&mut r, "SPZ")?;
    end(&mut r.into_inner(), "SPZ")?;
    cloud.validate()?;
    Ok(cloud)
}
