use super::*;
use anyhow::bail;
use std::collections::BTreeSet;

enum Format { Ascii, Little, Big }
#[derive(Clone, Copy)]
enum Scalar { Float, Double }
#[derive(Clone, Copy)]
enum Field { Position(usize), Scale(usize), Opacity, Rotation(usize), Dc(usize), Rest(usize), Normal }
struct Property { scalar: Scalar, field: Field }

pub(crate) struct Header {
    pub metadata: Metadata,
    format: Format,
    properties: Vec<Property>,
}

// Header lines and ASCII point records have a bounded number of supported fields.
// Bound a single line before allocation, including untrusted comments.
fn line(r: &mut impl BufRead, out: &mut String) -> Result<()> {
    out.clear();
    let bytes = r.take(65537).read_line(out).context("invalid PLY text; re-export the capture")?;
    ensure!(bytes != 0 && bytes <= 65536,
        "PLY line is missing or exceeds 65536 bytes; re-export standard 3DGS PLY");
    Ok(())
}

impl Header {
    pub fn read(mut r: impl BufRead) -> Result<Self> {
        let mut text = String::new();
        line(&mut r, &mut text)?;
        ensure!(text.trim_end() == "ply", "invalid PLY magic; export standard 3DGS PLY");
        line(&mut r, &mut text)?;
        let format = match text.trim_end() {
            "format ascii 1.0" => Format::Ascii,
            "format binary_little_endian 1.0" => Format::Little,
            "format binary_big_endian 1.0" => Format::Big,
            _ => bail!("unsupported PLY encoding; export ASCII or binary PLY 1.0"),
        };
        let mut count = None;
        let mut properties = Vec::new();
        let mut names = BTreeSet::new();
        let mut filter = None;
        let mut vertical_axis = None;
        let mut declared_degree = None;
        let mut header_bytes = 0;
        loop {
            line(&mut r, &mut text)?;
            header_bytes += text.len();
            ensure!(header_bytes <= 1024*1024, "PLY header exceeds 1 MiB; re-export standard 3DGS PLY");
            let words: Vec<_> = text.split_whitespace().collect();
            match words.as_slice() {
                ["end_header"] => break,
                ["comment", "SplatRenderMode:", mode] => {
                    ensure!(filter.is_none(), "duplicate PLY SplatRenderMode; re-export the capture");
                    filter = Some(match *mode {
                        "classic" => Filter::Classic,
                        "mip" => Filter::Mip,
                        _ => bail!("unsupported PLY SplatRenderMode {mode}; export classic or mip 3DGS"),
                    });
                }
                ["comment", "Vertical", "axis:", x, y, z] => {
                    ensure!(vertical_axis.is_none(), "duplicate PLY Vertical axis; re-export the capture");
                    let parse = |v: &str| v.parse::<f32>().context("invalid PLY Vertical axis; export a finite nonzero up vector");
                    let v = [parse(x)?, parse(y)?, parse(z)?];
                    ensure!(v.iter().all(|v| v.is_finite()) && v.iter().any(|v| *v != 0.),
                        "invalid PLY Vertical axis; export a finite nonzero up vector");
                    vertical_axis = Some(v);
                }
                ["comment", "SH", "degree:", degree] => {
                    ensure!(declared_degree.is_none(), "duplicate PLY SH degree; re-export the capture");
                    declared_degree = Some(degree.parse::<u8>().context("invalid PLY SH degree; export degree 0 through 3")?);
                }
                ["comment", "SplatRenderMode:", ..] | ["comment", "Vertical", "axis:", ..] | ["comment", "SH", "degree:", ..] => {
                    bail!("invalid PLY splat metadata; re-export the capture");
                }
                ["comment", ..] | ["obj_info", ..] => {},
                ["element", "vertex", n] if count.is_none() => {
                    let n = n.parse::<u32>().context("invalid PLY vertex count; export at most 4294967295 points")?;
                    ensure!(n > 0, "PLY contains no Gaussian splats; export a nonempty capture");
                    count = Some(n);
                }
                ["element", ..] => bail!("unsupported PLY element; export a Gaussian point cloud without mesh faces"),
                ["property", kind, name] if count.is_some() => {
                    ensure!(names.insert((*name).to_owned()), "duplicate PLY property {name}; re-export the capture");
                    let scalar = match *kind {
                        "float" | "float32" => Scalar::Float,
                        "double" | "float64" => Scalar::Double,
                        _ => bail!("unsupported PLY scalar {kind}; export float or double Gaussian attributes"),
                    };
                    let indexed = |prefix: &str, max: usize| -> Option<usize> {
                        name.strip_prefix(prefix).and_then(|s| s.parse::<usize>().ok())
                            .filter(|n| *n < max && format!("{prefix}{n}") == *name)
                    };
                    let field = match *name {
                        "x" => Field::Position(0), "y" => Field::Position(1), "z" => Field::Position(2),
                        "opacity" => Field::Opacity,
                        "nx" | "ny" | "nz" => Field::Normal,
                        _ if indexed("scale_",3).is_some() => Field::Scale(indexed("scale_",3).unwrap()),
                        _ if indexed("rot_",4).is_some() => Field::Rotation(indexed("rot_",4).unwrap()),
                        _ if indexed("f_dc_",3).is_some() => Field::Dc(indexed("f_dc_",3).unwrap()),
                        _ if indexed("f_rest_",45).is_some() => Field::Rest(indexed("f_rest_",45).unwrap()),
                        _ => bail!("unsupported PLY property {name}; export standard 3DGS attributes with SH degree 0 through 3"),
                    };
                    properties.push(Property {scalar, field});
                }
                _ => bail!("unsupported PLY header entry; export standard 3DGS PLY"),
            }
        }
        for required in ["x","y","z","scale_0","scale_1","scale_2","opacity","rot_0","rot_1","rot_2","rot_3","f_dc_0","f_dc_1","f_dc_2"] {
            ensure!(names.contains(required), "PLY is missing Gaussian attribute {required}; export standard 3DGS PLY");
        }
        let rest = properties.iter().filter(|p| matches!(p.field,Field::Rest(_))).count();
        let degree = match rest {0 => 0, 9 => 1, 24 => 2, 45 => 3,
            _ => bail!("incomplete PLY spherical harmonics; export every coefficient through degree 0, 1, 2 or 3"),};
        for i in 0..rest {
            ensure!(names.contains(&format!("f_rest_{i}")), "PLY is missing f_rest_{i}; export complete spherical harmonics");
        }
        ensure!(declared_degree.is_none_or(|d| d == degree), "PLY SH degree disagrees with its attributes; re-export the capture");
        Ok(Self { metadata: Metadata {
            count: count.context("PLY has no vertex element; export a Gaussian point cloud")?,
            degree, filter: filter.unwrap_or(Filter::Classic), vertical_axis,
        }, format, properties })
    }
}

pub(crate) fn read(mut r: impl BufRead, available: u64) -> Result<Cloud> {
    let header = Header::read(&mut r)?;
    let mut cloud = Cloud::allocate(header.metadata, available)?;
    let rest_per_channel = cloud.metadata.coefficient_count() - 1;
    let mut text = String::new();
    for row in 0..cloud.metadata.count as usize {
        let mut ascii = if matches!(header.format, Format::Ascii) {
            line(&mut r, &mut text)?;
            Some(text.split_whitespace())
        } else {None};
        for property in &header.properties {
            let value = if let Some(values) = &mut ascii {
                let value = values.next().context("PLY vertex is truncated; re-export the capture")?;
                value.parse::<f32>().context("invalid PLY attribute number; re-export the capture")?
            } else {
                let read = |r: &mut dyn Read, b: &mut [u8]| -> Result<()> {
                    r.read_exact(b).context("PLY payload is truncated; re-export the capture")
                };
                match property.scalar {
                    Scalar::Float => {
                        let mut b = [0;4]; read(&mut r, &mut b)?;
                        if matches!(header.format, Format::Little) {f32::from_le_bytes(b)} else {f32::from_be_bytes(b)}
                    }
                    Scalar::Double => {
                        let mut b = [0;8]; read(&mut r, &mut b)?;
                        (if matches!(header.format, Format::Little) {f64::from_le_bytes(b)} else {f64::from_be_bytes(b)}) as f32
                    }
                }
            };
            // Sigmoid accepts infinite logits (exactly transparent/opaque); no
            // other meaningful attribute may silently lose a nonfinite value.
            ensure!(value.is_finite() || (matches!(property.field,Field::Opacity) && !value.is_nan()),
                "PLY splat {row} has a nonfinite attribute; re-export the capture");
            match property.field {
                Field::Position(i) => cloud.positions[row][i] = value,
                Field::Scale(i) => cloud.scales[row][i] = value.exp(),
                Field::Opacity => cloud.opacities[row] = if value >= 0. {1. / (1. + (-value).exp())} else {let e=value.exp();e/(1.+e)},
                // 3DGS PLY stores WXYZ; canonical glTF attributes store XYZW.
                Field::Rotation(i) => cloud.rotations[row][(i+3)%4] = value,
                Field::Dc(i) => cloud.harmonics[0][row][i] = value,
                // PLY packs all coefficients for R, then G, then B.
                Field::Rest(i) => cloud.harmonics[1+i%rest_per_channel][row][i/rest_per_channel] = value,
                Field::Normal => {},
            }
        }
        ensure!(ascii.is_none_or(|mut values| values.next().is_none()),
            "PLY vertex has extra values; re-export the capture");
    }
    if matches!(header.format, Format::Ascii) {
        loop {
            let bytes = r.fill_buf()?;
            if bytes.is_empty() {break;}
            ensure!(bytes.iter().all(|b| b.is_ascii_whitespace()), "PLY has extra vertices or data; correct its vertex count");
            let n=bytes.len();r.consume(n);
        }
    } else {end(&mut r, "PLY")?;}
    cloud.validate()?;
    Ok(cloud)
}
