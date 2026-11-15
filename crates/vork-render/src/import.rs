//! glTF document compatibility and geometry conversion into static upload pages.
use anyhow::{Context, Result, bail, ensure};

use std::path::Path;

/// Reads a glTF or GLB file once and parses it from those bytes. gltf-json's
/// validation can itself panic on malformed documents; that is reported as an error.
pub(crate) fn open(path: &Path) -> Result<gltf::Gltf> {
    check_extension(path)?;
    let bytes = (|| -> Result<Vec<u8>> {
        let mut file = crate::source_file(path)?;
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut file, &mut bytes)?;
        Ok(bytes)
    })().with_context(|| format!("read {}", path.display()))?;
    let gltf = crate::failure::contain("the glTF parser", || from_slice(&bytes))?;
    warn_instancing(&gltf.document,path);
    Ok(gltf)
}

fn check_extension(path: &Path) -> Result<()> {
    if let Some(extension) = path.extension().and_then(|e| e.to_str()) {
        ensure!(extension.eq_ignore_ascii_case("gltf") || extension.eq_ignore_ascii_case("glb"),
            "unsupported asset format .{extension}: {}; supported formats are glTF/GLB, SPZ and Gaussian PLY", path.display());
    }
    Ok(())
}

fn warn_instancing(document: &gltf::Document,path: &Path) {
    for node in document.nodes() {
        if node.extension_value("EXT_mesh_gpu_instancing").is_some() {
            crate::warnings::dropped("gltf_instancing", serde_json::json!({"file":path,"node":node.index(),"name":node.name(),"extension":"EXT_mesh_gpu_instancing"}),
                "instance placements are ignored; using the node's mesh once. Export separate mesh nodes instead of EXT_mesh_gpu_instancing.".into());
        }
    }
}

/// Source admission needs the GLB document and BIN extent, not the BIN payload.
/// Installation still uses `open` and the original buffer loader.
pub(crate) fn validate_file(path: &Path,base: &Path) -> Result<Vec<std::path::PathBuf>> {
    check_extension(path)?;
    let mut file=crate::source_file(path).with_context(||format!("read {}",path.display()))?;
    let length=file.metadata().with_context(||format!("read {}",path.display()))?.len();
    let (gltf,bin_length)=crate::failure::contain("the glTF parser",||source_document(&mut file,length))
        .map_err(|error|if error.is::<std::io::Error>() {error.context(format!("read {}",path.display()))} else {error})?;
    warn_instancing(&gltf.document,path);
    validate_sources(&gltf.document,bin_length,base)
}

fn source_document(reader: &mut (impl std::io::Read+std::io::Seek),length: u64) -> Result<(gltf::Gltf,Option<u64>)> {
    use std::io::{Read,Seek,SeekFrom};
    let mut magic=Vec::new();
    (&mut *reader).take(4).read_to_end(&mut magic)?;
    reader.seek(SeekFrom::Start(0))?;
    if magic!=b"glTF" {
        // A binary file with an external BOM is refused without touching BIN.
        if magic.starts_with(b"\xEF\xBB\xBF") {
            let mut prefix=Vec::new();
            (&mut *reader).take(7).read_to_end(&mut prefix)?;
            ensure!(!vork_scene::diagnostics::without_bom(&prefix).starts_with(b"glTF"),
                "a UTF-8 byte-order mark precedes binary glTF");
            reader.seek(SeekFrom::Start(0))?;
        }
        let mut bytes=Vec::new();
        reader.read_to_end(&mut bytes)?;
        return Ok((from_slice(&bytes)?,None));
    }
    ensure!(length>=12,"GLB header is truncated");
    let mut header=[0u8;12];reader.read_exact(&mut header)?;
    let declared=u32::from_le_bytes(header[8..12].try_into().unwrap()) as u64;
    ensure!(declared>=12 && declared==length,"GLB header length {declared} does not match the {length}-byte file");
    // Match check_glb: validate every chunk extent before version/type/JSON errors.
    // gltf 1.4 only interprets the first two chunk types, ignoring later chunks.
    let mut chunks=Vec::new();
    let mut reader=std::io::BufReader::new(reader);
    let mut at=12;
    while at<length {
        ensure!(length-at>=8,"GLB chunk header at byte {at} is truncated");
        let mut chunk=[0u8;8];
        // Do not read ahead into JSON or BIN. Later extension chunks can be
        // scanned in buffered windows; seek_relative keeps small skips in
        // memory and seeks over payloads larger than the remaining buffer.
        if chunks.len()<2 {reader.get_mut().read_exact(&mut chunk)?;}
        else {reader.read_exact(&mut chunk)?;}
        let size=u32::from_le_bytes(chunk[..4].try_into().unwrap()) as u64;
        ensure!(size<=length-at-8,"GLB chunk at byte {at} has length {size}, past the declared length {length}");
        if chunks.len()<2 {chunks.push((size,<[u8;4]>::try_from(&chunk[4..]).unwrap()));}
        at+=8+size;
        reader.seek_relative(size as i64)?;
    }
    let current=reader.seek(SeekFrom::End(0))?;
    ensure!(current==length,"GLB header length {declared} does not match the {current}-byte file");
    let version=u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version!=2 {return Err(gltf::Error::Binary(gltf::binary::Error::Version(version)).into());}
    for (index,&(_,kind)) in chunks.iter().enumerate() {
        use gltf::binary::{ChunkType,Error};
        let kind=match &kind {b"JSON"=>ChunkType::Json,b"BIN\0"=>ChunkType::Bin,
            _=>return Err(gltf::Error::Binary(Error::UnknownChunkType(kind)).into())};
        if !matches!((index,kind),(0,ChunkType::Json)|(1,ChunkType::Bin)) {
            return Err(gltf::Error::Binary(Error::ChunkType(kind)).into());
        }
    }
    // Preserve the embedded-JSON parser, including its BOM and normalization
    // behavior, without a placeholder BIN allocation or a second JSON parser.
    let json_length=chunks.first().map(|&(size,_)|size);
    let mut json_glb=header.to_vec();
    if let Some(size)=json_length {
        json_glb.extend((size as u32).to_le_bytes());json_glb.extend(b"JSON");
        json_glb.resize(20+size as usize,0);
        reader.seek(SeekFrom::Start(20))?;reader.get_mut().read_exact(&mut json_glb[20..])?;
    }
    let json_total=json_glb.len() as u32;
    json_glb[8..12].copy_from_slice(&json_total.to_le_bytes());
    Ok((from_slice(&json_glb)?,chunks.get(1).map(|&(size,_)|size)))
}

/// Check filesystem facts without loading buffer payloads or decoding images.
/// Match gltf's URI resolution and four-byte buffer padding. Missing sampled
/// images retain the installer's magenta substitution; other I/O errors refuse.
fn validate_sources(gltf: &gltf::Document, bin_length: Option<u64>, base: &Path) -> Result<Vec<std::path::PathBuf>> {
    fn external(uri: &str, base: &Path) -> Result<Option<std::path::PathBuf>> {
        if uri.starts_with("data:") { return Ok(None); }
        if let Some(path) = uri.strip_prefix("file://").or_else(|| uri.strip_prefix("file:")) {
            return Ok(Some(path.into()));
        }
        ensure!(!uri.contains(':'), "unsupported glTF URI scheme: {uri}; use a local file or data URI");
        Ok(Some(base.join(urlencoding::decode(uri).context("correct the percent-encoded URI")?.as_ref())))
    }
    let mut errors = Vec::new();
    let mut files = Vec::new();
    for buffer in gltf.buffers() {
        let source = buffer.source();
        let result = (|| -> Result<()> {
            let length = match source {
                gltf::buffer::Source::Uri(uri) => match external(uri, base)? {
                    Some(path) => {
                        // Use the importer's error type, retaining its failure category.
                        let length = crate::source_file(&path).map_err(gltf::Error::Io)?
                            .metadata().map_err(gltf::Error::Io)?.len();
                        files.push(path);
                        length
                    }
                    None => return Ok(()), // Embedded data is checked when consumed.
                },
                gltf::buffer::Source::Bin => bin_length.context("missing GLB binary chunk")?,
            };
            let padded = length.checked_add(3).context("buffer size overflow")? / 4 * 4;
            ensure!(padded >= buffer.length() as u64,
                "expected {} bytes but received {length} bytes ({} after glTF padding)", buffer.length(), padded);
            Ok(())
        })();
        if let Err(error) = result {
            let source = match source { gltf::buffer::Source::Uri(uri) => uri, _ => "GLB binary chunk" };
            errors.push(error.context(format!("glTF buffer {} ({source}): restore the buffer file or correct its URI/byteLength", buffer.index())));
        }
    }
    for image in gltf.images() {
        let gltf::image::Source::Uri {uri,..} = image.source() else {continue};
        let result = (|| -> Result<()> {
            let Some(path) = external(uri, base)? else {return Ok(())};
            files.push(path.clone());
            match crate::source_file(&path) {
                Ok(_) => {},
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {},
                Err(error) => return Err(gltf::Error::Io(error).into()),
            }
            Ok(())
        })();
        if let Err(error) = result {
            errors.push(error.context(format!("glTF image {} ({uri}): restore the image file or correct its URI", image.index())));
        }
    }
    vork_scene::v2::diagnostics::joined(errors)?;
    Ok(files)
}

/// The document's buffers, read as `gltf::import_buffers` reads them (it checks each
/// holds its declared length), with its panics on malformed URIs reported as errors.
pub(crate) fn buffers(
    document: &gltf::Document,
    base: Option<&Path>,
    blob: Option<Vec<u8>>,
) -> Result<Vec<gltf::buffer::Data>> {
    crate::failure::contain("the glTF buffer reader", || {
        Ok(gltf::import_buffers(document, base, blob)?)
    })
}

/// The bytes of a bufferView, which `check_document` bounds by its buffer.
pub(crate) fn view_bytes<'a>(
    buffers: &'a [gltf::buffer::Data],
    view: &gltf::buffer::View<'_>,
) -> Result<&'a [u8]> {
    buffers
        .get(view.buffer().index())
        .and_then(|data| data.get(view.offset()..view.offset().checked_add(view.length())?))
        .with_context(|| format!("glTF bufferView {} runs past its buffer", view.index()))
}

/// Checks the GLB header and chunk lengths against the bytes. gltf 1.4.1 subtracts the 12-byte
/// header from the declared length unchecked and allocates the declared length before reading.
fn check_glb(bytes: &[u8]) -> Result<()> {
    let word = |at: usize| {
        bytes
            .get(at..at + 4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) as usize)
    };
    let length = word(8).context("GLB header is truncated")?;
    // The declared length is the whole file; gltf's slice parser reads chunks to the end of the bytes.
    ensure!(
        length >= 12 && length == bytes.len(),
        "GLB header length {length} does not match the {}-byte file",
        bytes.len()
    );
    let mut at = 12;
    while at < length {
        let chunk = word(at)
            .filter(|_| length - at >= 8)
            .with_context(|| format!("GLB chunk header at byte {at} is truncated"))?;
        ensure!(
            chunk <= length - at - 8,
            "GLB chunk at byte {at} has length {chunk}, past the declared length {length}"
        );
        at += 8 + chunk;
    }
    Ok(())
}

/// Blender can emit texCoord=-1 for a missing UV-map selection. Omission has
/// the glTF default TEXCOORD_0 meaning. Keep every other validation error intact.
pub(crate) fn from_slice(bytes: &[u8]) -> Result<gltf::Gltf> {
    let text = vork_scene::diagnostics::without_bom(bytes);
    ensure!(!bytes.starts_with(b"Kaydara FBX Binary")
        && !text.trim_ascii_start().starts_with(b"; FBX"),
        "unsupported asset format FBX; supported formats are glTF/GLB, SPZ and Gaussian PLY");
    let glb = bytes.starts_with(b"glTF");
    let bytes = if glb { bytes } else { vork_scene::diagnostics::without_bom(bytes) };
    // gltf's parser detects GLB again; a BOM must not route binary past check_glb.
    ensure!(glb || !bytes.starts_with(b"glTF"), "a UTF-8 byte-order mark precedes binary glTF");
    if glb {
        check_glb(bytes)?;
    }
    let original = match gltf::Gltf::from_slice(bytes) {
        Ok(gltf) => return check_document(gltf),
        Err(error) => error,
    };
    let (json, blob) = if glb {
        let glb = gltf::binary::Glb::from_slice(bytes)?;
        (glb.json.into_owned(), glb.bin.map(|b| b.into_owned()))
    } else {
        (bytes.to_vec(), None)
    };
    let mut json: serde_json::Value = serde_json::from_slice(&json)?;
    fn absent_uv(value: &mut serde_json::Value) -> bool {
        let mut changed = false;
        match value {
            serde_json::Value::Object(object) => {
                if object.get("texCoord").and_then(|v| v.as_i64()) == Some(-1) {
                    object.remove("texCoord");
                    changed = true;
                }
                for child in object.values_mut() {
                    changed |= absent_uv(child);
                }
            }
            serde_json::Value::Array(array) => {
                for child in array {
                    changed |= absent_uv(child);
                }
            }
            _ => {}
        }
        changed
    }
    // gltf-json only accepts its own required-extension list; drop the material
    // extensions vork maps onto OpenPBR (materials.rs) and keep them in use.
    let mut implemented = false;
    if let Some(required) = json
        .get_mut("extensionsRequired")
        .and_then(|v| v.as_array_mut())
    {
        let before = required.len();
        required.retain(|e| {
            !e.as_str()
                .is_some_and(|e| crate::materials::GLTF_OPENPBR_EXTENSIONS.contains(&e))
        });
        implemented = required.len() != before;
    }
    if !(absent_uv(&mut json) | implemented) {
        return Err(original.into());
    }
    let mut gltf = gltf::Gltf::from_slice(&serde_json::to_vec(&json)?)?;
    gltf.blob = blob;
    check_document(gltf)
}

/// gltf-json validates indices and byteStride's range but none of the following:
/// image sources (`Image::source` unwraps a missing uri or mimeType), accessor
/// types per use (gltf's readers match only the types glTF allows, else
/// `unreachable!`), byteStride against the element, and views and accessors
/// against the bytes they address (refused rather than let gltf's readers drop
/// truncated data).
fn check_document(gltf: gltf::Gltf) -> Result<gltf::Gltf> {
    // Validate every component, including nodes outside the selected scene.
    // An explicit DFS stack also keeps validation safe for deep hierarchies.
    let mut state = vec![0u8; gltf.nodes().len()];
    for root in gltf.nodes() {
        let mut stack = vec![(root, false)];
        while let Some((node, leaving)) = stack.pop() {
            let index = node.index();
            if leaving {
                state[index] = 2;
                continue;
            }
            ensure!(state[index] != 1, "glTF node hierarchy contains a cycle at node {index}");
            if state[index] == 2 { continue; }
            state[index] = 1;
            stack.push((node.clone(), true));
            stack.extend(node.children().map(|child| (child, false)));
        }
    }
    check_accessor_types(&gltf.document)?;
    for (index, image) in gltf.document.as_json().images.iter().enumerate() {
        match (&image.buffer_view, &image.uri) {
            (Some(_), _) => ensure!(
                image.mime_type.is_some(),
                "glTF image {index} is stored in a bufferView without the required mimeType"
            ),
            (None, Some(_)) => {}
            (None, None) => bail!("glTF image {index} has neither a uri nor a bufferView"),
        }
    }
    for view in gltf.views() {
        let buffer = view.buffer();
        ensure!(
            view.offset().checked_add(view.length()).is_some_and(|end| end <= buffer.length()),
            "glTF bufferView {} ({} bytes at offset {}) runs past buffer {} ({} bytes)",
            view.index(),
            view.length(),
            view.offset(),
            buffer.index(),
            buffer.length()
        );
    }
    // Elements an accessor (or a sparse index/value list) reads from its view.
    let within = |what: String, view: gltf::buffer::View, offset: usize, count: usize, size: usize| {
        let stride = view.stride().unwrap_or(size);
        ensure!(stride >= size, "glTF {what}: byteStride {stride} is smaller than its {size}-byte element");
        ensure!(count > 0, "glTF {what} has no elements");
        let end = (count - 1)
            .checked_mul(stride)
            .and_then(|start| start.checked_add(offset))
            .and_then(|start| start.checked_add(size));
        ensure!(
            end.is_some_and(|end| end <= view.length()),
            "glTF {what}: {count} elements of {size} bytes at offset {offset}, stride {stride}, run past bufferView {} ({} bytes)",
            view.index(),
            view.length()
        );
        Ok(())
    };
    for accessor in gltf.accessors() {
        let index = accessor.index();
        if let Some(sparse) = accessor.sparse() {
            let (indices, values) = (sparse.indices(), sparse.values());
            ensure!(
                indices.view().stride().is_none() && values.view().stride().is_none(),
                "glTF accessor {index} sparse views must not declare byteStride"
            );
            within(format!("accessor {index} sparse indices"), indices.view(), indices.offset(), sparse.count(), indices.index_type().size())?;
            within(format!("accessor {index} sparse values"), values.view(), values.offset(), sparse.count(), accessor.size())?;
        }
        if let Some(view) = accessor.view() {
            within(format!("accessor {index}"), view, accessor.offset(), accessor.count(), accessor.size())?;
        }
    }
    Ok(gltf)
}

/// The accessor types glTF 2.0 allows for each use; gltf's readers assume them.
fn check_accessor_types(document: &gltf::Document) -> Result<()> {
    use gltf::accessor::{DataType::*, Dimensions::*};
    use gltf::{Semantic, animation::Property};
    let check = |what: &str,
                 accessor: gltf::Accessor,
                 dimensions: &[gltf::accessor::Dimensions],
                 types: &[gltf::accessor::DataType]| {
        ensure!(
            dimensions.contains(&accessor.dimensions()) && types.contains(&accessor.data_type()),
            "glTF {what}: accessor {} is {:?} {:?}; glTF allows {dimensions:?} of {types:?}",
            accessor.index(),
            accessor.dimensions(),
            accessor.data_type()
        );
        Ok(())
    };
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let what = format!("mesh {} primitive {}", mesh.index(), primitive.index());
            if let Some(indices) = primitive.indices() {
                check(&format!("{what} indices"), indices, &[Scalar], &[U8, U16, U32])?;
            }
            for (semantic, accessor) in primitive.attributes() {
                let (dimensions, types): (&[_], &[_]) = match semantic {
                    Semantic::Positions | Semantic::Normals => (&[Vec3], &[F32]),
                    Semantic::Tangents => (&[Vec4], &[F32]),
                    Semantic::Colors(_) => (&[Vec3, Vec4], &[U8, U16, F32]),
                    Semantic::TexCoords(_) => (&[Vec2], &[U8, U16, F32]),
                    Semantic::Joints(_) => (&[Vec4], &[U8, U16]),
                    Semantic::Weights(_) => (&[Vec4], &[U8, U16, F32]),
                    // Application attributes: install checks the types it reads.
                    _ => continue,
                };
                check(&format!("{what} {semantic:?}"), accessor, dimensions, types)?;
            }
            for (target, morph) in primitive.morph_targets().enumerate() {
                for accessor in [morph.positions(), morph.normals(), morph.tangents()].into_iter().flatten() {
                    check(&format!("{what} morph target {target}"), accessor, &[Vec3], &[F32])?;
                }
            }
        }
    }
    for animation in document.animations() {
        for channel in animation.channels() {
            let what = format!("animation {} channel {}", animation.index(), channel.index());
            let sampler = channel.sampler();
            check(&format!("{what} input"), sampler.input(), &[Scalar], &[F32])?;
            let (dimensions, types): (&[_], &[_]) = match channel.target().property() {
                Property::Translation | Property::Scale => (&[Vec3], &[F32]),
                Property::Rotation => (&[Vec4], &[I8, U8, I16, U16, F32]),
                Property::MorphTargetWeights => (&[Scalar], &[I8, U8, I16, U16, F32]),
            };
            check(&format!("{what} output"), sampler.output(), dimensions, types)?;
        }
    }
    for skin in document.skins() {
        if let Some(matrices) = skin.inverse_bind_matrices() {
            check(&format!("skin {} inverse bind matrices", skin.index()), matrices, &[Mat4], &[F32])?;
        }
    }
    Ok(())
}

use super::*;
use crate::install::Draw;

pub(crate) struct GeometryUpload<'a> {
    pub(crate) attribute_names: &'a [String],
    pub(crate) evaluated_layout: bool,
    pub(crate) stream_layout: bool,
    /// Per material: alpha-tested in a ray scene (empty without ray techniques).
    pub(crate) asset: Option<usize>,
    pub(crate) animated: bool,
    pub(crate) r: &'a Renderer,
    /// The scene traces rays, so geometry buffers are also BLAS inputs.
    pub(crate) rays: bool,
    pub(crate) buffers: &'a [gltf::buffer::Data],
    pub(crate) draws: &'a mut Vec<Draw>,
    pub(crate) object: u32,
    pub(crate) material_map: &'a [usize],
    pub(crate) materials: &'a [materials::PendingMaterial],
    pub(crate) default: usize,
    pub(crate) pages: &'a mut Vec<crate::geometry::GeometryPage>,
    pub(crate) placeholder: &'a wgpu::Buffer,
    pub(crate) min: &'a mut Vec3,
    pub(crate) max: &'a mut Vec3,
    pub(crate) meshes: std::collections::HashMap<usize, std::ops::Range<usize>>,
    pub(crate) path: Vec<(usize, String)>,
}
/// Packed geometry is a storage binding for camera and ray-hit reconstruction,
/// even when the same allocation also feeds vertex/index fetch.
fn admit_geometry_streams(limits: &wgpu::Limits, bytes: [u64; 3]) -> Result<u64> {
    let limit = limits.max_buffer_size.min(u64::from(limits.max_storage_buffer_binding_size));
    for (name, demand) in ["geometry vertices", "geometry attributes", "geometry indices"].into_iter().zip(bytes) {
        crate::gpu_memory::storage_binding(limits, name, demand, "reduce the geometry buffer size")?;
    }
    Ok(limit)
}
impl GeometryUpload<'_> {
    pub(crate) fn upload(
        &mut self,
        geometry: &crate::geometry::Geometry,
        deform: Option<Deform>,
        material: usize,
        node: (usize, String),
        local: Mat4,
    ) -> Result<()> {
        // Deformation, subdivision and displacement retain source vertex identity.
        // Rigid animated instances can share the immutable indexed mesh normally.
        // Streams with more corners than vertices already provide index reuse.
        // Do not rehash their complete tuples on every install: large imported
        // meshes commonly have no further duplicates, so that only adds startup.
        let indexed = if deform.is_none() && !self.evaluated_layout && geometry.refinement.is_none()
            && geometry.indices.len() <= geometry.vertices.len() {
            geometry.indexed()
        } else { None };
        let geometry = indexed.as_ref().unwrap_or(geometry);
        let crate::geometry::Geometry {
            vertices,
            indices,
            bounds,
            uv_sets,
            attributes,
            ..
        } = geometry;
        let bounds = *bounds;
        let mut layout = crate::geometry::VertexLayout::for_vertices(vertices, deform.is_some() || self.evaluated_layout);
        if self.stream_layout || (self.evaluated_layout && !geometry.point_centers.is_empty()) { layout.uv = [0,1]; layout.color = true; layout.points = !geometry.point_centers.is_empty(); }
        layout.custom = self.attribute_names.len() as u8;
        if vertices.is_empty() || indices.is_empty() {
            return Ok(());
        }

        let byte_count = |count: usize, stride: u64| (count as u64).checked_mul(stride).context("geometry byte demand overflow");
        let binding_limit = admit_geometry_streams(&self.r.device.limits(), [
            byte_count(vertices.len(), layout.vertex_stride())?,
            byte_count(vertices.len(), layout.attribute_stride())?,
            byte_count(indices.len(), std::mem::size_of::<u32>() as u64)?,
        ])?;
        let (packed_vertices, packed_attributes) =
            layout.pack(vertices, self.attribute_names, attributes);
        let (vertex_buffer, index_buffer, geometry_page, vertex_offset, index_offset) =
            if !layout.deform {
                // Bound upload allocations; a scene can exceed one
                // device buffer even when each primitive fits.
                let limit = binding_limit.min(constants::STATIC_GEOMETRY_PAGE_BYTES) as usize;
                let page = self
                    .pages
                    .iter()
                    .rposition(|page| {
                        page.layout == layout
                            && page.vertices.len() + packed_vertices.len() <= limit
                            && page.attributes.len() + packed_attributes.len() <= limit
                            && (page.indices.len() + indices.len()) * 4 <= limit
                    })
                    .unwrap_or_else(|| {
                        self.pages.push(crate::geometry::GeometryPage::new(layout));
                        self.pages.len() - 1
                    });
                let page_data = &mut self.pages[page];
                let vertex_offset = page_data.vertices.len() as u64;
                let index_offset = (page_data.indices.len() * 4) as u64;
                page_data.vertices.extend_from_slice(&packed_vertices);
                page_data.attributes.extend_from_slice(&packed_attributes);
                page_data.indices.extend_from_slice(indices);
                (
                    self.placeholder.clone(),
                    self.placeholder.clone(),
                    Some(page),
                    vertex_offset,
                    index_offset,
                )
            } else {
                (
                    init(
                        &self.r.device,
                        "deform raster vertices",
                        &packed_vertices,
                        // Evaluated layout is asset-wide: rigid siblings still
                        // build static BLASes from this imported vertex buffer.
                        wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE
                            | crate::rt::geometry_usage(self.rays),
                    ),
                    init(
                        &self.r.device,
                        "indices",
                        bytemuck::cast_slice(&indices),
                        wgpu::BufferUsages::INDEX
                            | wgpu::BufferUsages::STORAGE
                            | crate::rt::geometry_usage(self.rays),
                    ),
                    None,
                    0,
                    0,
                )
            };
        *self.min = self.min.min(bounds.0);
        *self.max = self.max.max(bounds.1);
        self.draws.push(Draw {
            splats: None,
            implicit: None,
            sort_order: 0,
            vertices: vertex_buffer,
            attributes: if !layout.deform {
                self.placeholder.clone()
            } else {
                init(
                    &self.r.device,
                    "immutable vertex attributes",
                    &packed_attributes,
                    // Ray hits read attribute streams in place (rt::hit_record).
                    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
                )
            },
            layout,
            uv_sets: *uv_sets,
            exit_geometry: Some(std::sync::Arc::new(geometry.clone())),
            exit_bvh: None,
            open_edges: None,
            indices: index_buffer,
            refit_indices: None,
            geometry_page,
            vertex_offset,
            index_offset,
            count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
            computed: false,
            ray_corners: None,
            material,
            material_slot: None,
            object: self.object,
            instances: self.object..self.object + 1,
            sort_stride: 1,
            local,
            gltf_node: node,
            gltf_path: self.path.clone(),
            primitive_start: 0,
            generated_bounds: bounds,
            center: (bounds.0 + bounds.1) * 0.5,
            bounds,
            deform,
            displacement: None,
            displacement_leader: None,
        });
        Ok(())
    }

    pub(crate) fn visit(&mut self, node: gltf::Node, parent: Mat4) -> Result<()> {
        self.path
            .push((node.index(), node.name().unwrap_or("").to_owned()));
        let local = Mat4::from_cols_array_2d(&node.transform().matrix());
        let mut transform = parent * local;
        // Keep an exact collapse marker while traversing static imported geometry;
        // multiplying a rank-deficient parent by rotations can round its determinant away from zero.
        if crate::collapsed(parent) || crate::collapsed(local) {
            transform.x_axis = glam::Vec4::ZERO; transform.y_axis = glam::Vec4::ZERO; transform.z_axis = glam::Vec4::ZERO;
        }
        if let Some(mesh) = node.mesh() {
            // Every rigid mesh retains its source object frame, including a
            // single-use mesh. Baking its node transform into vertices changes
            // Object/Normal/Generated coordinates seen by surface programs.
            // Animated nodes retain their deformation binding.
            let shared = !self.animated
                && node.skin().is_none()
                && mesh.primitives().all(|p| p.morph_targets().len() == 0);
            if shared && let Some(range) = self.meshes.get(&mesh.index()).cloned() {
                for i in range {
                    let mut draw = self.draws[i].clone();
                    draw.local = transform;
                    draw.gltf_node = (node.index(), node.name().unwrap_or("").to_owned());
                    draw.gltf_path = self.path.clone();
                    self.draws.push(draw);
                }
                for child in node.children() {
                    self.visit(child, transform)?;
                }
                self.path.pop();
                return Ok(());
            }
            let draw_start = self.draws.len();
            for prim in mesh.primitives() {
                ensure!(
                    prim.mode() == gltf::mesh::Mode::Triangles,
                    "only triangle primitives are supported"
                );

                let reader = prim.reader(|b| Some(&self.buffers[b.index()]));
                let positions: Vec<_> = reader
                    .read_positions()
                    .context("mesh missing POSITION")?
                    .collect();
                let indices: Vec<u32> = reader
                    .read_indices()
                    .map(|x| x.into_u32().collect())
                    .unwrap_or_else(|| (0..positions.len() as u32).collect());
                ensure!(
                    indices.len() % 3 == 0
                        && indices.iter().all(|i| (*i as usize) < positions.len()),
                    "invalid triangle indices"
                );
                let supplied_normals = reader.read_normals();
                let generate_flat = supplied_normals.is_none();
                let normals = supplied_normals.map(|values|values.collect::<Vec<_>>())
                    .unwrap_or_else(||vec![[0.;3];positions.len()]);
                let material = prim.material().index()
                    .map_or(self.default, |i| self.material_map[i]);
                let uv_sources = self.materials[material].uv_sources;
                let mut uv = vec![[[0.; 2]; 2]; positions.len()];
                let mut uv_sets = 0;
                for (set, source) in uv_sources.into_iter().enumerate() {
                    if let Some(values) = reader.read_tex_coords(source) {
                        let values: Vec<_> = values.into_f32().collect();
                        ensure!(
                            values.len() == positions.len(),
                            "mismatched TEXCOORD_{source} length"
                        );
                        for (uv, value) in uv.iter_mut().zip(values) {
                            uv[set as usize] = value;
                        }
                        uv_sets |= 1 << set;
                    }
                }
                ensure!(
                    normals.len() == positions.len(),
                    "mismatched vertex normals"
                );
                let colors = reader
                    .read_colors(0)
                    .map(|c| c.into_rgba_f32().collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![[1.; 4]; positions.len()]);
                ensure!(colors.len() == positions.len(), "mismatched vertex colours");
                let tangents = reader
                    .read_tangents()
                    .map(|v| v.collect::<Vec<_>>())
                    .unwrap_or_else(|| vec![[0.; 4]; positions.len()]);
                ensure!(tangents.len() == positions.len(), "mismatched tangents");
                let mut bound_attributes=std::collections::BTreeMap::new();
                for name in self.attribute_names {
                    let Some((semantic,accessor))=prim.attributes().find(|(s,_)|s.to_string()==*name) else {continue;};
                    let values:Vec<[f32;4]>=match semantic {
                        gltf::Semantic::Colors(set)=>reader.read_colors(set).context("colour attribute")?.into_rgba_f32().collect(),
                        gltf::Semantic::TexCoords(set)=>reader.read_tex_coords(set).context("UV attribute")?.into_f32().map(|v|[v[0],v[1],0.,0.]).collect(),
                        _=>{
                            ensure!(accessor.data_type()==gltf::accessor::DataType::F32,"attribute {name}: expected floating-point data");
                            let buffer=|b:gltf::Buffer|Some(&self.buffers[b.index()].0[..]);
                            match accessor.dimensions() {
                                gltf::accessor::Dimensions::Scalar=>gltf::accessor::Iter::<f32>::new(accessor,buffer).context("scalar attribute")?.map(|v|[v,0.,0.,0.]).collect(),
                                gltf::accessor::Dimensions::Vec2=>gltf::accessor::Iter::<[f32;2]>::new(accessor,buffer).context("float2 attribute")?.map(|v|[v[0],v[1],0.,0.]).collect(),
                                gltf::accessor::Dimensions::Vec3=>gltf::accessor::Iter::<[f32;3]>::new(accessor,buffer).context("float3 attribute")?.map(|v|[v[0],v[1],v[2],0.]).collect(),
                                gltf::accessor::Dimensions::Vec4=>gltf::accessor::Iter::<[f32;4]>::new(accessor,buffer).context("float4 attribute")?.collect(),
                                _=>anyhow::bail!("attribute {name}: only float through float4 are supported"),
                            }
                        }
                    };
                    ensure!(values.len()==positions.len(),"attribute {name}: count differs from POSITION");
                    bound_attributes.insert(name.clone(),values);
                }
                let source_count = positions.len();
                // Keep the packed local stream through deformation-source upload and
                // final transformation. Only omitted normals need corner expansion.
                let source_indices = generate_flat.then(||indices.clone());
                let source_index = |i: usize| source_indices.as_ref().map_or(i, |indices|indices[i] as usize);
                let vertices: Vec<_> = positions.into_iter().zip(normals).zip(uv).zip(colors).zip(tangents)
                    .map(|((((position,normal),uv),color),tangent)| Vertex {
                        POSITION:position,NORMAL:normal,TEXCOORD_0:uv[0],TEXCOORD_1:uv[1],pad:[0.;2],COLOR_0:color,TANGENT:tangent,
                    }).collect();
                let (local_vertices,indices,bound_attributes) = if generate_flat {
                    let mut local=crate::geometry::Geometry::new(vertices,indices,uv_sets);
                    local.attributes=bound_attributes;
                    local.flat_normals()?;
                    (local.vertices,local.indices,local.attributes)
                } else {(vertices,indices,bound_attributes)};
                let deform = if self.animated
                    || node.skin().is_some()
                    || prim.morph_targets().len() > 0
                {
                    let joint_count = node.skin().map_or(1, |s| s.joints().len());
                    ensure!(joint_count > 0, "skin has no joints");
                    let sets = if node.skin().is_some() {
                        prim.attributes()
                            .filter_map(|(s, _)| {
                                if let gltf::Semantic::Joints(i) = s {
                                    Some(i as usize + 1)
                                } else {
                                    None
                                }
                            })
                            .max()
                            .unwrap_or(0)
                    } else {
                        1
                    };
                    ensure!(sets > 0, "skinned mesh requires joint influences");
                    let mut influences = vec![
                        Influence {
                            JOINTS_0: [0; 4],
                            WEIGHTS_0: [0.; 4]
                        };
                        local_vertices.len() * sets
                    ];
                    for set in 0..sets {
                        if node.skin().is_none() {
                            for inf in &mut influences {
                                inf.WEIGHTS_0 = [1., 0., 0., 0.];
                            }
                            break;
                        }
                        let joints: Vec<_> = reader
                            .read_joints(set as u32)
                            .context("missing JOINTS set")?
                            .into_u16()
                            .map(|v| v.map(u32::from))
                            .collect();
                        let weights: Vec<_> = reader
                            .read_weights(set as u32)
                            .context("missing WEIGHTS set")?
                            .into_f32()
                            .collect();
                        ensure!(
                            joints.len() == source_count && weights.len() == source_count,
                            "mismatched skin attributes"
                        );
                        for i in 0..local_vertices.len() {
                            let index=source_index(i);
                            let joints=joints[index];let weights=weights[index];
                            ensure!(
                                joints.iter().all(|j| (*j as usize) < joint_count)
                                    && weights.iter().all(|w| w.is_finite() && *w >= 0.),
                                "invalid skin influence"
                            );
                            influences[i * sets + set] = Influence { JOINTS_0: joints, WEIGHTS_0: weights };
                        }
                    }
                    let mut morphs = Vec::<f32>::new();
                    for (p, n, tangent) in reader.read_morph_targets() {
                        let p: Vec<_> = p
                            .map(|v| v.collect())
                            .unwrap_or_else(|| vec![[0.; 3]; source_count]);
                        let n: Vec<_> = n
                            .map(|v| v.collect())
                            .unwrap_or_else(|| vec![[0.; 3]; source_count]);
                        let tangent: Vec<_> = tangent
                            .map(|v| v.collect())
                            .unwrap_or_else(|| vec![[0.; 3]; source_count]);
                        for i in 0..local_vertices.len() {
                            let index=source_index(i);
                            let (p,n,tangent)=(p[index],n[index],tangent[index]);
                            morphs.extend(p);
                            morphs.extend(n);
                            morphs.extend(tangent);
                        }
                    }
                    if morphs.is_empty() {
                        morphs.push(0.);
                    }
                    Some(crate::geometry::deformation(
                        self.r,
                        self.asset,
                        node.index(),
                        node.skin().map(|skin| skin.index()),
                        &local_vertices,
                        influences,
                        morphs,
                        joint_count,
                        sets,
                        prim.morph_targets().len(),
                    )?)
                } else {
                    None
                };
                let vertex_transform = if shared { Mat4::IDENTITY } else { transform };
                let normal = crate::inverse_or_identity(vertex_transform).transpose();
                let mut bounds = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                let vertex_ids = if self.evaluated_layout {
                    let accessor=(prim.get(&gltf::Semantic::Positions).unwrap().index() as u64)<<32;
                    (0..local_vertices.len()).map(|i|accessor|source_index(i) as u64).collect()
                } else {Vec::new()};
                let mut vertices=local_vertices;
                for vertex in &mut vertices {
                    let p=vertex_transform.transform_point3(Vec3::from(vertex.POSITION));
                    bounds.0=bounds.0.min(p);bounds.1=bounds.1.max(p);
                    vertex.POSITION=p.to_array();
                    vertex.NORMAL=normal.transform_vector3(Vec3::from(vertex.NORMAL)).normalize_or_zero().to_array();
                    vertex.TANGENT=vertex_transform.transform_vector3(Vec3::from_slice(&vertex.TANGENT[..3]))
                        .extend(vertex.TANGENT[3]*vertex_transform.determinant().signum()).to_array();
                }
                let geometry = crate::geometry::Geometry {
                    refinement: None,
                    point_centers: Vec::new(),
                    point_widths: Vec::new(),
                    primitive_ids: (0..(indices.len()/3) as u32).collect(),
                    vertices,
                    vertex_ids,
                    indices,
                    bounds,
                    uv_sets,
                    attributes: bound_attributes,
                };
                self.upload(
                    &geometry,
                    deform,
                    material,
                    (node.index(), node.name().unwrap_or("").to_owned()),
                    if shared { transform } else { Mat4::IDENTITY },
                )?;
                if let Some(draw) = self.draws.last_mut() {
                    draw.material_slot = prim.material().index();
                }
            }
            let generated_bounds = self.draws[draw_start..].iter().fold(
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                |(lo, hi), draw| (lo.min(draw.bounds.0), hi.max(draw.bounds.1)),
            );
            for draw in &mut self.draws[draw_start..] {
                draw.generated_bounds = generated_bounds;
            }
            if shared {
                self.meshes
                    .insert(mesh.index(), draw_start..self.draws.len());
            }
        }
        for child in node.children() {
            self.visit(child, transform)?;
        }
        self.path.pop();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn optional_node_instancing_warns_for_each_affected_node() {
        let collector = crate::warnings::Collector::default();
        let _scope = collector.enter();
        let root = std::env::temp_dir().join(format!("vork-optional-instancing-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let data: Vec<u8> = [-1f32,0.,0., 1.,0.,0., 0.,1.,0., -3.,0.,0., -1.,0.,0., 1.,0.,0., 3.,0.,0.]
            .into_iter().flat_map(f32::to_le_bytes).collect();
        std::fs::write(root.join("mesh.bin"), data).unwrap();
        let asset = serde_json::json!({"asset":{"version":"2.0"},"extensionsUsed":["EXT_mesh_gpu_instancing"],
            "buffers":[{"uri":"mesh.bin","byteLength":84}],
            "bufferViews":[{"buffer":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":48}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[-1,0,0],"max":[1,1,0]},
                {"bufferView":1,"componentType":5126,"count":4,"type":"VEC3"}],
            "meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],
            "nodes":[{"name":"first","mesh":0,"extensions":{"EXT_mesh_gpu_instancing":{"attributes":{"TRANSLATION":1}}}},
                {"name":"second","mesh":0,"translation":[0,2,0],"scale":[2,1,1],"extensions":{"EXT_mesh_gpu_instancing":{"attributes":{"TRANSLATION":1}}}}],
            "scenes":[{"nodes":[0,1]}],"scene":0});
        let path = root.join("scene.gltf");
        std::fs::write(&path, serde_json::to_vec(&asset).unwrap()).unwrap();
        super::open(&path).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        let warnings = collector.snapshot();
        assert_eq!(warnings.len(), 2);
        assert!(warnings.iter().all(|warning| warning.code == "gltf_instancing" && warning.dropped.contains("Export separate mesh nodes")));
    }
    #[test]
    fn cyclic_node_hierarchies_are_errors() {
        for nodes in [
            serde_json::json!([{"children":[0]}]),
            serde_json::json!([{"children":[1]},{"children":[2]},{"children":[0]}]),
            serde_json::json!([{}, {"children":[1]}]),
        ] {
            let bytes = serde_json::to_vec(&serde_json::json!({"asset":{"version":"2.0"},"nodes":nodes})).unwrap();
            let error = super::from_slice(&bytes).unwrap_err().to_string();
            assert!(error.contains("node hierarchy contains a cycle"), "{error}");
        }
        super::from_slice(br#"{"asset":{"version":"2.0"},"nodes":[{"children":[1,2]},{"children":[3]},{"children":[3]},{}]}"#).unwrap();
    }

    #[test]
    fn geometry_admission_respects_storage_before_allocation_limits() {
        let limits = wgpu::Limits {max_buffer_size: 256 << 20,
            max_storage_buffer_binding_size: 128 << 20, ..Default::default()};
        let limit = u64::from(limits.max_storage_buffer_binding_size);
        assert_eq!(super::admit_geometry_streams(&limits, [limit; 3]).unwrap(), limit);
        // The random-scene regression fits a device allocation but cannot be
        // bound for hit reconstruction. Refuse before creating GPU resources.
        for stream in 0..3 {
            let mut bytes = [16; 3]; bytes[stream] = 181_678_560;
            let error = super::admit_geometry_streams(&limits, bytes).unwrap_err().to_string();
            assert!(error.contains(["vertices", "attributes", "indices"][stream])
                && error.contains("181678560") && error.contains("134217728"), "{error}");
        }
        let generous = wgpu::Limits {max_storage_buffer_binding_size: 256 << 20, ..limits.clone()};
        super::admit_geometry_streams(&generous, [181_678_560; 3]).unwrap();
        let small = wgpu::Limits {max_buffer_size: 64 << 20, ..limits};
        assert_eq!(super::admit_geometry_streams(&small, [16; 3]).unwrap(), 64 << 20);
    }
    use super::*;

    #[test]
    fn unsupported_asset_formats_report_the_format_before_parsing() {
        for path in ["character.fbx", "character.FBX", "mesh.obj", "external.usdz"] {
            let error = format!("{:#}", open(Path::new(path)).unwrap_err());
            assert!(error.contains("unsupported asset format"), "{error}");
            assert!(error.contains("glTF/GLB, SPZ and Gaussian PLY"), "{error}");
        }
        for bytes in [b"Kaydara FBX Binary  \0\x1a\0".as_slice(), b"  ; FBX 7.4.0 project file"] {
            let error = format!("{:#}", from_slice(bytes).unwrap_err());
            assert!(error.contains("unsupported asset format FBX"), "{error}");
        }
        assert!(format!("{:#}", from_slice(b"{bad gltf json").unwrap_err()).contains("key must be a string"));
    }

    fn glb(json: &[u8], length: u32) -> Vec<u8> {
        let mut bytes = b"glTF".to_vec();
        bytes.extend(2u32.to_le_bytes());
        bytes.extend(length.to_le_bytes());
        bytes.extend((json.len() as u32).to_le_bytes());
        bytes.extend(b"JSON");
        bytes.extend(json);
        bytes
    }

    /// QA fuzz reproducers. Each panicked in gltf-json's validation, gltf's
    /// unwraps and `unreachable!` type matches, or vork's bufferView slice; each
    /// is now an error naming the problem.
    #[test]
    #[cfg(vork_private_tests)]
    fn malformed_documents_are_errors() {
        let seeds = Path::new(env!("CARGO_MANIFEST_DIR")).join("../vork-fuzz/seeds/gltf");
        for (seed, cause) in [
            ("position-accessor-out-of-range.gltf", "the glTF parser failed on malformed input (upstream panic at gltf-json-1.4.1/src/mesh.rs:151"),
            ("image-without-source.gltf", "glTF image 0 has neither a uri nor a bufferView"),
            ("image-view-without-mime-type.glb", "glTF image 0 is stored in a bufferView without the required mimeType"),
            ("image-view-past-buffer.glb", "glTF bufferView 5 (172 bytes at offset 574) runs past buffer 0 (272 bytes)"),
            ("buffer-uri-not-utf8.gltf", "the glTF buffer reader failed on malformed input (upstream panic at gltf-1.4.1/src/import.rs:50"),
            ("index-accessor-type.gltf", "indices: accessor 2 is Vec2 F32; glTF allows [Scalar] of [U8, U16, U32]"),
            ("texcoord-accessor-type.gltf", "TexCoords(0): accessor 2 is Vec2 U32; glTF allows [Vec2] of [U8, U16, F32]"),
            ("accessor-element-size.glb", "animation 0 channel 0 input: accessor 4 is Vec3 F32; glTF allows [Scalar] of [F32]"),
        ] {
            let error = format!("{:#}", crate::fuzz::gltf(&seeds.join(seed)).expect_err(seed));
            assert!(error.contains(cause), "{seed}: {error}");
        }
    }

    #[test]
    fn short_accessor_stride_is_an_error() {
        let bytes=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":12,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAA"}],"bufferViews":[{"buffer":0,"byteLength":12,"byteStride":4}],"accessors":[{"bufferView":0,"componentType":5126,"count":1,"type":"VEC3","min":[0,0,0],"max":[0,0,0]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}
"#;
        let error=from_slice(bytes).unwrap_err().to_string();
        assert!(error.contains("byteStride 4 is smaller than its 12-byte element"),"{error}");
        let bytes=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":16,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAA=="}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":1},{"buffer":0,"byteOffset":4,"byteLength":12,"byteStride":4}],"accessors":[{"componentType":5126,"count":1,"type":"VEC3","min":[0,0,0],"max":[0,0,0],"sparse":{"count":1,"indices":{"bufferView":0,"componentType":5121},"values":{"bufferView":1}}}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]}
"#;
        let error=from_slice(bytes).unwrap_err().to_string();
        assert!(error.contains("sparse views must not declare byteStride"),"{error}");
    }

    #[test]
    fn malformed_glb_lengths_are_errors() {
        // Fuzz reproducers: a declared length below the header underflows gltf's `length - 12`
        // in Glb::from_reader (version 2) and Glb::from_slice (version 0).
        for bytes in [*b"glTF\x02\0\0\0\0\0\0\0", *b"glTF\0\0\0\0\0\0\0\0"] {
            let error = from_slice(&bytes).unwrap_err().to_string();
            assert!(error.contains("GLB header length 0"), "{error}");
        }
        let json = br#"{"asset":{"version":"2.0"}}     "#;
        let length = 20 + json.len() as u32;
        let error = from_slice(&glb(json, u32::MAX)).unwrap_err().to_string();
        assert!(error.contains("GLB header length"), "{error}");
        let mut chunk_past_end = glb(json, length);
        chunk_past_end[12] += 4;
        let error = from_slice(&chunk_past_end).unwrap_err().to_string();
        assert!(error.contains("GLB chunk at byte 12"), "{error}");
        let error = from_slice(&glb(json, 16)[..16]).unwrap_err().to_string();
        assert!(error.contains("GLB chunk header at byte 12"), "{error}");
        // A chunk appended past the declared length is not read.
        let mut appended = glb(json, length);
        appended.extend(4u32.to_le_bytes());
        appended.extend(b"BIN\0");
        appended.extend([0; 4]);
        let error = from_slice(&appended).unwrap_err().to_string();
        assert!(error.contains("GLB header length"), "{error}");
        assert!(from_slice(&glb(json, length)).unwrap().blob.is_none());
    }

    #[test]
    fn gltf_json_accepts_a_leading_byte_order_mark() {
        let json = br#"{"asset":{"version":"2.0"},"nodes":[{"name":"a"}]}"#;
        let document = from_slice(&[b"\xEF\xBB\xBF".as_slice(), json].concat()).unwrap().document;
        assert_eq!(document.nodes().next().unwrap().name(), Some("a"));
        // Binary glTF has no byte-order mark, valid or with a malformed length.
        let json = br#"{"asset":{"version":"2.0"}}     "#;
        for length in [20 + json.len() as u32, 12] {
            let error = from_slice(&[b"\xEF\xBB\xBF".as_slice(), &glb(json, length)].concat()).unwrap_err().to_string();
            assert!(error.contains("byte-order mark precedes binary glTF"), "{error}");
        }
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;

    fn container(chunks: &[(&[u8;4],&[u8])]) -> Vec<u8> {
        let mut bytes=b"glTF\x02\0\0\0\0\0\0\0".to_vec();
        for &(kind,data) in chunks {
            bytes.extend((data.len() as u32).to_le_bytes());bytes.extend(kind);bytes.extend(data);
        }
        let length=bytes.len() as u32;bytes[8..12].copy_from_slice(&length.to_le_bytes());bytes
    }

    fn admission(bytes: &[u8],base: &Path) -> Result<(serde_json::Value,Option<u64>,Vec<std::path::PathBuf>)> {
        let (gltf,length)=source_document(&mut std::io::Cursor::new(bytes),bytes.len() as u64)?;
        assert!(gltf.blob.is_none(),"admission must not manufacture a BIN payload");
        let files=validate_sources(&gltf.document,length,base)?;
        Ok((serde_json::to_value(gltf.document.as_json())?,length,files))
    }

    fn assert_same_admission(bytes: &[u8],base: &Path) {
        let old=(|| -> Result<_> {
            let gltf=from_slice(bytes)?;let length=gltf.blob.as_ref().map(|b|b.len() as u64);
            let files=validate_sources(&gltf.document,length,base)?;
            Ok((serde_json::to_value(gltf.document.as_json())?,length,files))
        })().map_err(|e|format!("{e:#}"));
        assert_eq!(admission(bytes,base).map_err(|e|format!("{e:#}")),old);
    }

    #[test]
    fn metadata_source_admission_matches_loader_refusals() {
        let base=Path::new(".");
        let json=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":4}]}"#;
        let basic=br#"{"asset":{"version":"2.0"}}"#;
        let valid=container(&[(b"JSON",json),(b"BIN\0",&[1,2,3,4])]);
        assert!(admission(&valid,base).is_ok());
        for bytes in [
            container(&[(b"JSON",basic)]),
            container(&[(b"JSON",basic),(b"BIN\0",&[])]),
            container(&[(b"JSON",json)]), // missing BIN
            container(&[(b"JSON",json),(b"BIN\0",&[])]), // short BIN
            container(&[(b"JSON",json),(b"BIN\0",&[1,2,3])]), // importer padding
            container(&[(b"JSON",json),(b"BIN\0",&[1,2,3,4]),(b"JUNK",&[9])]),
            container(&[(b"BIN\0",basic)]),
            container(&[(b"JUNK",basic)]),
            container(&[(b"JSON",basic),(b"JSON",&[])]),
            container(&[(b"JSON",basic),(b"JUNK",&[])]),
            container(&[]),
            container(&[(b"JSON",b"not JSON")]),
            container(&[(b"JSON",&[b"\xEF\xBB\xBF".as_slice(),basic].concat())]),
            [b"\xEF\xBB\xBF".as_slice(),&valid].concat(),
            [b"\xEF\xBB\xBF".as_slice(),basic].concat(),
            b"Kaydara FBX Binary  \0".to_vec(),b"; FBX 7.4".to_vec(),
        ] {assert_same_admission(&bytes,base);}
        // Exact error precedence, header/chunk truncation and ignored trailing
        // chunk behavior are compared against the unchanged installer parser.
        for end in 0..valid.len() {assert_same_admission(&valid[..end],base);}
        for index in 0..valid.len() {
            for value in [0,255] {let mut bytes=valid.clone();bytes[index]=value;assert_same_admission(&bytes,base);}
        }
        for tail in [vec![1],vec![0;7],vec![255;8]] {
            let mut bytes=valid.clone();bytes.extend(tail);let length=bytes.len() as u32;
            bytes[8..12].copy_from_slice(&length.to_le_bytes());assert_same_admission(&bytes,base);
        }
        // The same normalization and document/range checks still run on JSON.
        for (index,json) in [
            serde_json::json!({"asset":{"version":"2.0"},"materials":[{"normalTexture":{"index":0,"texCoord":-1}}],"textures":[{"source":0}],"images":[{"uri":"missing-normal.png"}]}),
            serde_json::json!({"asset":{"version":"2.0"},"extensionsRequired":["KHR_materials_clearcoat"],"materials":[{"extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":1}}}]}),
            serde_json::json!({"asset":{"version":"2.0"},"nodes":[{"children":[0]}]}),
            serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"byteLength":4}],"bufferViews":[{"buffer":0,"byteLength":8}]}),
            serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"byteLength":4}],"bufferViews":[{"buffer":0,"byteLength":4}],"accessors":[{"bufferView":0,"componentType":5126,"count":2,"type":"SCALAR"}]}),
            serde_json::json!({"asset":{"version":"2.0"},"images":[{}]}),
        ].into_iter().enumerate() {
            let bytes=container(&[(b"JSON",&serde_json::to_vec(&json).unwrap()),(b"BIN\0",&[0;4])]);
            assert_same_admission(&bytes,base);
            assert_eq!(admission(&bytes,base).is_ok(),index<2,"normalization accepts; invalid document ranges refuse");
        }
    }

    #[test]
    fn metadata_source_admission_buffers_extension_chunks() -> Result<()> {
        use std::io::{Read,Seek,SeekFrom};
        struct Counted {bytes:std::io::Cursor<Vec<u8>>,reads:usize,seeks:usize,read_bytes:usize}
        impl Read for Counted {
            fn read(&mut self,out:&mut [u8])->std::io::Result<usize> {
                self.reads+=1;
                let count=self.bytes.read(out)?;self.read_bytes+=count;Ok(count)
            }
        }
        impl Seek for Counted {
            fn seek(&mut self,pos:SeekFrom)->std::io::Result<u64> {
                self.seeks+=1;self.bytes.seek(pos)
            }
        }
        // Zero and nonzero small payloads exercise buffered header/skip paths,
        // including headers straddling buffer boundaries. Large payloads still
        // seek rather than reading the whole extension into memory.
        for size in [0,4,13,16384] {
            let json=br#"{"asset":{"version":"2.0"}}"#;
            let payload=vec![0;size];
            let count=4096;
            let mut chunks=vec![(b"JSON",json.as_slice()),(b"BIN\0",&[][..])];
            chunks.extend(std::iter::repeat_n((b"TEST",payload.as_slice()),count));
            let bytes=container(&chunks);let length=bytes.len() as u64;
            let mut reader=Counted {bytes:std::io::Cursor::new(bytes),reads:0,seeks:0,read_bytes:0};
            let (_,bin_length)=source_document(&mut reader,length)?;
            assert_eq!(bin_length,Some(0));
            if size<8192 {
                assert!(reader.reads<count/100,"{} reads for {count} small chunks",reader.reads);
                assert!(reader.seeks<20,"{} seeks for {count} small chunks",reader.seeks);
            } else {
                assert!(reader.read_bytes<(length as usize)/2,"large extension payloads must be skipped");
            }
            // Even an ignored chunk after many buffer refills must be checked.
            let bytes=reader.bytes.get_mut();let end=bytes.len();
            bytes[end-size-8..end-size-4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert_same_admission(bytes,Path::new("."));
        }
        Ok(())
    }

    #[test]
    fn metadata_source_admission_skips_binary_payload() -> Result<()> {
        use std::io::{Read,Seek,SeekFrom};
        struct Guard {bytes:std::io::Cursor<Vec<u8>>,forbidden:std::ops::Range<u64>}
        impl Read for Guard {
            fn read(&mut self,out:&mut [u8])->std::io::Result<usize> {
                let at=self.bytes.position();let end=(at+out.len() as u64).min(self.bytes.get_ref().len() as u64);
                assert!(end<=self.forbidden.start || at>=self.forbidden.end,"admission read BIN bytes");
                self.bytes.read(out)
            }
        }
        impl Seek for Guard {fn seek(&mut self,pos:SeekFrom)->std::io::Result<u64>{self.bytes.seek(pos)}}
        let json=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":32768}]}"#;
        let bytes=container(&[(b"JSON",json),(b"BIN\0",&vec![0;32768]),(b"JUNK",&[1,2,3])]);
        let length=bytes.len() as u64;let bin_start=28+json.len() as u64;
        let mut reader=Guard {bytes:std::io::Cursor::new(bytes),forbidden:bin_start..bin_start+32768};
        let (gltf,bin_length)=source_document(&mut reader,length)?;
        assert_eq!(bin_length,Some(32768));assert!(gltf.blob.is_none());
        validate_sources(&gltf.document,bin_length,Path::new("."))?;
        Ok(())
    }

    #[test]
    fn metadata_source_admission_validates_external_files() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-gltf-metadata-source-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;std::fs::write(root.join("short.bin"),[1,2,3])?;
        std::fs::create_dir_all(root.join("directory.png"))?;
        for (uri,length) in [("short.bin",4),("short.bin",8),("missing.bin",4),("https://example.invalid/data",4)] {
            let json=serde_json::to_vec(&serde_json::json!({"asset":{"version":"2.0"},"buffers":[{"uri":uri,"byteLength":length}]}))?;
            assert_same_admission(&container(&[(b"JSON",&json)]),&root);
        }
        for image in ["missing.png","directory.png","https://example.invalid/image"] {
            let json=serde_json::to_vec(&serde_json::json!({"asset":{"version":"2.0"},"images":[{"uri":image}]}))?;
            assert_same_admission(&container(&[(b"JSON",&json)]),&root);
        }
        let json=br#"{"asset":{"version":"2.0"},"buffers":[{"byteLength":4}]}"#;
        let path=root.join("valid.glb");std::fs::write(&path,container(&[(b"JSON",json),(b"BIN\0",&[0;4])]))?;
        assert!(validate_file(&path,&root)?.is_empty());
        std::fs::remove_dir_all(root)?;Ok(())
    }

    #[test]
    fn source_admission_matches_import_uri_and_padding_rules() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-gltf-source-rules-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("three bytes.bin"), [1, 2, 3])?;
        let document = from_slice(&serde_json::to_vec(&serde_json::json!({
            "asset":{"version":"2.0"},
            "buffers":[
                {"uri":"three%20bytes.bin","byteLength":4},
                {"uri":format!("file:{}",root.join("three bytes.bin").display()),"byteLength":4},
                {"uri":"data:application/octet-stream;base64,AQID","byteLength":4}
            ],
            "images":[{"uri":"missing-sampled-image.png"}]
        }))?)?;
        assert_eq!(validate_sources(&document.document, document.blob.as_ref().map(|b|b.len() as u64), &root)?, vec![
            root.join("three bytes.bin"), root.join("three bytes.bin"), root.join("missing-sampled-image.png")]);
        // Installation accepts these exact external paths and pads 3 bytes to 4.
        let payloads = buffers(&document.document, Some(&root), None)?;
        assert!(payloads.iter().all(|data| data.len() == 4));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
