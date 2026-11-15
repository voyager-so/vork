//! Install inline geometry and retain contiguous instance records.
use super::*;
use crate::install::{Draw, EvalNode, MeshRoot};
use serde_json::json;

pub(crate) fn admit_triangle_instances(triangles: u64, instances: u64, total: &mut u64) -> Result<()> {
    let limit = constants::MAX_INSTALL_TRIANGLE_INSTANCES;
    let expanded = triangles.checked_mul(instances).context("expanded triangle instance count overflow")?;
    *total = total.checked_add(expanded).context("expanded triangle instance count overflow")?;
    ensure!(*total <= limit,
        "scene expands to {total} triangle instances, exceeding the install-time work limit of {limit}; reduce geometry density or instance count");
    Ok(())
}

/// Check point expansion before allocating its sphere meshes or installing GPU work.
pub(crate) fn admit_points(scene: &Scene, attributes: &SceneBuffers, sides: &std::collections::HashMap<String, usize>) -> Result<()> {
    fn visit(scene: &Scene, nodes: &[Node], attributes: &SceneBuffers, sides: &std::collections::HashMap<String, usize>, total: &mut u64) -> Result<()> {
        for node in nodes {
            if let NodeKind::Mesh { geometry, instances, .. } = &node.kind {
                let name = geometry;
                let geometry = &scene.geometry[geometry];
                if matches!(geometry.topology, vork_scene::v2::Topology::Points) {
                    let points = attributes.data[&geometry.attributes["POSITION"]].count() as u64;
                    let instances = instances.as_ref().map(|i| instance_count(i, attributes)).transpose()?.unwrap_or(1) as u64;
                    admit_triangle_instances(points.checked_mul(crate::points::triangles_per_point(sides[name])?).context("point triangle count overflow")?, instances, total)
                        .with_context(|| format!("node {:?}", node.name))?;
                }
            }
            visit(scene, &node.children, attributes, sides, total)?;
        }
        Ok(())
    }
    visit(scene, &scene.nodes, attributes, sides, &mut 0)
}

pub(crate) fn instance_count(definition: &vork_scene::Instances, buffers: &SceneBuffers) -> Result<usize> {
    let name = definition.attributes.values().next().context("instances.attributes must contain at least one column")?;
    let count = buffers.data.get(name).context("unknown instance buffer")?.count();
    for name in definition.attributes.values() {
        ensure!(buffers.data.get(name).context("unknown instance buffer")?.count() == count,
            "instance attribute columns must have equal counts");
    }
    Ok(count)
}

pub(crate) fn instance_transforms(definition: &vork_scene::Instances, buffers: &SceneBuffers) -> Result<Vec<Mat4>> {
    let count = instance_count(definition, buffers)?;
    let column = |name: &str| definition.attributes.get(name).map(|name| &buffers.data[name]);
    let translations = column("TRANSLATION").map(|b| b.floats::<3>()).transpose()?;
    let rotations = column("ROTATION").map(|b| b.floats::<4>()).transpose()?;
    let scales = column("SCALE").map(|b| b.floats::<3>()).transpose()?;
    (0..count).map(|i| {
        let translation = translations.as_ref().map_or(Vec3::ZERO, |v| Vec3::from_array(v[i]));
        let rotation = rotations.as_ref().map_or(glam::Quat::IDENTITY, |v| glam::Quat::from_array(v[i]));
        let scale = scales.as_ref().map_or(Vec3::ONE, |v| Vec3::from_array(v[i]));
        ensure!(rotation.is_finite() && (rotation.length_squared()-1.).abs() <= 1e-4,
            "instance ROTATION must be a unit XYZW quaternion");
        let local = Mat4::from_scale_rotation_translation(scale, rotation.normalize(), translation);
        ensure!(translation.is_finite() && scale.is_finite() && local.is_finite(),
            "instance TRS must produce a finite affine matrix");
        Ok(local)
    }).collect()
}

pub(crate) fn instances(
    nodes: &[EvalNode],
    roots: &mut Vec<MeshRoot>,
    objects: &mut Vec<Object>,
    draws: &mut Vec<Draw>,
    attributes: &SceneBuffers,
) -> Result<()> {
    // Preflight before deforming draws or object records expand in host memory.
    let mut total = 0;
    for draw in draws.iter() {
        let node = &nodes[roots[draw.object as usize].node].node;
        let instances = match &node.kind {
            NodeKind::Mesh { instances: Some(instances), .. } => instance_count(instances, attributes)? as u64,
            _ => 1,
        };
        admit_triangle_instances(u64::from(draw.count / 3), instances, &mut total)?;
    }
    let old_roots = std::mem::take(roots);
    let mut grouped: Vec<Vec<Draw>> = (0..old_roots.len()).map(|_| Vec::new()).collect();
    for draw in std::mem::take(draws) {
        grouped[draw.object as usize].push(draw);
    }
    let old_objects = std::mem::take(objects);
    let mut sort_order = 0;
    for ((root, source_draws), old_object) in old_roots.into_iter().zip(grouped).zip(old_objects) {
        let node = &nodes[root.node];
        let definition = match &node.node.kind {
            NodeKind::Mesh { instances, .. } => instances.as_ref(),
            _ => None,
        };
        let Some(definition) = definition else {
            let id = objects.len() as u32;
            objects.push(old_object);
            roots.push(root);
            draws.extend(source_draws.into_iter().map(|mut d| {
                d.object = id;
                d.instances = id..id + 1;
                d.sort_order = sort_order;
                sort_order += 1;
                d
            }));
            continue;
        };
        let data = instance_transforms(definition, attributes)?;
        let offsets = definition
            .attributes
            .get("time_offset")
            .map(|name| -> Result<_> {
                let data = &attributes.data[name];
                ensure!(
                    matches!(data.element_type, ::vork_scene::v2::BufferType::F32),
                    "time_offset requires f32"
                );
                ensure!(
                    data.count() == instance_count(definition, attributes)?,
                    "time_offset count differs from instance columns"
                );
                data.floats::<1>()
            })
            .transpose()?;
        let first = objects.len() as u32;
        for (i, &local) in data.iter().enumerate() {
            let offset = offsets.as_ref().map_or(0., |v| v[i][0]);
            let mut values = Object::new(node.world * local);
            values.material_index = old_object.material_index;
            objects.push(values);
            roots.push(MeshRoot {
                node: root.node,
                local,
                time_offset: offset,
                instance_index: Some(i as u32),
                imported_visibility: constants::RAY_VISIBILITY_ALL,
                imported_pose: None,
            });
        }
        let stride = source_draws.len();
        for (i, mut draw) in source_draws.into_iter().enumerate() {
            draw.object = first;
            draw.instances = first..objects.len() as u32;
            draw.sort_order = sort_order + i;
            draw.sort_stride = stride;
            if draw.layout.deform {
                for instance in 0..data.len() {
                    let mut instance_draw = draw.clone();
                    instance_draw.object = first + instance as u32;
                    instance_draw.instances = instance_draw.object..instance_draw.object + 1;
                    instance_draw.sort_order = sort_order + instance * stride + i;
                    instance_draw.sort_stride = 1;
                    draws.push(instance_draw);
                }
            } else if !draw.instances.is_empty() {
                draws.push(draw);
            }
        }
        sort_order += data.len() * stride;
    }
    Ok(())
}

/// Every scene buffer decoded once at install; geometry, instances, skins and
/// tracks read from it, and it is dropped once install finishes.
pub(crate) struct SceneBuffers {
    pub(crate) data: std::collections::BTreeMap<String, AttributeData>,
    pub(crate) grids: std::collections::BTreeMap<String, crate::vdb::Grid>,
}
pub(crate) struct AttributeData {
    pub(crate) element_type: ::vork_scene::v2::BufferType,
    pub(crate) words: Vec<u32>,
    pub(crate) width: usize,
}
impl AttributeData {
    pub(crate) fn file(buffer: &::vork_scene::v2::Buffer, base: &Path) -> Result<(std::fs::File, u64)> {
        let path = base.join(
            buffer
                .file
                .as_ref()
                .context("buffer requires data or file")?,
        );
        let file = crate::source_file(&path)
            .with_context(|| format!("open buffer {}", path.display()))?;
        let available = file
            .metadata()?
            .len()
            .checked_sub(buffer.offset)
            .context("buffer offset exceeds file")?;
        let stride = buffer.element_type.width()
            .checked_mul(std::mem::size_of::<u32>())
            .context("buffer stride overflow")?;
        let length = if let Some(count) = buffer.len() {
            u64::try_from(count)?
                .checked_mul(stride as u64)
                .context("buffer size overflow")?
        } else {
            ensure!(
                available % stride as u64 == 0,
                "buffer file has a partial element"
            );
            available
        };
        ensure!(length <= available, "buffer file is too short");
        ensure!(length > 0, "buffer is empty");
        Ok((file, length))
    }

    pub(crate) fn decode(buffer: &::vork_scene::v2::Buffer, base: &Path) -> Result<Self> {
        use ::vork_scene::v2::BufferType;
        ensure!(buffer.grid.is_none(), "grid buffers cannot be used as numeric attributes");
        let width = buffer.element_type.width();
        let words = if let Some(data) = &buffer.data {
            let mut words = Vec::with_capacity(data.len() * width);
            for row in data {
                fn flatten<'a>(value: &'a serde_json::Value, out: &mut Vec<&'a serde_json::Value>) {
                    if let Some(values) = value.as_array() {
                        for value in values {
                            flatten(value, out);
                        }
                    } else {
                        out.push(value);
                    }
                }
                let mut values = Vec::new();
                flatten(row, &mut values);
                ensure!(values.len() == width, "attribute element width mismatch");
                for value in values {
                    words.push(match buffer.element_type {
                        BufferType::U32 | BufferType::Uvec4 => {
                            u32::try_from(value.as_u64().context("attribute requires u32")?)?
                        }
                        BufferType::I32 => {
                            i32::try_from(value.as_i64().context("attribute requires i32")?)? as u32
                        }
                        _ => {
                            let v = value.as_f64().context("attribute requires a number")? as f32;
                            ensure!(v.is_finite(), "attribute requires finite f32");
                            v.to_bits()
                        }
                    });
                }
            }
            words
        } else if buffer.dims.is_some() && buffer.file.is_none() {
            vec![0; buffer.len().context("dense grid size overflow")?.checked_mul(width).context("dense grid size overflow")?]
        } else {
            let path = base.join(buffer.file.as_ref().context("buffer requires a file")?);
            use std::io::{Read, Seek, SeekFrom};
            let (mut file, length) = Self::file(buffer, base)?;
            let mut bytes = vec![0; usize::try_from(length)?];
            file.seek(SeekFrom::Start(buffer.offset))?;
            file.read_exact(&mut bytes)
                .with_context(|| format!("read buffer {}", path.display()))?;
            let words: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if !matches!(
                buffer.element_type,
                BufferType::U32 | BufferType::I32 | BufferType::Uvec4
            ) {
                ensure!(
                    words.iter().all(|v| f32::from_bits(*v).is_finite()),
                    "buffer contains nonfinite f32"
                );
            }
            words
        };
        ensure!(!words.is_empty(), "buffer is empty");
        Ok(Self {
            element_type: buffer.element_type.clone(),
            words,
            width,
        })
    }
    pub(crate) fn count(&self) -> usize {
        self.words.len() / self.width
    }
    pub(crate) fn floats<const N: usize>(&self) -> Result<Vec<[f32; N]>> {
        ensure!(self.width == N, "attribute requires {N} components");
        Ok(self
            .words
            .chunks_exact(N)
            .map(|row| std::array::from_fn(|i| f32::from_bits(row[i])))
            .collect())
    }
}
/// Validate install-time vertex and instance streams before material binding.
pub(crate) fn validate_attributes(scene: &::vork_scene::v2::Scene) -> Result<()> {
    fn visit(scene: &Scene, nodes: &[Node]) -> Result<()> {
        for node in nodes {
            if let NodeKind::Mesh { instances: Some(instances), .. } = &node.kind {
                for (name, buffer) in &instances.attributes {
                    if name != "pass_index" {
                        crate::attributes::arity(scene.buffers[buffer].element_type)
                            .with_context(|| format!("node {:?}: instance attribute {name}", node.name))?;
                    }
                }
            }
            visit(scene, &node.children)?;
        }
        Ok(())
    }
    visit(scene, &scene.nodes)
}

impl SceneBuffers {
    pub(crate) fn decode(scene: &::vork_scene::v2::Scene, base: &Path) -> Result<Self> {
        let mut data = std::collections::BTreeMap::new();
        let mut grids = std::collections::BTreeMap::new();
        let mut errors = Vec::new();
        for (name, buffer) in &scene.buffers {
            let result = (|| -> Result<()> {
            if buffer.dims.is_some() && buffer.is_volume_grid() {
                grids.insert(name.clone(), crate::vdb::Grid { words: Vec::new(), bounds: (Vec3::splat(-1.), Vec3::ONE) });
            }
            if buffer.grid.is_some() {
                grids.insert(name.clone(), crate::vdb::Grid::load(buffer, base)
                    .with_context(|| format!("buffer {name}"))?);
                return Ok(());
            }
            data.insert(
                name.clone(),
                AttributeData::decode(buffer, base).with_context(|| format!("buffer {name}"))?,
            );
            Ok(())
            })();
            if let Err(error) = result {
                errors.push(format!("{error:#}; restore the buffer/grid file or correct its data, offset and count"));
            }
        }
        ensure!(errors.is_empty(), "{}", errors.join("\n"));
        for (name, geometry) in &scene.geometry {
            if !crate::computed_geometry::dynamic(scene, geometry) {
                if let Some(ids) = &geometry.ids {
                    let mut seen = std::collections::HashSet::new();
                    for &id in &data[ids].words {
                        ensure!(seen.insert(id), "geometry {name}: duplicate point identity {id} in immutable ids buffer {ids}");
                    }
                }
            }
        }
        Ok(Self { data, grids })
    }
}

/// Install standard vertex channels directly from the v2 geometry table.
pub(crate) fn geometry_v2(
    attributes: &SceneBuffers,
    source: &::vork_scene::v2::Geometry,
    names: &[String],
    curve_sides: usize,
    cache: Option<&crate::disk_cache::Store>,
) -> Result<crate::geometry::Geometry> {
    use ::vork_scene::v2::BufferType;
    if matches!(source.topology, ::vork_scene::v2::Topology::Implicit) {
        // Installation bounds only. Never submitted as visible triangles.
        return Ok(crate::vdb::bounds_geometry((Vec3::splat(-1.),Vec3::ONE)));
    }
    if matches!(source.topology, ::vork_scene::v2::Topology::Volume) {
        let mut bounds = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for name in source.attributes.values() {
            let grid = attributes.grids.get(name).context("volume field requires a grid buffer")?;
            bounds = (bounds.0.min(grid.bounds.0), bounds.1.max(grid.bounds.1));
        }
        ensure!(bounds.0.is_finite() && bounds.1.is_finite(), "volume requires grid fields");
        return Ok(crate::vdb::bounds_geometry(bounds));
    }
    let get = |name: &str| -> Result<&AttributeData> {
        let buffer = source
            .attributes
            .get(name)
            .with_context(|| format!("missing geometry attribute {name}"))?;
        attributes
            .data
            .get(buffer)
            .with_context(|| format!("unknown attribute buffer {buffer}"))
    };
    let positions = get("POSITION")?;
    ensure!(
        matches!(positions.element_type, BufferType::Vec3),
        "POSITION requires vec3"
    );
    let positions = positions.floats::<3>()?;
    if matches!(source.topology, ::vork_scene::v2::Topology::Points) {
        let widths = attributes.data.get(source.widths.as_ref().context("points require widths")?)
            .context("unknown point widths buffer")?;
        ensure!(widths.element_type == BufferType::F32, "point widths require f32");
        let widths: Vec<_> = widths.floats::<1>()?.into_iter().map(|v| v[0]).collect();
        let mut geometry = crate::points::cached_spheres(&positions, &widths, curve_sides, cache)?;
        geometry.point_centers = positions.clone();
        geometry.point_widths = widths;
        let point_vertices = geometry.vertices.len() / positions.len();
        geometry.primitive_ids=geometry.indices.chunks_exact(3).map(|tri|tri[0]/point_vertices as u32).collect();
        apply_point_attributes(attributes, source, names, positions.len(), point_vertices, &mut geometry)?;
        return Ok(geometry);
    }
    if matches!(source.topology, ::vork_scene::v2::Topology::Curves) {
        let counts = &attributes.data[source.curve_vertex_counts.as_ref().context("curves require curve_vertex_counts")?];
        let widths = &attributes.data[source.widths.as_ref().context("curves require widths")?];
        ensure!(counts.element_type == BufferType::U32 && widths.element_type == BufferType::F32,
            "curves require u32 counts and f32 widths");
        let widths: Vec<_> = widths.floats::<1>()?.into_iter().map(|v| v[0]).collect();
        let mut geometry = crate::curves::cached_tubes(&positions, &widths, &counts.words, curve_sides, cache)?;
        geometry.primitive_ids=geometry.indices.chunks_exact(3).map(|tri|geometry.vertices[tri[0] as usize].TEXCOORD_1[0] as u32).collect();
        apply_point_attributes(attributes, source, names, positions.len(), curve_sides + 1, &mut geometry)?;
        return Ok(geometry);
    }
    let indices = if let Some(name) = &source.indices {
        let data = attributes
            .data
            .get(name)
            .context("unknown geometry index buffer")?;
        ensure!(
            matches!(data.element_type, BufferType::U32),
            "indices require u32"
        );
        data.words.clone()
    } else {
        (0..u32::try_from(positions.len())?).collect()
    };
    let indices = if let Some(sizes) = &source.face_vertex_counts {
        crate::polygon::triangulate(&positions, &indices, sizes)?
    } else {
        indices
    };
    ensure!(
        indices.len() % 3 == 0 && indices.iter().all(|i| (*i as usize) < positions.len()),
        "indices require in-range triangle triples"
    );
    for (name, buffer) in &source.attributes {
        ensure!(
            attributes.data[buffer].count() == positions.len(),
            "attribute {name}: count must match vertices"
        );
    }
    let normals = if source.attributes.contains_key("NORMAL") {
        let data = get("NORMAL")?;
        ensure!(
            matches!(data.element_type, BufferType::Vec3),
            "normal requires vec3"
        );
        data.floats::<3>()?
    } else {
        vec![[0.;3];positions.len()]
    };
    let mut uvs = vec![[[0.; 2]; 2]; positions.len()];
    let mut uv_sets = 0;
    for (set, name) in ["TEXCOORD_0", "TEXCOORD_1"].into_iter().enumerate() {
        if source.attributes.contains_key(name) {
            let data = get(name)?;
            ensure!(
                matches!(data.element_type, BufferType::Vec2),
                "{name} requires vec2"
            );
            for (uv, value) in uvs.iter_mut().zip(data.floats::<2>()?) {
                uv[set] = value;
            }
            uv_sets |= 1 << set;
        }
    }
    let mut geometry =
        crate::geometry::Geometry::channels(positions, normals, uvs, indices, uv_sets);
    if let Some(sizes)=&source.face_vertex_counts {
        geometry.primitive_ids=sizes.iter().enumerate().flat_map(|(face,n)|std::iter::repeat_n(face as u32,(*n-2) as usize)).collect();
    }
    if source.attributes.contains_key("TANGENT") {
        for (vertex, value) in geometry
            .vertices
            .iter_mut()
            .zip(get("TANGENT")?.floats::<4>()?)
        {
            vertex.TANGENT = value;
        }
    }
    apply_point_attributes(attributes, source, names, geometry.vertices.len(), 1, &mut geometry)?;
    if !source.attributes.contains_key("NORMAL") && source.subdivision.as_ref().is_none_or(|s|s.render_levels==0) {geometry.flat_normals()?;}
    Ok(geometry)
}

pub(crate) fn apply_point_attributes(
    attributes: &SceneBuffers,
    source: &::vork_scene::v2::Geometry,
    names: &[String],
    point_count: usize,
    repeats: usize,
    geometry: &mut crate::geometry::Geometry,
) -> Result<()> {
    use ::vork_scene::v2::BufferType;
    for (name, buffer) in &source.attributes {
        let data = &attributes.data[buffer];
        ensure!(data.count() == point_count, "attribute {name}: count must match points");
        if !names.contains(name) && name != "COLOR_0" {
            continue;
        }
        let width = match data.element_type {
            BufferType::F32 => 1,
            BufferType::Vec2 => 2,
            BufferType::Vec3 => 3,
            BufferType::Vec4 => 4,
            _ => continue,
        };
        let values = data
            .words
            .chunks_exact(width)
            .map(|row| {
                let mut value = [0.; 4];
                if width == 1 {
                    value.fill(f32::from_bits(row[0]));
                    value[3] = 1.;
                } else {
                    for (i, word) in row.iter().enumerate() {
                        value[i] = f32::from_bits(*word);
                    }
                    if width < 4 {
                        value[3] = 1.;
                    }
                }
                value
            })
            .flat_map(|value| std::iter::repeat_n(value, repeats))
            .collect();
        geometry.attributes.insert(name.clone(), values);
    }
    if let Some(colors) = geometry.attributes.get("COLOR_0") {
        for (v, c) in geometry.vertices.iter_mut().zip(colors) {
            v.COLOR_0 = *c;
        }
    }
    Ok(())
}

/// Geometry group ranges are index ranges, independent of vertex allocation.
pub(crate) fn groups(
    source: &::vork_scene::v2::Geometry,
    count: usize,
    slots: usize,
) -> Result<Vec<(std::ops::Range<usize>, usize)>> {
    if source.material_ranges.is_empty() {
        return Ok(vec![(0..count, 0)]);
    }
    let covered: usize = source.material_ranges.iter().map(|g| g.count as usize).sum();
    ensure!(
        covered == count,
        "geometry material_ranges cover {covered} of {count} indices; they must partition the index range"
    );
    source
        .material_ranges
        .iter()
        .map(|group| {
            let start = group.start as usize;
            let end = start
                .checked_add(group.count as usize)
                .context("geometry material_ranges range overflow")?;
            ensure!(
                end <= count,
                "geometry material_ranges entry requires an in-range index range"
            );
            ensure!(
                (group.material_index as usize) < slots,
                "geometry material_ranges material_index exceeds the mesh's {slots} materials"
            );
            Ok((start..end, group.material_index as usize))
        })
        .collect()
}

#[cfg(test)]
mod buffer_decode_tests {
    use super::*;
    #[test]
    fn shared_sidecar_decodes_only_each_authored_range() -> Result<()> {
        let file =
            std::env::temp_dir().join(format!("vork-attribute-shared-{}.bin", std::process::id()));
        let mut bytes = vec![0xcc; 16];
        for value in [1.25_f32, -2., 3.5] {
            bytes.extend(value.to_le_bytes());
        }
        bytes.extend(0xffff_ffff_u32.to_le_bytes());
        bytes.extend(17_u32.to_le_bytes());
        bytes.push(0xcc); // Explicit counts must ignore unrelated trailing bytes.
        std::fs::write(&file, bytes)?;
        let decode = |element_type: &str, offset, count| -> Result<AttributeData> {
            AttributeData::decode(
                &vork_scene::v2::from_test_value(json!({"type":element_type,
                "file":&file,"offset":offset,"count":count}))?,
                Path::new("."),
            )
        };
        assert_eq!(
            decode("vec3", 16, 1)?.floats::<3>()?,
            vec![[1.25, -2., 3.5]]
        );
        assert_eq!(decode("u32", 28, 2)?.words, vec![u32::MAX, 17]);
        std::fs::remove_file(file)?;
        Ok(())
    }

    #[test]
    fn integer_bits_and_nested_inline_values_match_scene_validation() -> Result<()> {
        let buffer = vork_scene::v2::from_test_value(json!({"type":"i32","data":[[-1]]}))?;
        assert_eq!(
            AttributeData::decode(&buffer, Path::new("."))?.words,
            vec![u32::MAX]
        );
        let buffer = vork_scene::v2::from_test_value(
            json!({"type":"mat4","data":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]]}),
        )?;
        assert_eq!(AttributeData::decode(&buffer, Path::new("."))?.count(), 1);
        Ok(())
    }
    #[test]
    fn bb107_sdk_written_implicit_field_loads() -> Result<()> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bb107/written.json");
        let scene = vork_scene::v2::Scene::load(&path)?;
        let buffers = SceneBuffers::decode(&scene, path.parent().unwrap())?;
        assert_eq!(buffers.data["field"].words, (0..1024).map(|i| ((i % 16) as f32 - 7.5).to_bits()).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn dense_binary_cells_use_dimensions_and_file_range() -> Result<()> {
        let path = std::env::temp_dir().join(format!("vork-dense-bb107-{}.bin", std::process::id()));
        let cells: Vec<f32> = (0..1024).map(|i| (i % 16) as f32 - 7.5).collect();
        let mut bytes = vec![0u8; 16];
        bytes.extend(cells.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend([0u8; 4]);
        std::fs::write(&path, &bytes)?;
        for count in [None, Some(1024)] {
            let buffer = vork_scene::v2::from_test_value(json!({"type":"f32", "dims":[16,8,8], "file":path, "offset":16, "count":count}))?;
            assert_eq!(AttributeData::decode(&buffer, Path::new("."))?.words,
                cells.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        }
        std::fs::write(&path, &bytes[..bytes.len()-8])?;
        let buffer = vork_scene::v2::from_test_value(json!({"type":"f32", "dims":[16,8,8], "file":path, "offset":16}))?;
        assert!(AttributeData::decode(&buffer, Path::new(".")).err().unwrap().to_string().contains("too short"));
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn binary_ranges_partial_elements_and_nonfinite_values_are_rejected() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("vork-attribute-decode-{}.bin", std::process::id()));
        std::fs::write(&path, [0u8; 4])?;
        let decode = |ty: &str, offset: u64, count: Option<u32>| -> Result<AttributeData> {
            AttributeData::decode(
                &vork_scene::v2::from_test_value(
                    json!({"type":ty,"file":path,"offset":offset,"count":count}),
                )?,
                Path::new("."),
            )
        };
        assert!(decode("u32", 8, None).is_err());
        assert!(decode("u32", 0, Some(2)).is_err());
        assert!(decode("vec2", 0, None).is_err());
        std::fs::write(&path, f32::NAN.to_le_bytes())?;
        assert!(decode("f32", 0, None).is_err());
        assert_eq!(decode("u32", 0, None)?.words, vec![f32::NAN.to_bits()]);
        std::fs::remove_file(path)?;
        Ok(())
    }
}

pub(crate) fn inherit_visibility(
    into: &mut ::vork_scene::v2::Visibility,
    from: &::vork_scene::v2::Visibility,
) {
    into.visible &= from.visible;
    into.camera &= from.camera;
    into.shadow &= from.shadow;
    into.diffuse &= from.diffuse;
    into.glossy &= from.glossy;
    into.transmission &= from.transmission;
}

/// Imported masks depend only on the primitive's ancestry, even when draws group.
pub(crate) fn imported_visibility_mask(
    overrides: &::vork_scene::v2::Table<::vork_scene::v2::AssetNodeOverrides>,
    path: &[(usize, String)],
) -> u32 {
    let mut visibility = ::vork_scene::v2::Visibility::default();
    for (index, name) in path {
        if let Some(value) =
            imported_override(overrides, *index, name).and_then(|node| node.visibility.as_ref())
        {
            inherit_visibility(&mut visibility, value);
        }
    }
    crate::lights::object_visibility_mask(&visibility)
}

pub(crate) fn imported_override<'a>(
    overrides: &'a ::vork_scene::v2::Table<::vork_scene::v2::AssetNodeOverrides>,
    index: usize,
    name: &str,
) -> Option<&'a ::vork_scene::v2::AssetNodeOverrides> {
    overrides
        .get(&format!("#{index}"))
        .or_else(|| overrides.get(name))
}

pub(crate) struct DrawId<'a> {
    base: Option<u32>,
    instances: Option<&'a [u32]>,
}
impl DrawId<'_> {
    pub(crate) fn instance(&self, index: Option<u32>) -> Result<u32> {
        if let Some(ids) = self.instances {
            Ok(*ids
                .get(index.context("instance ID requires an instance index")? as usize)
                .context("instance ID count differs from instance columns")?)
        } else {
            self.base.context("missing segmentation ID")
        }
    }
}
pub(crate) struct SegmentationIds {
    reserved: std::collections::HashSet<u32>,
    assigned: std::collections::HashMap<(usize, Option<usize>), u32>,
    metadata: std::collections::BTreeMap<u32, serde_json::Value>,
    owners: std::collections::HashSet<(u32, usize, Option<usize>)>,
    next: u32,
}
impl SegmentationIds {
    pub(crate) fn new(
        input: &::vork_scene::v2::Scene,
        attributes: &SceneBuffers,
    ) -> Result<Self> {
        let mut reserved = std::collections::HashSet::new();
        fn reserve(reserved: &mut std::collections::HashSet<u32>, id: u32) -> Result<()> {
            ensure!(
                (1..=constants::MAX_OBJECT_ID as u32).contains(&id),
                "object ID must be 1..=65535; 0 is background"
            );
            reserved.insert(id);
            Ok(())
        }
        fn visit(
            nodes: &[::vork_scene::v2::Node],
            attributes: &SceneBuffers,
            reserved: &mut std::collections::HashSet<u32>,
        ) -> Result<()> {
            use ::vork_scene::v2::{BufferType, NodeKind};
            for node in nodes {
                if let Some(id) = node.pass_index {
                    reserve(reserved, u32::from(id))?;
                }
                match &node.kind {
                    NodeKind::Asset { nodes, .. } => {
                        for node in nodes.values() {
                            if let Some(id) = node.pass_index {
                                reserve(reserved, u32::from(id))?;
                            }
                        }
                    }
                    NodeKind::Mesh {
                        instances: Some(instances),
                        ..
                    } => {
                        if let Some(buffer) = instances.attributes.get("pass_index") {
                            let data = &attributes.data;
                            let ids = data.get(buffer).context("unknown instance ID buffer")?;
                            ensure!(
                                matches!(ids.element_type, BufferType::U32) && ids.width == 1,
                                "instance id requires scalar u32"
                            );
                            ensure!(
                                ids.count() == instance_count(instances, attributes)?,
                                "instance ID count differs from instance columns"
                            );
                            for &id in &ids.words {
                                reserve(reserved, id)?;
                            }
                        }
                    }
                    _ => {}
                }
                visit(&node.children, attributes, reserved)?;
            }
            Ok(())
        }
        visit(&input.nodes, attributes, &mut reserved)?;
        Ok(Self {
            reserved,
            assigned: Default::default(),
            metadata: Default::default(),
            owners: Default::default(),
            next: 1,
        })
    }
    fn automatic(&mut self, owner: usize, imported: Option<usize>) -> Result<u32> {
        let owner = (owner, imported);
        if let Some(id) = self.assigned.get(&owner) {
            return Ok(*id);
        }
        while self.reserved.contains(&self.next) {
            self.next += 1;
        }
        ensure!(
            self.next as usize <= constants::MAX_OBJECT_ID,
            "16-bit object ID limit exceeded"
        );
        let id = self.next;
        self.next += 1;
        self.assigned.insert(owner, id);
        Ok(id)
    }
    fn record(
        &mut self,
        id: u32,
        owner: usize,
        name: &str,
        imported: Option<&(usize, String)>,
        instances: Option<usize>,
    ) {
        if !self.owners.insert((id, owner, imported.map(|v| v.0))) {
            return;
        }
        let mut metadata = json!({"pass_index":id,"node_index":owner,"name":name});
        if let Some((index, name)) = imported {
            metadata["gltf_node_index"] = json!(index);
            metadata["gltf_node_name"] = json!(name);
        }
        if let Some(count) = instances {
            metadata["instance_count"] = json!(count);
        }
        if let Some(previous) = self.metadata.get_mut(&id) {
            previous
                .as_object_mut()
                .unwrap()
                .entry("owners")
                .or_insert(json!([]))
                .as_array_mut()
                .unwrap()
                .push(metadata);
        } else {
            self.metadata.insert(id, metadata);
        }
    }
    /// Resolve subtree ownership once per draw; instance lookup below is O(1).
    pub(crate) fn prepare<'a>(
        &mut self,
        node: usize,
        path: &[(usize, String)],
        nodes: &[EvalNode],
        attributes: &'a SceneBuffers,
    ) -> Result<DrawId<'a>> {
        let source = &nodes[node].node;
        if let NodeKind::Mesh {
            instances: Some(instances),
            ..
        } = &source.kind
        {
            if let Some(buffer) = instances.attributes.get("pass_index") {
                let ids = &attributes
                    .data
                    .get(buffer)
                    .context("unknown instance ID buffer")?
                    .words;
                let mut counts = std::collections::BTreeMap::new();
                for &id in ids {
                    *counts.entry(id).or_insert(0usize) += 1;
                }
                for (id, count) in counts {
                    self.record(id, node, &nodes[node].node.name, None, Some(count));
                }
                return Ok(DrawId {
                    base: None,
                    instances: Some(ids),
                });
            }
        }
        let mut explicit = None;
        let mut owner = node;
        let mut imported = None;
        let mut current = Some(node);
        while let Some(index) = current {
            if let Some(id) = nodes[index].node.pass_index {
                explicit = Some(u32::from(id));
                owner = index;
                break;
            }
            current = nodes[index].parent;
        }
        if let NodeKind::Asset {
            nodes: overrides, ..
        } = &source.kind
        {
            for (index, name) in path.iter().rev() {
                if let Some(id) = imported_override(overrides, *index, name).and_then(|n| n.pass_index) {
                    explicit = Some(u32::from(id));
                    owner = node;
                    imported = Some((*index, name.clone()));
                    break;
                }
            }
        }
        let id = if let Some(id) = explicit {
            id
        } else {
            if matches!(source.kind, NodeKind::Asset { .. }) {
                imported = path.last().cloned();
            }
            self.automatic(owner, imported.as_ref().map(|(index, _)| *index))?
        };
        self.record(id, owner, &nodes[owner].node.name, imported.as_ref(), None);
        Ok(DrawId {
            base: Some(id),
            instances: None,
        })
    }
    pub(crate) fn metadata(self) -> Vec<serde_json::Value> {
        self.metadata.into_values().collect()
    }
}

#[cfg(test)]
mod segmentation_tests {
    use super::*;
    fn eval(name: &str, parent: Option<usize>) -> EvalNode {
        let node: Node = vork_scene::v2::from_test_value(json!({"type":"node","name":name})).unwrap();
        EvalNode { camera: None,
            node,
            parent,
            world: Mat4::IDENTITY,
            dynamic: false,
            embedded: None,
        }
    }
    #[test]
    fn imported_masks_keep_siblings_independent_after_grouping() -> Result<()> {
        let overrides = vork_scene::v2::from_test_value(json!({
            "Wheel":{"visible_camera":false},
            "#3":{"visible_glossy":false}
        }))?;
        let wheel = imported_visibility_mask(&overrides, &[(0, "Car".into()), (2, "Wheel".into())]);
        let body = imported_visibility_mask(&overrides, &[(0, "Car".into()), (3, "Body".into())]);
        assert_eq!(wheel & constants::RAY_CAMERA, 0);
        assert_ne!(wheel & constants::RAY_SPECULAR, 0);
        assert_ne!(body & constants::RAY_CAMERA, 0);
        assert_eq!(body & constants::RAY_SPECULAR, 0);
        assert_eq!(
            imported_visibility_mask(&overrides, &[(4, "Other".into())]),
            constants::RAY_VISIBILITY_ALL
        );
        Ok(())
    }

    /// The installed preorder of an authored tree.
    fn installed(input: &::vork_scene::v2::Scene) -> Vec<EvalNode> {
        fn visit(nodes: &[Node], parent: Option<usize>, out: &mut Vec<EvalNode>) {
            for node in nodes {
                let index = out.len();
                out.push(EvalNode { camera: None,
                    node: node.without_children(),
                    parent,
                    world: Mat4::IDENTITY,
                    dynamic: false,
                    embedded: None,
                });
                visit(&node.children, Some(index), out);
            }
        }
        let mut out = Vec::new();
        visit(&input.nodes, None, &mut out);
        out
    }
    #[test]
    fn immutable_point_ids_are_unique_for_inline_and_file_buffers() -> Result<()> {
        let directory=std::env::temp_dir().join(format!("vork-point-ids-{}",std::process::id()));
        std::fs::create_dir_all(&directory)?;
        std::fs::write(directory.join("ids.bin"),bytemuck::cast_slice(&[7u32,7]))?;
        for ids in [json!({"type":"u32","data":[7,7]}),json!({"type":"u32","file":"ids.bin","count":2})] {
            let mut value=json!({"version":2,"nodes":[{"type":"camera"}],"buffers":{
                "p":{"type":"vec3","data":[[0,0,0],[1,0,0]]},"w":{"type":"f32","data":[1,1]},"ids":ids},
                "geometry":{"points":{"topology":"points","attributes":{"POSITION":"p"},"widths":"w","ids":"ids"}}});
            let scene=crate::test_scene(value.clone(),&directory)?;
            let error=SceneBuffers::decode(&scene,&directory).err().context("duplicate immutable point IDs admitted")?;
            assert!(format!("{error:#}").contains("duplicate point identity 7"));
            value["buffers"]["ids"]=json!({"type":"u32","data":[7,8]});
            SceneBuffers::decode(&crate::test_scene(value,&directory)?,&directory)?;
        }
        std::fs::remove_dir_all(&directory)?;
        Ok(())
    }

    fn attributes(ids: Vec<u32>, transforms: usize) -> SceneBuffers {
        let mut data = std::collections::BTreeMap::new();
        data.insert(
            "ids".into(),
            AttributeData {
                element_type: ::vork_scene::v2::BufferType::U32,
                width: 1,
                words: ids,
            },
        );
        data.insert(
            "transforms".into(),
            AttributeData {
                element_type: ::vork_scene::v2::BufferType::Vec3,
                width: 3,
                words: vec![0; transforms * 3],
            },
        );
        SceneBuffers { data, grids: Default::default() }
    }
    #[test]
    fn visibility_is_anded_through_the_installed_parent_chain() {
        let mut parent = eval("parent", None);
        parent.node.visibility.camera = false;
        parent.node.visibility.glossy = false;
        let mut child = eval("child", Some(0));
        child.node.visibility.shadow = false;
        let mask = node_visibility_mask(&vec![parent, child], 1);
        assert_eq!(mask & constants::RAY_CAMERA, 0);
        assert_eq!(mask & constants::RAY_SPECULAR, 0);
        assert_eq!(mask & constants::RAY_SHADOW, 0);
        assert_ne!(mask & constants::RAY_DIFFUSE, 0);
    }
    #[test]
    fn hidden_nodes_and_imported_ancestors_disable_every_ray() -> Result<()> {
        let mut parent = eval("parent", None);
        parent.node.visibility.visible = false;
        let child = eval("child", Some(0));
        assert_eq!(node_visibility_mask(&vec![parent, child], 1), 0);
        let overrides = vork_scene::v2::from_test_value(json!({"Car":{"visible":false}}))?;
        assert_eq!(imported_visibility_mask(&overrides, &[(0,"Car".into()),(1,"Wheel".into())]),0);
        assert_eq!(imported_visibility_mask(&overrides, &[(2,"Other".into())]),constants::RAY_VISIBILITY_ALL);
        Ok(())
    }
    #[test]
    fn only_explicit_ids_are_inherited_by_unnamed_meshes() -> Result<()> {
        for explicit in [None, Some(7)] {
            let input = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[{
                "type":"node","name":"props","pass_index":explicit,"children":[
                    {"type":"mesh","geometry":"g"},{"type":"mesh","geometry":"g"}]}]}))?;
            let nodes = installed(&input);
            let attrs = attributes(vec![], 0);
            let mut ids = SegmentationIds::new(&input, &attrs)?;
            let a = ids.prepare(1, &[], &nodes, &attrs)?.instance(None)?;
            let b = ids.prepare(2, &[], &nodes, &attrs)?.instance(None)?;
            if explicit.is_some() {
                assert_eq!((a, b), (7, 7));
            } else {
                assert_eq!((a, b), (1, 2));
            }
        }
        Ok(())
    }

    #[test]
    fn imported_mesh_nodes_get_distinct_automatic_ids_but_share_primitive_ids() -> Result<()> {
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"asset","name":"car","file":"car.glb"}]}),
        )?;
        let nodes = installed(&input);
        let attrs = attributes(vec![], 0);
        let mut ids = SegmentationIds::new(&input, &attrs)?;
        for (index, expected) in [(3, 1), (4, 2), (3, 1)] {
            assert_eq!(
                ids.prepare(0, &[(index, format!("mesh{index}"))], &nodes, &attrs)?
                    .instance(None)?,
                expected
            );
        }
        let metadata = ids.metadata();
        assert_eq!(metadata[0]["gltf_node_index"], 3);
        assert_eq!(metadata[1]["gltf_node_index"], 4);
        Ok(())
    }

    #[test]
    fn reserves_explicit_ids_and_reports_the_matched_imported_ancestor() -> Result<()> {
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"node","name":"parent","children":[{"type":"asset","name":"car","file":"car.glb","nodes":{"#3":{"pass_index":7}}}],"pass_index":1}]}),
        )?;
        let nodes = installed(&input);
        let none = attributes(vec![], 0);
        let mut ids = SegmentationIds::new(&input, &none)?;
        assert_eq!(ids.automatic(9, None)?, 2);
        assert_eq!(ids.automatic(9, None)?, 2);
        let draw = ids.prepare(1, &[(3, "Car".into()), (4, "Wheel".into())], &nodes, &none)?;
        assert_eq!(draw.instance(None)?, 7);
        let metadata = ids.metadata();
        assert_eq!(metadata[0]["gltf_node_name"], "Car");
        assert_eq!(metadata[0]["gltf_node_index"], 3);
        Ok(())
    }
    #[test]
    fn class_label_metadata_is_per_id_and_instance_ids_override_subtree_ids() -> Result<()> {
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"mesh","name":"scatter","geometry":"g","instances":{"attributes":{"TRANSLATION":"transforms","pass_index":"ids"}},"pass_index":8}]}),
        )?;
        let attrs = attributes(vec![77; 100], 100);
        let nodes = installed(&input);
        let mut ids = SegmentationIds::new(&input, &attrs)?;
        for _ in 0..2 {
            let draw = ids.prepare(0, &[], &nodes, &attrs)?;
            for index in 0..100 {
                assert_eq!(draw.instance(Some(index))?, 77);
            }
        }
        let metadata = ids.metadata();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata[0]["instance_count"], 100);
        assert!(metadata[0].get("owners").is_none());
        Ok(())
    }
    #[test]
    fn rejects_background_id_and_mismatched_instance_ids() -> Result<()> {
        for node in [
            json!({"type":"node","pass_index":0}),
            json!({"type":"asset","file":"x.glb","nodes":{"#0":{"pass_index":0}}}),
        ] {
            let input = vork_scene::v2::from_test_value(json!({"version":2,"nodes":[node]}))?;
            assert!(SegmentationIds::new(&input, &attributes(vec![], 0)).is_err());
        }
        let input = vork_scene::v2::from_test_value(
            json!({"version":2,"nodes":[{"type":"mesh","geometry":"g","instances":{"attributes":{"TRANSLATION":"transforms","pass_index":"ids"}}}]}),
        )?;
        assert!(SegmentationIds::new(&input, &attributes(vec![2, 3], 1)).is_err());
        assert!(SegmentationIds::new(&input, &attributes(vec![0], 1)).is_err());
        Ok(())
    }
}

pub(crate) fn deformation(
    r: &Renderer,
    attributes: &SceneBuffers,
    source: &::vork_scene::v2::Geometry,
    skin_definition: Option<&::vork_scene::v2::Skin>,
    geometry: &crate::geometry::Geometry,
    binding: std::sync::Arc<crate::geometry::NativeDeform>,
) -> Result<Option<Deform>> {
    use ::vork_scene::v2::BufferType;
    if skin_definition.is_none() && source.morphs.is_empty() {
        return Ok(None);
    }
    let count = geometry.refinement.as_ref().map_or_else(|| if geometry.vertex_ids.is_empty() {geometry.vertices.len()} else {attributes.data[&source.attributes["POSITION"]].count()}, |r|r.positions.len());
    let joint_count = skin_definition.map_or(1, |skin| skin.joints.len());
    let influences: Vec<Influence> = if skin_definition.is_some() {
        let joints = &attributes.data[source
            .attributes
            .get("JOINTS_0")
            .context("skin requires JOINTS_0 attribute")?];
        let weights = &attributes.data[source
            .attributes
            .get("WEIGHTS_0")
            .context("skin requires WEIGHTS_0 attribute")?];
        ensure!(joints.count() == count, "JOINTS_0 requires matching uvec4");
        ensure!(weights.count() == count, "WEIGHTS_0 requires matching vec4");
        joints
            .words
            .chunks_exact(4)
            .zip(weights.floats::<4>()?)
            .map(|(joints, weights)| Influence {
                JOINTS_0: joints.try_into().unwrap(),
                WEIGHTS_0: weights,
            })
            .collect()
    } else {
        vec![
            Influence {
                JOINTS_0: [0; 4],
                WEIGHTS_0: [1., 0., 0., 0.]
            };
            count
        ]
    };
    let mut morphs = Vec::new();
    for morph in source.morphs.values() {
        let positions = &attributes.data[&morph.POSITION];
        ensure!(
            matches!(positions.element_type, BufferType::Vec3) && positions.count() == count,
            "morph positions require matching vec3"
        );
        let normals = if let Some(name) = &morph.NORMAL {
            let normals = &attributes.data[name];
            ensure!(
                matches!(normals.element_type, BufferType::Vec3) && normals.count() == count,
                "morph normals require matching vec3"
            );
            normals.floats::<3>()?
        } else {
            vec![[0.; 3]; count]
        };
        for (position, normal) in positions.floats::<3>()?.into_iter().zip(normals) {
            morphs.extend(position);
            morphs.extend(normal);
            morphs.extend([0.; 3]);
        }
    }
    if let Some(refinement) = &geometry.refinement {
        let positions = morphs.chunks_exact(constants::MORPH_FLOATS).map(|v| [v[0],v[1],v[2]]).collect();
        let mut cpu = crate::subdivision::CpuDeform::new(std::sync::Arc::clone(refinement), influences, positions, joint_count)?;
        cpu.base = geometry.vertices.clone();
        let mut deform = crate::geometry::deformation(r, None, 0, None, &geometry.vertices,
            vec![Influence {JOINTS_0:[0;4],WEIGHTS_0:[1.,0.,0.,0.]}; geometry.vertices.len()],
            vec![0.; 3*geometry.vertices.len()*constants::MORPH_FLOATS], 1, 1, 3)?;
        deform.cpu = Some(std::sync::Arc::new(cpu));
        deform.native = Some(binding);
        return Ok(Some(deform));
    }
    // Flat corners retain their authored vertex identity. Expand the same skin
    // and morph records; subdivision above still consumes its shared cage.
    let (influences,morphs) = if geometry.vertex_ids.is_empty() {(influences,morphs)} else {
        let influences=geometry.vertex_ids.iter().map(|&id|influences[id as usize]).collect();
        let mut expanded=Vec::with_capacity(geometry.vertices.len()*source.morphs.len()*constants::MORPH_FLOATS);
        for target in 0..source.morphs.len() {for &id in &geometry.vertex_ids {
            let at=(target*count+id as usize)*constants::MORPH_FLOATS;
            expanded.extend_from_slice(&morphs[at..at+constants::MORPH_FLOATS]);
        }}
        (influences,expanded)
    };
    let mut deform = crate::geometry::deformation(
        r,
        None,
        0,
        None,
        &geometry.vertices,
        influences,
        morphs,
        joint_count,
        1,
        source.morphs.len(),
    )?;
    deform.native = Some(binding);
    Ok(Some(deform))
}

pub(crate) fn native_binding(
    scene: &::vork_scene::v2::Scene,
    nodes: &[EvalNode],
    node_names: &std::collections::HashMap<String, usize>,
    mesh: usize,
    attributes: &SceneBuffers,
) -> Result<std::sync::Arc<crate::geometry::NativeDeform>> {
    use ::vork_scene::v2::BufferType;
    let source = &nodes[mesh].node;
    let NodeKind::Mesh {
        geometry,
        skin,
        morph_weights,
        ..
    } = &source.kind
    else {
        bail!("deformation source is not a mesh")
    };
    let geometry = &scene.geometry[geometry];
    ensure!(
        morph_weights
            .keys()
            .all(|name| geometry.morphs.contains_key(name)),
        "unknown morph weight on mesh {:?}",
        source.name
    );
    let joints = if let Some(skin) = skin {
        let inverse = &attributes.data[&skin.inverse_bind_matrices];
        ensure!(
            matches!(inverse.element_type, BufferType::Mat4)
                && inverse.count() == skin.joints.len(),
            "inverse-bind count must match joints"
        );
        skin.joints
            .iter()
            .zip(inverse.floats::<16>()?)
            .map(|(name, inverse)| {
                let joint = *node_names.get(name).with_context(|| {
                    format!("skin joint {name:?} must name exactly one installed node")
                })?;
                let inverse = Mat4::from_cols_array(&inverse);
                ensure!(
                    inverse.is_finite() && inverse.determinant() != 0.,
                    "inverse-bind matrix must be finite and invertible"
                );
                Ok((joint, inverse))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![(mesh, Mat4::IDENTITY)]
    };
    Ok(std::sync::Arc::new(crate::geometry::NativeDeform {
        mesh,
        joints: std::sync::Arc::new(joints),
        morphs: std::sync::Arc::new(geometry.morphs.keys().cloned().collect()),
    }))
}

/// The same local node masks feed install values and slot-local sampled values.
pub(crate) fn node_visibility_mask<N>(nodes: &N, node: usize) -> u32
where
    N: std::ops::Index<usize, Output = EvalNode> + ?Sized,
{
    let mut mask = constants::RAY_VISIBILITY_ALL;
    let mut current = Some(node);
    while let Some(index) = current {
        mask &= lights::object_visibility_mask(&nodes[index].node.visibility);
        current = nodes[index].parent;
    }
    mask
}

/// Geometry inherits collapsed scale; analytic lights only inherit visibility.
pub(crate) fn node_collapsed<N: std::ops::Index<usize, Output = EvalNode> + ?Sized>(nodes: &N, node: usize) -> bool {
    let mut current = Some(node);
    while let Some(index) = current {
        if nodes[index].node.transform.scale.contains(&0.) {return true;}
        current = nodes[index].parent;
    }
    false
}

/// Install-time bindings into persistent compute storage. Decode remains shared
/// with geometry, skin and tracks; no second CPU copy/cache is retained.
pub(crate) struct ProgramBuffers {
    pub storage: Vec<wgpu::Buffer>,
    pub initial_identity: std::collections::BTreeMap<String, blake3::Hash>,
    pub bindings: std::collections::BTreeMap<String, AttributeBinding>,
}
#[derive(Clone)]
pub(crate) struct AttributeBinding {
    pub count: u32,
    pub width: u32,
    pub spans: Vec<AttributeSpan>,
}
#[derive(Clone, Debug)]
pub(crate) struct AttributeSpan {
    pub page: u32,
    pub offset: u32,
    pub first: u32,
    pub count: u32,
}
impl ProgramBuffers {
    pub fn install(r: &Renderer, scene: &Scene, decoded: &SceneBuffers) -> Result<Self> {
        Self::install_with_page_limit(r, scene, decoded, constants::SCENE_BUFFER_PAGE_BYTES)
    }
    pub(crate) fn install_with_page_limit(r: &Renderer, scene: &Scene, decoded: &SceneBuffers, page_bytes: usize) -> Result<Self> {
        let mut names: std::collections::BTreeSet<String> = scene
            .programs
            .values()
            .filter(|p| p.is_compute())
            .flat_map(|p| p.bindings.values().cloned())
            .collect();
        names.extend(scene.buffers.iter().filter(|(_,b)|b.dims.is_some()).map(|(n,_)|n.clone()));
        names.extend(scene.simulation.iter().flat_map(|s| s.state.iter().cloned()));
        // Status is a retained consumer even when no compute program writes it.
        names.extend(scene.geometry.values().filter_map(|g|g.status.as_ref()).map(|s|s.buffer.clone()));
        for g in scene.geometry.values().filter(|g| crate::computed_geometry::dynamic(scene,g)) {
            names.extend(g.attributes.values().chain(g.widths.iter()).chain(g.count.iter()).chain(g.ids.iter()).cloned());
        }
        fn instances(scene: &Scene, nodes: &[Node], names: &mut std::collections::BTreeSet<String>) {
            for node in nodes {
                if let NodeKind::Mesh {
                    instances: Some(inst),
                    ..
                } = &node.kind
                {
                    if crate::compute::places_instances(scene, node) {
                        names.extend(inst.attributes.values().cloned());
                    }
                }
                instances(scene, &node.children, names);
            }
        }
        instances(scene, &scene.nodes, &mut names);
        let limit = (r.device.limits().max_storage_buffer_binding_size as usize / 4)
            .min(r.device.limits().max_buffer_size as usize / 4)
            .min(page_bytes / 4);
        let mut pages: Vec<Vec<u32>> = vec![vec![]];
        let mut bindings = std::collections::BTreeMap::new();
        let mut initial_identity = std::collections::BTreeMap::new();
        for name in names {
            let data = &decoded.data[&name];
            if scene.simulation.as_ref().is_some_and(|sim| sim.state.contains(&name)) {
                let mut identity = blake3::Hasher::new();
                identity.update(&serde_json::to_vec(&data.element_type)?);
                identity.update(&(data.words.len() as u64).to_le_bytes());
                identity.update(bytemuck::cast_slice(&data.words));
                initial_identity.insert(name.clone(), identity.finalize());
            }
            ensure!(data.width <= limit, crate::gpu_memory::device_limit(format_args!("compute buffer {name} element"), format_args!("{} words",data.width), format_args!("{limit} words"), "reduce the compute element width"));
            let mut spans = Vec::new();
            let mut first = 0;
            while first < data.count() {
                let page = pages.len() - 1;
                let count = ((limit - pages[page].len()) / data.width).min(data.count() - first);
                if count == 0 {
                    pages.push(vec![]);
                    continue;
                }
                spans.push(AttributeSpan {
                    page: u32::try_from(page)?,
                    offset: u32::try_from(pages[page].len())?,
                    first: u32::try_from(first)?,
                    count: u32::try_from(count)?,
                });
                pages[page].extend_from_slice(
                    &data.words[first * data.width..(first + count) * data.width],
                );
                first += count;
            }
            bindings.insert(
                name,
                AttributeBinding {
                    count: u32::try_from(data.count())?,
                    width: u32::try_from(data.width)?,
                    spans,
                },
            );
        }
        let storage = pages
            .iter()
            .map(|words| {
                init(
                    &r.device,
                    "persistent scene buffer page",
                    if words.is_empty() {
                        &[0; 4]
                    } else {
                        bytemuck::cast_slice(words)
                    },
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                )
            })
            .collect();
        Ok(Self { storage, bindings, initial_identity })
    }
}

#[cfg(test)]
mod admission_tests {
    #[test]
    fn expanded_work_is_checked_without_allocating_it() {
        let mut total = 0;
        let error = super::admit_triangle_instances(50_000 * crate::points::triangles_per_point(12).unwrap(),
            20_000, &mut total).unwrap_err();
        assert!(error.to_string().contains("reduce geometry density or instance count"));
        let mut total = 0;
        super::admit_triangle_instances(1_000, 20_000, &mut total).unwrap();
        assert_eq!(total, 20_000_000);
        assert!(super::admit_triangle_instances(u64::MAX, 2, &mut total).is_err());
    }
}

#[cfg(test)]
mod instance_trs_tests {
    use super::*;
    #[test]
    fn instance_columns_default_compose_and_validate() -> Result<()> {
        let mut value=json!({"version":2,"nodes":[],"buffers":{
            "t":{"type":"vec3","data":[[2,3,4]]},"r":{"type":"vec4","data":[[0,0,0.6,0.8]]},
            "s":{"type":"vec3","data":[[-2,3,4]]}}});
        let definition: vork_scene::Instances=vork_scene::v2::from_test_value(json!({"attributes":{"TRANSLATION":"t","ROTATION":"r","SCALE":"s"}}))?;
        let decode=|v:&serde_json::Value| SceneBuffers::decode(&vork_scene::v2::from_test_value(v.clone())?,Path::new("."));
        let expected=Mat4::from_scale_rotation_translation(Vec3::new(-2.,3.,4.),glam::Quat::from_xyzw(0.,0.,0.6,0.8).normalize(),Vec3::new(2.,3.,4.));
        assert!(instance_transforms(&definition,&decode(&value)?)?[0].abs_diff_eq(expected,1e-6));
        let only_translation: vork_scene::Instances=vork_scene::v2::from_test_value(json!({"attributes":{"TRANSLATION":"t"}}))?;
        assert_eq!(instance_transforms(&only_translation,&decode(&value)?)?[0],Mat4::from_translation(Vec3::new(2.,3.,4.)));
        value["buffers"]["r"]["data"]=json!([[0,0,0,0]]);
        assert!(instance_transforms(&definition,&decode(&value)?).unwrap_err().to_string().contains("unit"));
        value["buffers"]["r"]["data"]=json!([[0,0,0,1],[0,0,0,1]]);
        assert!(instance_transforms(&definition,&decode(&value)?).unwrap_err().to_string().contains("equal counts"));
        value["buffers"]["r"]["data"]=json!([[0,0,0,1]]);value["buffers"]["s"]["data"]=json!([[0,1,1]]);
        assert!(crate::collapsed(instance_transforms(&definition,&decode(&value)?)?[0]));
        Ok(())
    }
}

/// Object matte flags inherit down authored groups and imported asset roots.
pub(crate) fn node_matte_flags<N: std::ops::Index<usize, Output = EvalNode> + ?Sized>(nodes:&N,node:usize)->u32 {
    let mut flags=0;
    for index in std::iter::successors(Some(node),|&i|nodes[i].parent) {
        if nodes[index].node.is_holdout {flags|=vork_slang::registry::OBJECT_HOLDOUT;}
        if nodes[index].node.is_shadow_catcher {flags|=vork_slang::registry::OBJECT_CATCHER;}
    }
    flags
}


#[cfg(test)]
mod flat_normal_tests {
    use super::*;
    #[test]
    fn omitted_normals_split_faces_without_losing_attributes_or_identity() -> Result<()> {
        let mut input=json!({"version":2,"nodes":[{"type":"camera"}],
            "buffers":{
                "p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]},
                "i":{"type":"u32","data":[0,1,2,0,3,1]},
                "uv":{"type":"vec2","data":[[0,0],[1,0],[0,1],[1,1]]},
                "t":{"type":"vec4","data":[[1,0,0,1],[1,0,0,1],[1,0,0,1],[1,0,0,1]]},
                "marker":{"type":"f32","data":[10,20,30,40]},
                "n":{"type":"vec3","data":[[0,1,1],[0,1,1],[0,1,1],[0,1,1]]}},
            "geometry":{"mesh":{"attributes":{"POSITION":"p","TEXCOORD_0":"uv","TANGENT":"t","_marker":"marker"},
                "indices":"i","face_vertex_counts":[3,3]}}});
        let load=|value:serde_json::Value| -> Result<geometry::Geometry> {
            let scene: ::vork_scene::v2::Scene=::vork_scene::v2::from_test_value(value)?;
            geometry_v2(&SceneBuffers::decode(&scene,Path::new("."))?,&scene.geometry["mesh"],&["_marker".into()],4,None)
        };
        let flat=load(input.clone())?;
        assert_eq!(flat.indices,vec![0,1,2,3,4,5]);
        assert_eq!(flat.vertex_ids,vec![0,1,2,0,3,1]);
        assert_eq!(flat.primitive_ids,vec![0,1]);
        for (index,vertex) in flat.vertices.iter().enumerate() {
            assert_eq!(vertex.NORMAL,if index<3 {[0.,0.,1.]} else {[0.,1.,0.]});
            assert_eq!(vertex.TANGENT,[1.,0.,0.,1.]);
            assert_eq!(flat.attributes["_marker"][index][0],[10.,20.,30.,10.,40.,20.][index]);
        }
        assert_eq!(flat.vertices[3].TEXCOORD_0,[0.,0.]);
        assert_eq!(flat.vertices[4].TEXCOORD_0,[1.,1.]);
        input["geometry"]["mesh"]["attributes"]["NORMAL"]=json!("n");
        let smooth=load(input)?;
        assert_eq!(smooth.vertices.len(),4);
        assert_eq!(smooth.indices,vec![0,1,2,0,3,1]);
        assert!(smooth.vertices.iter().all(|v|v.NORMAL==Vec3::new(0.,1.,1.).normalize().to_array()));
        Ok(())
    }
}
