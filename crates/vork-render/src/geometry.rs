//! Loader output, independent of the file format and GPU allocation.
use super::*;

/// One evaluated vertex range, shared by raster, shadow and ray consumers.
#[derive(Clone)]
pub(crate) struct VertexStream {
    pub buffer: wgpu::Buffer,
    pub offset: u64,
    pub size: u64,
}
impl VertexStream {
    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size: std::num::NonZeroU64::new(self.size),
        })
    }
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(self.offset..self.offset + self.size)
    }
    /// The first vertex of this range in DeformedVertex units.
    pub fn first_vertex(&self) -> u32 {
        (self.offset / std::mem::size_of::<crate::DeformedVertex>() as u64) as u32
    }
}

#[derive(Clone)]
pub(crate) struct EvaluatedGeometry {
    /// Compact point endpoints: previous/current centre and diameter, eight floats per source row.
    pub(crate) point_poses: Option<wgpu::Buffer>,
    pub(crate) point_domain: Option<(u32,u32)>,
    pub(crate) gpu_bounds: Option<wgpu::Buffer>,
    pub(crate) output: VertexStream,
    pub(crate) previous_output: VertexStream,
    pub(crate) attributes: Option<wgpu::Buffer>,
    /// Two interleaved attribute rows per vertex, matching previous_output.
    /// Absent historical storage means the attributes are unchanged.
    pub(crate) previous_attributes: Option<wgpu::Buffer>,
}
impl From<&FrameDeform> for EvaluatedGeometry {
    fn from(frame: &FrameDeform) -> Self {
        Self {
            point_poses: None,
            point_domain: None,
            gpu_bounds: None,
            attributes: None,
            previous_attributes: None,
            output: frame.output.clone(),
            previous_output: VertexStream { buffer: frame.previous_output.clone(), offset: 0, size: frame.previous_output.size() },
        }
    }
}



#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Geometry {
    /// Authored point domain, restored after loading the sphere expansion cache.
    #[serde(skip)]
    pub(crate) point_centers: Vec<[f32; 3]>,
    #[serde(skip)]
    pub(crate) point_widths: Vec<f32>,
    #[serde(skip)]
    pub(crate) refinement: Option<std::sync::Arc<crate::subdivision::Refinement>>,
    pub(crate) vertices: Vec<Vertex>,
    /// Source POSITION accessor/index identity across imported splits; empty means identity indexing.
    pub(crate) vertex_ids: Vec<u64>,
    pub(crate) indices: Vec<u32>,
    pub(crate) primitive_ids: Vec<u32>,
    pub(crate) bounds: (Vec3, Vec3),
    /// Bit n: the source has TEXCOORD_n.
    pub(crate) uv_sets: u8,
    pub(crate) attributes: std::collections::BTreeMap<String, Vec<[f32; 4]>>,
}
/// A position's identity for welding attribute-split vertices: its bits, with
/// both zeros one key.
pub(crate) fn weld_key(position: [f32; 3]) -> [u32; 3] {
    position.map(|v| if v == 0. { 0 } else { v.to_bits() })
}
/// Keys that weld positions within `epsilon` (per axis) of an earlier one to
/// it: computed seams (a UV sphere's) differ by a few ulps.
pub(crate) fn weld_within(positions: &[[f32; 3]], epsilon: f32) -> Vec<[u32; 3]> {
    let cell = |p: [f32; 3]| p.map(|v| (v / epsilon).floor() as i64);
    let mut grid = std::collections::HashMap::<[i64; 3], Vec<[f32; 3]>>::new();
    positions.iter().map(|&p| {
        let c = cell(p);
        let near = (0..27).flat_map(|n| grid.get(&[c[0] + n % 3 - 1, c[1] + n / 3 % 3 - 1, c[2] + n / 9 - 1]).into_iter().flatten())
            .find(|r| (0..3).all(|k| (r[k] - p[k]).abs() <= epsilon)).copied();
        let representative = near.unwrap_or_else(|| {grid.entry(c).or_default().push(p); p});
        weld_key(representative)
    }).collect()
}
/// An undirected edge between welded positions, smaller endpoint first.
pub(crate) type Edge = [[u32; 3]; 2];
/// Per welded edge of a triangle list: uses, and winding (+1 per use from the
/// smaller endpoint, -1 per use from the larger). A closed, consistently
/// oriented manifold uses every edge (2, 0). Triangles that repeat a welded
/// position bound nothing and are left out; the flag says whether any did.
pub(crate) fn edge_uses(indices: &[u32], key: impl Fn(u32) -> [u32; 3]) -> (std::collections::HashMap<Edge, (u32, i32)>, bool) {
    let mut edges = std::collections::HashMap::<Edge, (u32, i32)>::new();
    let mut degenerate = false;
    for triangle in indices.chunks_exact(3) {
        let corners = [0, 1, 2].map(|k| key(triangle[k]));
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[2] == corners[0] {degenerate = true; continue;}
        for k in 0..3 {
            let (a, b) = (corners[k], corners[(k + 1) % 3]);
            let edge = edges.entry(if a < b {[a, b]} else {[b, a]}).or_default();
            edge.0 += 1;
            edge.1 += if a < b {1} else {-1};
        }
    }
    (edges, degenerate)
}

impl Geometry {
    /// Merge byte-identical immutable vertices without changing triangle order.
    /// Evaluated streams must keep their source indexing for skin/morph and
    /// topology-dependent processing; their caller deliberately skips this.
    pub(crate) fn indexed(&self) -> Option<Self> {
        use std::hash::{Hash, Hasher};
        struct Key<'a>(&'a Geometry, usize);
        impl Hash for Key<'_> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                bytemuck::bytes_of(&self.0.vertices[self.1]).hash(state);
                for values in self.0.attributes.values() {
                    bytemuck::bytes_of(&values[self.1]).hash(state);
                }
                self.0.vertex_ids.get(self.1).hash(state);
            }
        }
        impl PartialEq for Key<'_> {
            fn eq(&self, other: &Self) -> bool {
                bytemuck::bytes_of(&self.0.vertices[self.1])
                    == bytemuck::bytes_of(&other.0.vertices[other.1])
                    && self.0.attributes.values().zip(other.0.attributes.values())
                        .all(|(a,b)| bytemuck::bytes_of(&a[self.1]) == bytemuck::bytes_of(&b[other.1]))
                    && self.0.vertex_ids.get(self.1) == other.0.vertex_ids.get(other.1)
            }
        }
        impl Eq for Key<'_> {}
        let mut unique = std::collections::HashMap::with_capacity(self.vertices.len());
        let mut sources = Vec::new();
        let remap: Vec<_> = (0..self.vertices.len()).map(|index| {
            *unique.entry(Key(self,index)).or_insert_with(|| {
                let new = sources.len() as u32;
                sources.push(index);
                new
            })
        }).collect();
        if sources.len() == self.vertices.len() { return None; }
        Some(Self {
            refinement: self.refinement.clone(),
            point_centers: self.point_centers.clone(),
            point_widths: self.point_widths.clone(),
            vertices: sources.iter().map(|&i| self.vertices[i]).collect(),
            vertex_ids: sources.iter().filter_map(|&i| self.vertex_ids.get(i).copied()).collect(),
            indices: self.indices.iter().map(|&i| remap[i as usize]).collect(),
            primitive_ids: self.primitive_ids.clone(),
            bounds: self.bounds,
            uv_sets: self.uv_sets,
            attributes: self.attributes.iter().map(|(name, values)| {
                (name.clone(), sources.iter().map(|&i| values[i]).collect())
            }).collect(),
        })
    }

    pub(crate) fn new(vertices: Vec<Vertex>, indices: Vec<u32>, uv_sets: u8) -> Self {
        let bounds = vertices.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), v| (lo.min(Vec3::from(v.POSITION)), hi.max(Vec3::from(v.POSITION))),
        );
        Self {
            refinement: None,
            point_centers: Vec::new(),
            point_widths: Vec::new(),
            primitive_ids: (0..(indices.len()/3) as u32).collect(),
            vertex_ids: Vec::new(),
            vertices,
            indices,
            bounds,
            uv_sets,
            attributes: Default::default(),
        }
    }
    pub(crate) fn channels(
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[[f32; 2]; 2]>,
        indices: Vec<u32>,
        uv_sets: u8,
    ) -> Self {
        let vertices = positions
            .into_iter()
            .zip(normals)
            .zip(uvs)
            .map(|((p, n), uv)| Vertex {
                POSITION: p,
                NORMAL: Vec3::from(n).normalize_or_zero().to_array(),
                TEXCOORD_0: uv[0],
                TEXCOORD_1: uv[1],
                pad: [0.; 2],
                COLOR_0: [1.; 4],
                TANGENT: [0.; 4],
            })
            .collect();
        Self::new(vertices, indices, uv_sets)
    }
    /// Give every triangle its geometric normal while retaining source identity,
    /// index order, primitive IDs and all corner attributes.
    pub(crate) fn flat_normals(&mut self) -> Result<()> {
        let count = u32::try_from(self.indices.len()).context("flat-normal corner count exceeds u32 addressing")?;
        let source = &self.indices;
        let mut vertices = Vec::with_capacity(source.len());
        let mut ids = Vec::with_capacity(source.len());
        for tri in source.chunks_exact(3) {
            let [a,b,c] = [tri[0],tri[1],tri[2]].map(|i| Vec3::from(self.vertices[i as usize].POSITION));
            let normal = (b-a).cross(c-a).normalize_or_zero().to_array();
            for &index in tri {
                let mut vertex = self.vertices[index as usize];
                vertex.NORMAL = normal;
                vertex.pad[0] = 1.; // Generated face normal: reevaluate after morph/skin.
                vertices.push(vertex);
                ids.push(if self.vertex_ids.is_empty() {u64::from(index)} else {self.vertex_ids[index as usize]});
            }
        }
        for values in self.attributes.values_mut() {
            *values = source.iter().map(|&index| values[index as usize]).collect();
        }
        self.vertices = vertices;
        self.vertex_ids = ids;
        self.indices = (0..count).collect();
        Ok(())
    }
    /// Bounds of the vertices referenced by a group's index range.
    pub(crate) fn range_bounds(&self, range: std::ops::Range<usize>) -> (Vec3, Vec3) {
        self.indices[range].iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), &index| {
                let p = Vec3::from(self.vertices[index as usize].POSITION);
                (lo.min(p), hi.max(p))
            },
        )
    }
}

/// Lossless vertex fetch layout. A UV set equal to TEXCOORD_0 shares its stored pair;
/// absent/default attributes read from a stride-zero constant stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct VertexLayout {
    pub(crate) points: bool,
    pub(crate) deform: bool,
    pub(crate) tangent: bool,
    pub(crate) uv: [u8; 2],
    pub(crate) color: bool,
    pub(crate) custom: u8,
}
impl VertexLayout {
    /// Fetch layouts emitted by packing, for setup's built-in driver registry.
    pub(crate) fn builtin_layouts(custom: u8) -> Vec<Self> {
        let absent = constants::VERTEX_DEFAULT_UV;
        let mut layouts = Vec::new();
        for deform in [false,true] { for tangent in [false,true] {
            if deform && !tangent { continue; }
            for uv in [[absent,absent], [0,absent], [absent,0], [0,0], [0,1]] {
                for color in [false,true] {
                    layouts.push(Self { points:false, deform, tangent, uv, color, custom });
                }
            }
        }}
        layouts
    }
    pub(crate) fn for_vertices(vertices: &[Vertex], deform: bool) -> Self {
        let mut out = Self {
            deform,
            tangent: deform || vertices.iter().any(|v| v.TANGENT != [0.; 4]),
            points: false,
            uv: [constants::VERTEX_DEFAULT_UV; 2],
            color: vertices.iter().any(|v| v.COLOR_0 != [1.; 4]),
            custom: 0,
        };
        let mut unique = 0;
        for slot in 0..2 {
            if vertices.iter().all(|v| v.texcoord(slot) == [0.; 2]) {
                continue;
            }
            if let Some(prior) = (0..slot).find(|&p| {
                vertices
                    .iter()
                    .all(|v| v.texcoord(p).map(f32::to_bits) == v.texcoord(slot).map(f32::to_bits))
            }) {
                out.uv[slot] = out.uv[prior];
            } else {
                out.uv[slot] = unique;
                unique += 1;
            }
        }
        out
    }
    pub(crate) fn vertex_stride(self) -> u64 {
        if self.deform { 40 } else { 12 }
    }
    fn uv_count(self) -> u64 {
        self.uv
            .iter()
            .copied()
            .filter(|&v| v != constants::VERTEX_DEFAULT_UV)
            .max()
            .map_or(0, |v| u64::from(v) + 1)
    }
    fn surface_bytes(self) -> u64 {
        if self.deform {
            12 // Immutable rest position for Generated texture coordinates.
        } else {
            12 + u64::from(self.tangent) * 16
        }
    }
    pub(crate) fn constant_attributes(self) -> bool {
        false // Rest positions keep the immutable stream live for deforming geometry.
    }
    pub(crate) fn attribute_stride(self) -> u64 {
        (self.surface_bytes()
            + self.uv_count() * 8
            + u64::from(self.color) * 16
            + u64::from(self.custom) * 16)
            .max(4)
    }
    pub(crate) fn pack(
        self,
        vertices: &[Vertex],
        names: &[String],
        custom: &std::collections::BTreeMap<String, Vec<[f32; 4]>>,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut changing = Vec::with_capacity(vertices.len() * self.vertex_stride() as usize);
        let mut attributes = Vec::with_capacity(vertices.len() * self.attribute_stride() as usize);
        for (index, v) in vertices.iter().enumerate() {
            changing.extend_from_slice(bytemuck::bytes_of(&v.POSITION));
            let surface = if self.deform {
                &mut changing
            } else {
                &mut attributes
            };
            surface.extend_from_slice(bytemuck::bytes_of(&v.NORMAL));
            if self.tangent {
                surface.extend_from_slice(bytemuck::bytes_of(&v.TANGENT));
            }
            if self.deform {
                attributes.extend_from_slice(bytemuck::bytes_of(&v.POSITION));
            }
            for packed in 0..self.uv_count() as u8 {
                let slot = self.uv.iter().position(|&v| v == packed).unwrap();
                attributes.extend_from_slice(bytemuck::bytes_of(&v.texcoord(slot)));
            }
            if self.color {
                attributes.extend_from_slice(bytemuck::bytes_of(&v.COLOR_0));
            }
            for name in names {
                let value = custom.get(name).map_or([0.; 4], |data| data[index]);
                attributes.extend_from_slice(bytemuck::bytes_of(&value));
            }
            if self.attribute_stride() == 4 {
                attributes.extend_from_slice(&[0; 4]);
            }
        }
        (changing, attributes)
    }
    pub(crate) fn attributes(self) -> [Vec<wgpu::VertexAttribute>; 4] {
        use wgpu::{VertexAttribute as A, VertexFormat as F};
        let a = |shader_location, offset, format| A {
            shader_location,
            offset,
            format,
        };
        let mut out = [
            vec![a(0, 0, F::Float32x3)],
            vec![a(3, 0, F::Float32x3), a(16, if self.deform {40} else {0}, F::Float32x3)],
            Vec::new(),
            Vec::new(),
        ];
        out[if self.deform { 2 } else { 0 }].push(a(15, 0, F::Float32x3));
        let surface_slot = if self.deform { 0 } else { 2 };
        let surface_offset = if self.deform { 12 } else { 0 };
        out[surface_slot].push(a(1, surface_offset, F::Float32x3));
        if self.tangent {
            out[surface_slot].push(a(5, surface_offset + 12, F::Float32x4));
        } else {
            out[3].push(a(5, 0, F::Float32x4));
        }
        for (slot, location) in [2, 6].into_iter().enumerate() {
            let packed = self.uv[slot];
            if packed == constants::VERTEX_DEFAULT_UV {
                out[3].push(a(location, 0, F::Float32x2));
            } else {
                out[2].push(a(
                    location,
                    self.surface_bytes() + u64::from(packed) * 8,
                    F::Float32x2,
                ));
            }
        }
        if self.color {
            out[2].push(a(
                4,
                self.surface_bytes() + self.uv_count() * 8,
                F::Float32x4,
            ));
        } else {
            out[3].push(a(4, 16, F::Float32x4));
        }
        for slot in 0..self.custom {
            out[2].push(a(
                if slot < 8 {7+u32::from(slot)} else {vork_slang::registry::VISIBILITY_LENS_LOCATION+1+u32::from(slot)-8},
                self.surface_bytes()
                    + self.uv_count() * 8
                    + u64::from(self.color) * 16
                    + u64::from(slot) * 16,
                F::Float32x4,
            ));
        }
        // Keep active fetch bindings contiguous when all immutable channels are constants.
        if self.constant_attributes() {
            out.swap(2, 3);
        }
        out
    }
}

/// One bounded upload allocation; CPU pages already use their final GPU format.
pub(crate) struct GeometryPage {
    pub(crate) layout: VertexLayout,
    pub(crate) vertices: Vec<u8>,
    pub(crate) attributes: Vec<u8>,
    pub(crate) indices: Vec<u32>,
}
impl GeometryPage {
    pub(crate) fn new(layout: VertexLayout) -> Self {
        Self {
            layout,
            vertices: Vec::new(),
            attributes: Vec::new(),
            indices: Vec::new(),
        }
    }
}

/// Native skin bindings reference the already-evaluated installed node worlds.
pub(crate) struct NativeDeform {
    pub(crate) mesh: usize,
    pub(crate) joints: std::sync::Arc<Vec<(usize, Mat4)>>,
    pub(crate) morphs: std::sync::Arc<Vec<String>>,
}

impl NativeDeform {
    /// glTF joint matrices and named morph values from the common frame evaluation.
    pub(crate) fn palette(
        &self,
        worlds: &[Mat4],
        weights: &std::collections::BTreeMap<String, f32>,
    ) -> (Vec<Mat4>, Vec<f32>) {
        let inverse = crate::inverse_or_identity(worlds[self.mesh]);
        let matrices = self
            .joints
            .iter()
            .map(|(joint, bind)| {
                if *joint == self.mesh && *bind == Mat4::IDENTITY {
                    Mat4::IDENTITY
                } else {
                    inverse * worlds[*joint] * *bind
                }
            })
            .collect();
        let morphs = self
            .morphs
            .iter()
            .map(|name| weights.get(name).copied().unwrap_or(0.))
            .collect();
        (matrices, morphs)
    }
}

#[cfg(test)]
mod native_deformation_tests {
    use super::*;
    #[test]
    fn indexing_preserves_corners_seams_and_primitive_order() {
        let vertex = Vertex { POSITION:[1.,2.,3.], NORMAL:[0.,1.,0.],
            TEXCOORD_0:[0.;2], TEXCOORD_1:[0.;2], pad:[0.;2],
            COLOR_0:[1.;4], TANGENT:[0.;4] };
        let mut geometry = Geometry::new(vec![vertex; 6], vec![0,1,2,3,4,5], 1);
        geometry.vertices[2].NORMAL[0] = -0.;
        geometry.vertices[3].TEXCOORD_0[0] = 1.;
        geometry.attributes.insert("custom".into(), vec![[0.;4];6]);
        geometry.attributes.get_mut("custom").unwrap()[4][0] = 1.;
        geometry.primitive_ids = vec![17,9];
        let indexed = geometry.indexed().unwrap();
        assert_eq!(indexed.vertices.len(),4);
        assert_eq!(indexed.indices, [0,0,1,2,3,0]);
        assert_eq!(indexed.primitive_ids, [17,9]);
        for (&before,&after) in geometry.indices.iter().zip(&indexed.indices) {
            assert_eq!(bytemuck::bytes_of(&geometry.vertices[before as usize]),
                bytemuck::bytes_of(&indexed.vertices[after as usize]));
            assert_eq!(geometry.attributes["custom"][before as usize],
                indexed.attributes["custom"][after as usize]);
        }
        assert!(indexed.indexed().is_none());
        // Source identity is significant when retained geometry carries it.
        geometry.vertex_ids = (0..6).collect();
        assert!(geometry.indexed().is_none());
    }
    #[test]
    fn two_joint_palette_and_named_morph_share_the_frame_worlds() {
        let mesh = Mat4::from_translation(Vec3::new(5., 0., 0.));
        let binding = NativeDeform {
            mesh: 0,
            joints: std::sync::Arc::new(vec![
                (1, Mat4::IDENTITY),
                (2, Mat4::from_translation(Vec3::NEG_Y)),
            ]),
            morphs: std::sync::Arc::new(vec!["bulge".into()]),
        };
        let worlds = [
            mesh,
            mesh,
            mesh * Mat4::from_translation(Vec3::Y)
                * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2),
        ];
        let (palette, weights) = binding.palette(&worlds, &[("bulge".into(), 0.75)].into());
        assert_eq!(palette.len(), 2);
        assert!(palette[0].abs_diff_eq(Mat4::IDENTITY, 1e-6));
        assert!(
            palette[1]
                .transform_point3(Vec3::new(0., 2., 0.))
                .abs_diff_eq(Vec3::new(-1., 1., 0.), 1e-6)
        );
        assert_eq!(weights, [0.75]);
        let (_, weights) = binding.palette(&worlds, &Default::default());
        assert_eq!(weights, [0.]);
    }
}

/// Shared glTF/native deformation installation. Bind vertices and all palette inputs
/// are immutable; frame-private outputs remain on the existing deformation path.
pub(crate) fn deformation(
    r: &Renderer,
    asset: Option<usize>,
    node: usize,
    skin: Option<usize>,
    source: &[Vertex],
    mut influences: Vec<Influence>,
    mut morphs: Vec<f32>,
    joint_count: usize,
    sets: usize,
    targets: usize,
) -> Result<Deform> {
    crate::deformation::normalize(&mut influences, source.len(), sets, joint_count)?;
    ensure!(
        targets == 0 || morphs.len() == targets * source.len() * constants::MORPH_FLOATS,
        "morph delta count differs from geometry"
    );
    if morphs.is_empty() {
        morphs.push(0.);
    }
    let mut bounds =
        vec![
            vec![(Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)); 1 + targets];
            joint_count
        ];
    for (i, vertex) in source.iter().enumerate() {
        for influence in &influences[i * sets..(i + 1) * sets] {
            for (&joint, &weight) in influence.JOINTS_0.iter().zip(&influence.WEIGHTS_0) {
                if weight == 0. {
                    continue;
                }
                for (target, interval) in bounds[joint as usize].iter_mut().enumerate() {
                    let value = if target == 0 {
                        Vec3::from(vertex.POSITION)
                    } else {
                        let offset = ((target - 1) * source.len() + i) * constants::MORPH_FLOATS;
                        Vec3::new(morphs[offset], morphs[offset + 1], morphs[offset + 2])
                    };
                    interval.0 = interval.0.min(value);
                    interval.1 = interval.1.max(value);
                }
            }
        }
    }
    Ok(Deform {
        cpu: None,
        native: None,
        job: 0,
        leader: 0,
        bounds: std::sync::Arc::new(bounds),
        asset,
        node,
        skin,
        source: init(
            &r.device,
            "bind vertices",
            bytemuck::cast_slice(source),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        ),
        influences: init(
            &r.device,
            "joint influences",
            bytemuck::cast_slice(&influences),
            wgpu::BufferUsages::STORAGE,
        ),
        morphs: init(
            &r.device,
            "morph deltas",
            bytemuck::cast_slice(&morphs),
            wgpu::BufferUsages::STORAGE,
        ),
        count: u32::try_from(source.len())?,
        targets,
        joints: joint_count,
    })
}
