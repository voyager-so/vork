//! Canonical sparse bricks containing receiver lookup stencils.
//! Allocation is conservative over shading/view normals, including back-side
//! receivers. Residency is not a CPU visibility or safety proof.
use super::*;
use std::collections::{BTreeMap, BTreeSet};
type BrickMap = BTreeMap<(u32, [i32; 3]), BTreeSet<[u32; 3]>>;
#[cfg(test)]
std::thread_local! {
    static FIELD_PACK_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[derive(Default)]
struct Bricks {
    cells: BrickMap,
    device: Option<wgpu::Device>,
    available: std::cell::Cell<Option<u64>>,
    binding_limit:Option<u64>,
}
impl std::ops::Deref for Bricks {
    type Target = BrickMap;
    fn deref(&self) -> &BrickMap { &self.cells }
}
impl std::ops::DerefMut for Bricks {
    fn deref_mut(&mut self) -> &mut BrickMap { &mut self.cells }
}
/// Probe storage beyond what the device can bind or hold now; placement retries coarser on it.
#[derive(Debug)]
pub(crate) struct MemoryPressure(String);
impl std::fmt::Display for MemoryPressure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for MemoryPressure {}
impl Bricks {
    fn new() -> Self { Self::default() }
    fn check_field(&self,nodes:u64,bricks:u64,res:[u32;3])->Result<()> {
        let sizes=field_sizes(nodes,bricks,res);
        if nodes>u64::from(u32::MAX) || sizes.metadata/std::mem::size_of::<Grid>() as u64>u64::from(u32::MAX)
            || self.binding_limit.is_some_and(|limit|!sizes.fits(limit)) {
            return Err(MemoryPressure(crate::gpu_memory::device_limit("probe storage",
                format_args!("{nodes} nodes; {} metadata rows; bindings in bytes: records={}, moments={}, sums={}, metadata={}, traced={}", sizes.metadata/std::mem::size_of::<Grid>() as u64, sizes.records,sizes.moments,sizes.sums,sizes.metadata,sizes.traced),
                format_args!("{} nodes and metadata rows; binding={}",u32::MAX,self.binding_limit.map_or_else(|| "not reported".to_owned(), |bytes| format!("{bytes} bytes"))),
                "reduce geometry density or choose --profile std/draft").0).into());
        }
        self.check(sizes.retained())
    }
    fn check(&self, bytes: u64) -> Result<()> {
        // Enumeration revisits shared cells many times. Native telemetry is
        // refreshed on crossing the observed headroom and before packing, not
        // once per triangle/cell. Allocations also check current native usage.
        if self.available.get().is_some_and(|available| bytes > available) {
            self.check_native(bytes)?;
        }
        Ok(())
    }
    fn check_native(&self, bytes: u64) -> Result<()> {
        let Some(device) = &self.device else { return Ok(()) };
        let mut available = crate::residency::available(device);
        if let Some(headroom) = available {
            if bytes > headroom {
                crate::gpu_memory::ledger(device).evict(bytes - headroom);
                let _ = device.poll(wgpu::PollType::Poll);
                available = crate::residency::available(device);
            }
        }
        self.available.set(available);
        if let Some(headroom) = available.filter(|headroom| bytes > *headroom) {
            return Err(MemoryPressure(crate::gpu_memory::device_limit("probe storage",
                format_args!("{bytes} bytes"), format_args!("{headroom} bytes of native memory headroom"),
                "reduce geometry density or choose --profile std/draft, or close other GPU applications").0).into());
        }
        Ok(())
    }
}

/// Inclusive interval of openings with identical visited spatial decisions.
#[derive(Clone, Copy)]
struct OpeningRange { low: u32, high: u32 }
impl Default for OpeningRange {
    fn default() -> Self { Self { low: 0, high: f32::MAX.to_bits() } }
}
impl OpeningRange {
    fn decision(&mut self, opening: f32, distance: f32, side: f32) {
        if distance <= 0. || !distance.is_finite() { return; }
        let mut threshold = side / distance;
        if !threshold.is_finite() { return; }
        while distance * threshold < side { threshold = f32::from_bits(threshold.to_bits()+1); }
        while threshold > 0. && distance * f32::from_bits(threshold.to_bits()-1) >= side {
            threshold = f32::from_bits(threshold.to_bits()-1);
        }
        if distance * opening >= side { self.low = self.low.max(threshold.to_bits()); }
        else { self.high = self.high.min(threshold.to_bits().saturating_sub(1)); }
    }
}

/// Search distinct support intervals, retaining only the unpacked best candidate.
fn finest_fit<T>(mut place: impl FnMut(f32, &mut OpeningRange) -> Result<T>) -> Result<T> {
    let mut low = 0u32;
    let mut high = f32::MAX.to_bits();
    let mut trial = 0.;
    let mut best = None;
    let mut pressure = None;
    loop {
        let mut range = OpeningRange::default();
        match place(trial, &mut range) {
            Ok(field) => {
                best = Some(field);
                if range.low == 0 { break; }
                high = range.low - 1;
            }
            Err(error) if error.is::<MemoryPressure>() => {
                low = range.high.saturating_add(1);
                pressure = Some(error);
            }
            Err(error) => return Err(error),
        }
        if low > high { break; }
        trial = if best.is_some() {
            f32::from_bits(low + (high-low)/2)
        } else {
            f32::from_bits(low.max(if trial == 0. { constants::PROBE_ANGULAR_SPACING.to_bits() }
                else { (trial*2.).min(f32::MAX).to_bits() }))
        };
    }
    best.ok_or_else(|| pressure.expect("a failed support search encountered memory pressure"))
}

/// An automatic cache is fitted in camera-relative scene units. Remote support
/// adds bricks without changing the local lattice or its sampling identities.
#[derive(Clone, Copy)]
pub(crate) struct Frame {
    origin: Vec3,
    rotation: glam::Quat,
    scale: f32,
    relative: bool,
    clip: Option<[Vec3;8]>,
}
impl Default for Frame {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: 1.,
            relative: false,
            clip: None,
        }
    }
}
impl Frame {
    pub(crate) fn for_camera(
        camera: &crate::install::EvalNode,
        camera_world: Mat4,
        draws: &[install::Draw],
        objects: &[Object],
        render: &vork_scene::Render,
    ) -> Self {
        let (_, rotation, origin) = camera_world.to_scale_rotation_translation();
        let view = Mat4::from_rotation_translation(rotation, origin).inverse();
        let projection = crate::values::camera_projection(
            camera,
            render.resolution[0] as f32 / render.resolution[1] as f32,
        );
        let inverse = projection.inverse();
        let corners = std::array::from_fn(|i| {
            inverse.project_point3(Vec3::new(
                if i & 1 == 0 { -1. } else { 1. },
                if i & 2 == 0 { -1. } else { 1. },
                if i & 4 == 0 { 0. } else { 1. },
            ))
        });
        let bounds = draws
            .iter()
            .flat_map(|draw| draw.instances.clone().map(move |object| (draw, object)))
            .filter(|(_, object)| {
                objects[*object as usize].metadata[1] & constants::RAY_CAMERA != 0
            })
            .map(|(draw, object)| {
                crate::shadow_envelope::transformed(
                    view * objects[object as usize].model,
                    draw.bounds,
                )
            })
            .filter(|&bounds| {
                bounds.0.is_finite()
                    && bounds.1.is_finite()
                    && !crate::shadow_envelope::clipped_bounds(&corners, bounds).is_empty()
            })
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
        let cells = vork_scene::ProbeVolume::default()
            .resolution()
            .into_iter()
            .max()
            .unwrap()
            .saturating_sub(1)
            .max(1) as f32;
        // Computed-only views may have no CPU receiver extent yet. The near
        // clip still supplies scene units and transforms with all other lengths.
        let fallback=camera.camera.as_ref().and_then(|c|c.calibration(render.resolution[0] as f32/render.resolution[1] as f32).ok().flatten()).map_or(1.,|c|c[1][0]);
        let scale = bounds.map_or(fallback, |(lo, hi)| (hi - lo).max_element() / cells);
        let scale=if scale.is_finite() && scale>0. {scale}else{fallback};
        Self {
            origin, rotation, scale,
            relative: true,
            clip: Some(corners.map(|p|p/scale)),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Field {
    cells: BrickMap,
    reference: Frame,
    pub(super) binding_limit:Option<u64>,
    pub(crate) demand: bool,
    pub(crate) header: Grid,
    pub(crate) bricks: Vec<Grid>,
    positions: Vec<Grid>,
    indices: Vec<Grid>,
    frame: Option<Grid>,
    #[cfg(test)]
    pub(crate) placement_ms: f64,
}
impl Field {
    pub(crate) fn in_frame(
        volume: &vork_scene::ProbeVolume,
        draws: &[install::Draw],
        objects: &[Object],
        eyes: &[Vec3],
        demand: bool,
        frame: Frame,
        device: &wgpu::Device,
    ) -> Result<Self> {
        let normalized = Mat4::from_scale(Vec3::splat(1. / frame.scale))
            * Mat4::from_quat(frame.rotation.conjugate())
            * Mat4::from_translation(-frame.origin);
        let objects = objects
            .iter()
            .map(|object| {
                let mut object = *object;
                object.model = normalized * object.model;
                object
            })
            .collect::<Vec<_>>();
        let eyes=eyes.iter().map(|&p|normalized.transform_point3(p)).collect::<Vec<_>>();
        let started = Instant::now();
        let (bricks, nodes) = finest_fit(|opening, range| Self::enumerate(volume, draws, &objects, &eyes,
            opening, demand, frame.clip, Some(device), range))?;
        let field = Self::pack_admitted(volume, bricks, nodes, started)?;
        let mut field = field.in_world(frame);
        field.demand = frame.relative || demand;
        Ok(field)
    }
    fn in_world(mut self, frame: Frame) -> Self {
        let field = &mut self;
        let transform = |g: &mut Grid| {
            let p =
                frame.origin + frame.rotation * (Vec3::from_slice(&g.origin[..3]) * frame.scale);
            g.origin[..3].copy_from_slice(&p.to_array());
            for v in &mut g.spacing {
                *v *= frame.scale;
            }
            g.bias[0] *= frame.scale;
            g.bias[1] *= frame.scale;
        };
        transform(&mut field.header);
        if frame.relative {
            field.header.visibility_policy[1] =
                (1 + field.bricks.len() + field.positions.len() + field.indices.len()) as u32;
            field.frame = Some(Grid {
                origin: frame.origin.extend(0.).to_array(),
                spacing: frame.rotation.to_array(),
                ..Grid::zeroed()
            });
        }
        for g in &mut field.bricks {
            transform(g);
        }
        for g in &mut field.positions {
            let p =
                frame.origin + frame.rotation * (Vec3::from_slice(&g.origin[..3]) * frame.scale);
            g.origin[..3].copy_from_slice(&p.to_array());
        }
        field.reference = frame;
        self
    }
    #[cfg(test)]
    fn place(volume: &vork_scene::ProbeVolume, draws: &[install::Draw], objects: &[Object], eyes: &[Vec3],
        opening: f32, demand: bool, clip: Option<[Vec3;8]>, device: Option<&wgpu::Device>) -> Result<Self> {
        let started=Instant::now();
        let (bricks,nodes)=Self::enumerate(volume,draws,objects,eyes,opening,demand,clip,device,&mut OpeningRange::default())?;
        Self::pack(volume,bricks,nodes,started)
    }
    fn enumerate(
        volume: &vork_scene::ProbeVolume,
        draws: &[install::Draw],
        objects: &[Object],
        eyes: &[Vec3],
        opening: f32,
        demand: bool,
        clip: Option<[Vec3;8]>,
        device: Option<&wgpu::Device>,
        range: &mut OpeningRange,
    ) -> Result<(Bricks, BTreeSet<[i64; 4]>)> {
        let resolution = volume.resolution();
        let mut bricks = Bricks { device: device.cloned(),
            binding_limit:device.map(|d|d.limits().max_storage_buffer_binding_size.min(d.limits().max_buffer_size)),
            available: std::cell::Cell::new(device.and_then(crate::residency::available)), ..Bricks::new() };
        let mut nodes = BTreeSet::new();
        if demand {
            // Bootstrap only a stencil. Actual camera, secondary and capture
            // receivers publish the remaining support; camera far clip and
            // unobserved geometry never reserve probes.
            let eye=eyes.first().copied().unwrap_or(Vec3::ZERO);
            mark_level_cell(resolution,0,(eye/constants::PROBE_SPACING).floor().as_ivec3(),&mut bricks,&mut nodes)?;
            bricks.check_native(resident_bytes(nodes.len(), bricks.len(), resolution))?;
            return Ok((bricks,nodes));
        }
        let mut placements = std::collections::HashSet::new();
        for draw in draws {
            let Some(geometry) = draw.exit_geometry.as_ref() else {
                continue;
            };
            let start = draw.primitive_start as usize * 3;
            for object in draw.instances.clone() {
                let object = &objects[object as usize];
                // Placement is a union of geometry, independent of material or
                // instance identity. Before draw grouping, equal placements
                // can occupy different draw rows. Keep the geometry identity
                // and triangle range in the key, then voxelize each union once.
                if !placements.insert((
                    std::sync::Arc::as_ptr(geometry) as usize,
                    draw.primitive_start,
                    draw.count,
                    object.model.to_cols_array().map(f32::to_bits),
                )) {
                    continue;
                }
                let absolute_model =
                    Mat4::from_cols_array(&object.model.to_cols_array().map(f32::abs));
                for triangle in geometry.indices[start..start + draw.count as usize].chunks_exact(3)
                {
                    let local = [0, 1, 2].map(|i| {
                        Vec3::from_array(geometry.vertices[triangle[i] as usize].POSITION)
                    });
                    let p = local.map(|p| object.model.transform_point3(p));
                    if (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero() == Vec3::ZERO {
                        continue;
                    }
                    // CPU-known visible receivers warm the sparse field.
                    // Off-camera support is admitted only when a ray actually
                    // consumes it; this is an optimization, never visibility proof.
                    if clip.as_ref().is_some_and(|corners|crate::shadow_envelope::clipped_bounds(corners,
                        (p[0].min(p[1]).min(p[2]),p[0].max(p[1]).max(p[2]))).is_empty()) {continue;}
                    let magnitude = p[0].abs().max(p[1].abs()).max(p[2].abs());
                    let products = local.map(|p| absolute_model.transform_point3(p.abs()));
                    let roundoff = vork_slang::registry::FP32_GAMMA7
                        * (magnitude
                            + products[0].max(products[1]).max(products[2])
                            + Vec3::splat(constants::PROBE_SPACING));
                    allocate_triangle(
                        resolution,
                        p,
                        roundoff,
                        eyes,
                        opening,
                        &mut bricks,
                        &mut nodes,
                        range,
                    )?;
                }
            }
        }
        if bricks.is_empty() {
            let eye=eyes.first().copied().unwrap_or(Vec3::ZERO);
            mark_level_cell(resolution,0,(eye/constants::PROBE_SPACING).floor().as_ivec3(),&mut bricks,&mut nodes)?;
        }
        bricks.check_native(resident_bytes(nodes.len(), bricks.len(), resolution))?;
        Ok((bricks, nodes))
    }
    #[cfg(test)]
    fn from_cells(
        volume: &vork_scene::ProbeVolume,
        cells: BTreeSet<[i32; 3]>,
        started: Instant,
    ) -> Result<Self> {
        let res = volume.resolution();
        let mut bricks = Bricks::new();
        let mut nodes = BTreeSet::new();
        for key in cells {
            let corners = bricks.entry((0, key)).or_default();
            for z in 0..res[2] {
                for y in 0..res[1] {
                    for x in 0..res[0] {
                        let local = [x, y, z];
                        corners.insert(local);
                        nodes.insert(level_node_key(res, 0, key, local));
                    }
                }
            }
        }
        Self::pack(volume, bricks, nodes, started)
    }
    fn pack(
        volume: &vork_scene::ProbeVolume,
        cells: Bricks,
        keys: BTreeSet<[i64; 4]>,
        _started: Instant,
    ) -> Result<Self> {
        let res = volume.resolution();
        cells.check_field(keys.len() as u64,cells.len() as u64,res)?;
        cells.check_native(resident_bytes(keys.len(), cells.len(), res))?;
        Self::pack_admitted(volume, cells, keys, _started)
    }
    /// Encode only after admission has selected the final spatial support.
    /// Telemetry checks belong to admission, where pressure can retry the search.
    fn pack_admitted(
        volume: &vork_scene::ProbeVolume,
        cells: Bricks,
        keys: BTreeSet<[i64; 4]>,
        _started: Instant,
    ) -> Result<Self> {
        #[cfg(test)]
        FIELD_PACK_COUNT.with(|count| count.set(count.get()+1));
        let res = volume.resolution();
        let binding_limit=cells.binding_limit;
        let nodes: BTreeMap<_, _> = keys
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, index as u32))
            .collect();
        let positions = nodes
            .keys()
            .map(|key| {
                let p = Vec3::new(key[1] as f32, key[2] as f32, key[3] as f32)
                    * constants::PROBE_SPACING
                    * 0.5
                    * 2f32.powi(key[0] as i32);
                Grid {
                    origin: [p.x, p.y, p.z, 0.],
                    field: [0, 0, 0, key[0] as u32],
                    counts: [key[1] as u32,key[2] as u32,key[3] as u32,key[0] as u32],
                    ..Grid::zeroed()
                }
            })
            .collect::<Vec<_>>();
        let saved_cells = cells.cells.clone();
        let brick_count = cells.len();
        let mut bricks = Vec::with_capacity(brick_count);
        let mut indices = Vec::new();
        for ((level, key), corners) in cells.cells {
            let mut brick = Grid::brick(volume, key);
            brick.field[0]=u32::from(corners.len()==res.iter().product::<u32>() as usize);
            let scale = 2f32.powi(level as i32);
            for axis in 0..3 {
                brick.origin[axis] *= scale;
                brick.spacing[axis] *= scale;
            }
            brick.spacing[3] *= scale;
            brick.bias[0] *= scale;
            brick.bias[1] *= scale;
            brick.field[3] = level;
            brick.field[1] = (1 + brick_count + positions.len() + indices.len()) as u32;
            let mut words = vec![u32::MAX; res.iter().product::<u32>() as usize];
            for local in corners {
                let index = local[0] + res[0] * (local[1] + res[1] * local[2]);
                words[index as usize] = nodes[&level_node_key(res, level, key, local)];
            }
            for chunk in words.chunks(4) {
                let mut record = Grid::zeroed();
                record.field = [u32::MAX; 4];
                record.field[..chunk.len()].copy_from_slice(chunk);
                indices.push(record);
            }
            bricks.push(brick);
        }
        let mut header = Grid::brick(volume, [0; 3]);
        header.field = [
            bricks.len() as u32,
            (1 + bricks.len()) as u32,
            positions.len() as u32,
            bricks.iter().map(|b| b.field[3]).max().unwrap_or(0),
        ];
        Ok(Self {
            cells: saved_cells, reference: Frame::default(), binding_limit, demand: false,
            header,
            bricks,
            positions,
            indices,
            frame: None,
            #[cfg(test)]
            placement_ms: _started.elapsed().as_secs_f64() * 1000.,
        })
    }
    pub(crate) fn coarsen(&self) -> Result<Self> {
        let level=self.header.field[3];
        self.grow_with(&BTreeSet::new(),level,false,level.checked_add(1).context("probe spatial level overflow")?)
    }
    pub(crate) fn grow_bricks(&self,requests:&BTreeSet<[i32;3]>,level:u32)->Result<Self> {
        self.grow_with(requests,level,true,level)
    }
    #[cfg(test)]
    pub(crate) fn grow(&self,requests:&BTreeSet<[i32;3]>,level:u32)->Result<Self> {
        self.grow_with(requests,level,false,level)
    }
    fn grow_with(&self,requests:&BTreeSet<[i32;3]>,level:u32,bricks:bool,mut target:u32)->Result<Self> {
        loop {
            ensure!(target<127,crate::gpu_memory::device_limit("probe spatial level under device pressure", target, 126, "reduce geometry density or choose --profile std/draft"));
            match self.grow_at_level(requests,level,target,bricks) {
                Err(error) if error.is::<MemoryPressure>()=> {
                    target=target.max(self.header.field[3]).checked_add(1).context("probe spatial level overflow")?;
                },
                result=>return result,
            }
        }
    }
    fn grow_at_level(&self,requests:&BTreeSet<[i32;3]>,level:u32,target:u32,bricks:bool)->Result<Self> {
        let volume = vork_scene::ProbeVolume { resolution_x:self.header.counts[0], resolution_y:self.header.counts[1], resolution_z:self.header.counts[2], rays_per_probe: self.header.counts[3] };
        let res = self.header.counts[..3].try_into().unwrap();
        let mut cells = Bricks { cells:if target==level {self.cells.clone()} else {BrickMap::new()},binding_limit:self.binding_limit,..Bricks::new() };
        let mut nodes = BTreeSet::new();
        for (&(level, brick), corners) in &cells.cells {
            for &corner in corners { nodes.insert(level_node_key(res, level, brick, corner)); }
        }
        // Expand one requested brick at a time, so binding admission precedes
        // construction of a potentially enormous intermediate cell set.
        let steps=if bricks {res.map(|n|n.saturating_sub(1).max(1))} else {[1;3]};
        let requested_cells=||requests.iter().flat_map(|key| {
            (0..steps.iter().product::<u32>()).map(move |i| {
                let offset=[i%steps[0],(i/steps[0])%steps[1],i/(steps[0]*steps[1])];
                std::array::from_fn(|axis|key[axis]*steps[axis] as i32+offset[axis] as i32)
            })
        });
        if target==level {
            for cell in requested_cells() {mark_level_cell(res,level,glam::IVec3::from_array(cell),&mut cells,&mut nodes)?;}
        } else {
            // Re-enumerate the complete old support and every pending request at
            // the coarser native-admissible level. No partial prefix survives.
            for (&(old_level,brick),corners) in &self.cells {
                for &corner in corners {
                    mark_coarse_corner(res,level_node_key(res,old_level,brick,corner),target,&mut cells,&mut nodes)?;
                }
            }
            for cell in requested_cells() {
                for z in 0..2 {for y in 0..2 {for x in 0..2 {
                    let offset=[x,y,z];
                    let key=[i64::from(level),
                        2*i64::from(cell[0])+if res[0]==1 {1}else{2*offset[0]},
                        2*i64::from(cell[1])+if res[1]==1 {1}else{2*offset[1]},
                        2*i64::from(cell[2])+if res[2]==1 {1}else{2*offset[2]}];
                    mark_coarse_corner(res,key,target,&mut cells,&mut nodes)?;
                }}}
            }
        }
        let mut field = Self::pack(&volume, cells, nodes, Instant::now())?.in_world(self.reference);
        field.header.visibility_policy[0] = self.header.visibility_policy[0];
        field.header.visibility_policy[3] = self.header.visibility_policy[3];
        field.demand = self.demand;
        Ok(field)
    }

    pub(crate) fn metadata(&self) -> Vec<Grid> {
        std::iter::once(self.header)
            .chain(self.bricks.iter().copied())
            .chain(self.positions.iter().copied())
            .chain(self.indices.iter().copied())
            .chain(self.frame)
            .collect()
    }
}
fn allocate_triangle(
    resolution: [u32; 3],
    p: [Vec3; 3],
    roundoff: Vec3,
    eyes: &[Vec3],
    opening: f32,
    bricks: &mut Bricks,
    nodes: &mut BTreeSet<[i64; 4]>,
    range: &mut OpeningRange,
) -> Result<()> {
    let lo = p[0].min(p[1]).min(p[2]);
    let hi = p[0].max(p[1]).max(p[2]);
    ensure!(
        lo.is_finite()
            && hi.is_finite()
            && (lo / constants::PROBE_SPACING).abs().max_element() < (i32::MAX / 2) as f32
            && (hi / constants::PROBE_SPACING).abs().max_element() < (i32::MAX / 2) as f32,
        "settings.probes.volume: receiver exceeds finite spatial address range"
    );
    // Start above the triangle's extent, including negative cells.
    // Descend only intersected boxes; huge ground triangles never
    // enumerate their fine-resolution bounding rectangle.
    let distance = eyes
        .iter()
        .map(|eye| (*eye - eye.clamp(lo, hi)).length())
        .fold(f32::INFINITY, f32::min);
    let extent = (hi - lo).max_element().max(distance * opening);
    let level = (extent / constants::PROBE_SPACING).max(1.).log2().ceil() as u32;
    // Root enumeration changes only when the rounded extent crosses a power
    // of two. Find that interval cheaply, without enumerating support.
    if opening > 0. {
        let root = |o: f32| ((hi-lo).max_element().max(distance*o) / constants::PROBE_SPACING).max(1.).log2().ceil() as u32;
        let bits = opening.to_bits();
        let mut a=0; let mut b=bits;
        while a<b { let m=a+(b-a)/2; if root(f32::from_bits(m))<level {a=m+1;} else {b=m;} }
        range.low=range.low.max(a);
        let mut a=bits; let mut b=f32::MAX.to_bits();
        while a<b { let m=a+(b-a)/2+1; if root(f32::from_bits(m))>level {b=m-1;} else {a=m;} }
        range.high=range.high.min(a);
    }
    // The halo policy distinguishes zero from positive openings.
    if opening == 0. { range.high=0; } else { range.low=range.low.max(1); }
    let side = constants::PROBE_SPACING * 2f32.powi(level as i32);
    ensure!(side.is_finite(), "probe cell exceeds finite spatial range");
    let pad =
        Vec3::splat((constants::PROBE_NORMAL_BIAS + constants::PROBE_VIEW_BIAS) * side) + roundoff;
    let first = ((lo - pad) / side).floor().as_ivec3();
    let last = ((hi + pad) / side).floor().as_ivec3();
    let mut pending = Vec::new();
    for z in first.z..=last.z {
        for y in first.y..=last.y {
            for x in first.x..=last.x {
                pending.push((level, glam::IVec3::new(x, y, z)));
            }
        }
    }
    while let Some((level, cell)) = pending.pop() {
        let side = constants::PROBE_SPACING * 2f32.powi(level as i32);
        let lower = cell.as_vec3() * side;
        let upper = lower + side;
        let pad = Vec3::splat((constants::PROBE_NORMAL_BIAS + constants::PROBE_VIEW_BIAS) * side)
            + roundoff;
        if !triangle_box(p, (lower + upper) * 0.5, Vec3::splat(side * 0.5) + pad) {
            continue;
        }
        let distance = eyes
            .iter()
            .map(|eye| (*eye - eye.clamp(lower, upper)).length())
            .fold(f32::INFINITY, f32::min);
        if level > 0 && opening > 0. { range.decision(opening, distance, side); }
        if level == 0 || side <= distance * opening {
            mark_level_cell(resolution, level, cell, bricks, nodes)?;
            if opening > 0. {
                // Bias can cross a distance-shell boundary. Allocate its halo
                // at this same level, independently of the neighbour's LOD,
                // so a receiver always has a complete biased stencil.
                let halo = (pad / side).ceil().as_ivec3();
                for z in -halo.z..=halo.z {
                    for y in -halo.y..=halo.y {
                        for x in -halo.x..=halo.x {
                            let neighbour = cell + glam::IVec3::new(x, y, z);
                            let centre = (neighbour.as_vec3() + 0.5) * side;
                            if triangle_box(p, centre, Vec3::splat(side * 0.5) + pad) {
                                mark_level_cell(resolution, level, neighbour, bricks, nodes)?;
                            }
                        }
                    }
                }
            }
        } else {
            for z in 0..2 {
                for y in 0..2 {
                    for x in 0..2 {
                        pending.push((level - 1, cell * 2 + glam::IVec3::new(x, y, z)));
                    }
                }
            }
        }
    }
    Ok(())
}
// Unit cells name the needed trilinear corners; bricks only pack their shared
// addresses. Unneeded vertices are not resident and never consume capture work.
#[cfg(test)]
fn mark_cell(
    res: [u32; 3],
    cell: glam::IVec3,
    bricks: &mut Bricks,
    nodes: &mut BTreeSet<[i64; 4]>,
) -> Result<()> {
    mark_level_cell(res, 0, cell, bricks, nodes)
}
fn mark_level_cell(
    res: [u32; 3],
    level: u32,
    cell: glam::IVec3,
    bricks: &mut Bricks,
    nodes: &mut BTreeSet<[i64; 4]>,
) -> Result<()> {
    let step = res.map(|n| n.saturating_sub(1).max(1) as i32);
    let key = std::array::from_fn(|i| cell[i].div_euclid(step[i]));
    let base = std::array::from_fn::<_, 3, _>(|i| cell[i].rem_euclid(step[i]) as u32);
    let corners = bricks.entry((level, key)).or_default();
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let o = [x, y, z];
                let local = std::array::from_fn(|i| if res[i] == 1 { 0 } else { base[i] + o[i] });
                corners.insert(local);
                nodes.insert(level_node_key(res, level, key, local));
            }
        }
    }
    bricks.check_field(nodes.len() as u64,bricks.len() as u64,res)?;
    Ok(())
}

fn mark_coarse_corner(res:[u32;3],node:[i64;4],target:u32,bricks:&mut Bricks,nodes:&mut BTreeSet<[i64;4]>)->Result<()> {
    let scale=2f64.powi(node[0] as i32-target as i32-1);
    let cell=glam::IVec3::from_array(std::array::from_fn(|i|(node[i+1] as f64*scale).floor() as i32));
    let halo=(constants::PROBE_NORMAL_BIAS+constants::PROBE_VIEW_BIAS).ceil() as i32;
    for z in -halo..=halo {for y in -halo..=halo {for x in -halo..=halo {
        mark_level_cell(res,target,cell+glam::IVec3::new(x,y,z),bricks,nodes)?;
    }}}
    Ok(())
}
fn level_node_key(res: [u32; 3], level: u32, brick: [i32; 3], local: [u32; 3]) -> [i64; 4] {
    let key = node_key(res, brick, local);
    [i64::from(level), key[0], key[1], key[2]]
}
fn node_key(res: [u32; 3], brick: [i32; 3], local: [u32; 3]) -> [i64; 3] {
    std::array::from_fn(|i| {
        i64::from(brick[i]) * i64::from(res[i].saturating_sub(1).max(1)) * 2
            + i64::from(local[i]) * 2
            + i64::from(res[i] == 1)
    })
}
fn field_sizes(nodes:u64,bricks:u64,res:[u32;3])->StorageSizes {
    // Header and optional camera-relative record, plus directory and positions.
    let metadata=nodes.saturating_add(bricks.saturating_mul(1+u64::from(res.iter().product::<u32>().div_ceil(4)))).saturating_add(2)
        .saturating_mul(std::mem::size_of::<Grid>() as u64);
    StorageSizes::new(nodes,bricks,metadata,2,1)
}
#[cfg(test)]
fn bindings_fit(nodes:usize,bricks:usize,res:[u32;3],limit:u64)->bool {field_sizes(nodes as u64,bricks as u64,res).fits(limit)}
fn resident_bytes(nodes:usize,bricks:usize,res:[u32;3])->u64 {field_sizes(nodes as u64,bricks as u64,res).retained()}

// Triangle/AABB separating axes. Conservative closed boundaries keep equivalent
// triangulations in the same cells; no material or visibility knowledge lives here.
fn triangle_box(p: [Vec3; 3], centre: Vec3, half: Vec3) -> bool {
    let p = p.map(|v| v - centre);
    let edges = [p[1] - p[0], p[2] - p[1], p[0] - p[2]];
    let separated = |axis: Vec3| {
        let d = p.map(|p| p.dot(axis));
        let radius = half.dot(axis.abs());
        d.into_iter().fold(f32::INFINITY, f32::min) > radius
            || d.into_iter().fold(f32::NEG_INFINITY, f32::max) < -radius
    };
    if separated(Vec3::X)
        || separated(Vec3::Y)
        || separated(Vec3::Z)
        || separated(edges[0].cross(edges[1]))
    {
        return false;
    }
    !edges.into_iter().any(|e| {
        [Vec3::X, Vec3::Y, Vec3::Z]
            .into_iter()
            .any(|axis| separated(e.cross(axis)))
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn exhausted_support_preserves_the_sized_limit_message() {
        let error = finest_fit::<()>(|_, _| {
            let field = Bricks {binding_limit: Some(128), ..Bricks::new()};
            field.check_field(64, 1, [4;3])?;
            unreachable!("the test field cannot fit the binding")
        }).unwrap_err().to_string();
        for expected in ["64 nodes", "binding=128 bytes", "reduce geometry density", "vork help limits"] {
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn pressure_search_refines_instead_of_accepting_the_first_coarse_jump() -> Result<()> {
        for threshold in [0., 1./512., 1./64., 0.249, 0.5] {
            let mut attempts=Vec::new();
            let accepted=finest_fit(|opening, range| {
                attempts.push(opening);
                range.decision(opening, 1., threshold);
                if opening < threshold {Err(MemoryPressure("synthetic memory pressure".into()).into())} else {Ok(threshold)}
            })?;
            assert_eq!(accepted,threshold);
            assert_eq!(attempts[0],0.);
            assert!(attempts.len() <= 3, "identical support must not be enumerated again: {attempts:?}");
            if threshold==0. {assert_eq!(attempts.len(),1);}
        }
        let mut attempts=0;
        assert!(finest_fit::<()>(|_, _|{attempts+=1;anyhow::bail!("bad geometry")}).is_err());
        assert_eq!(attempts,1);
        Ok(())
    }
    #[test]
    fn missing_telemetry_stops_enumeration_at_binding_capacity_and_retries() -> Result<()> {
        for available in [None,Some(u64::MAX)] {
            let mut attempts=0;
            let accepted=finest_fit(|opening, range| {
                attempts+=1;
                let mut bricks=Bricks {binding_limit:Some(1<<20), available:std::cell::Cell::new(available), ..Bricks::new()};
                let mut nodes=BTreeSet::new();
                let result=allocate_triangle([4;3], [Vec3::new(-100.,0.,-100.),Vec3::new(100.,0.,-100.),Vec3::new(0.,0.,100.)],
                    Vec3::ZERO, &[Vec3::new(0.,200.,0.)], opening, &mut bricks,&mut nodes, range);
                if result.is_err() {
                    assert!(!bindings_fit(nodes.len(),bricks.len(),[4;3],1<<20));
                    assert!(bindings_fit(nodes.len().saturating_sub(8),bricks.len().saturating_sub(1),[4;3],1<<20),
                        "enumeration stops at the first oversized stencil");
                }
                result?;
                assert!(bindings_fit(nodes.len(),bricks.len(),[4;3],1<<20));
                Ok(opening)
            })?;
            assert!(attempts>2 && accepted>0.);
        }
        Ok(())
    }
    #[test]
    fn pressure_install_enumerates_distinct_support_and_packs_once() -> Result<()> {
        FIELD_PACK_COUNT.with(|count| count.set(0));
        let started=Instant::now();
        let mut enumerations=0;
        let mut signatures=BTreeSet::new();
        let volume=vork_scene::ProbeVolume::default();
        let (bricks,nodes)=finest_fit(|opening,range| {
            enumerations+=1;
            let mut bricks=Bricks { binding_limit:Some(1<<20), ..Bricks::new() };
            let mut nodes=BTreeSet::new();
            let result=allocate_triangle(volume.resolution(),
                [Vec3::new(-100.,0.,-100.),Vec3::new(100.,0.,-100.),Vec3::new(0.,0.,100.)],
                Vec3::ZERO,&[Vec3::new(0.,200.,0.)],opening,&mut bricks,&mut nodes,range);
            // Zero's halo policy differs from every positive opening; otherwise
            // each trial must cross a recorded spatial decision.
            assert!(signatures.insert((opening==0.,range.low,range.high)));
            result?;
            Ok((bricks,nodes))
        })?;
        let field=Field::pack(&volume,bricks,nodes,started)?;
        assert!(enumerations<20,"pressure enumeration count: {enumerations}");
        assert!(bindings_fit(field.positions.len(),field.bricks.len(),volume.resolution(),1<<20));
        let packs=FIELD_PACK_COUNT.with(|count| count.get());
        assert_eq!(packs,1,"only the admitted field may be encoded");
        eprintln!("pressure install: enumerations={enumerations} packs={packs} nodes={} bricks={} wall_ms={:.3}",
            field.positions.len(),field.bricks.len(),started.elapsed().as_secs_f64()*1000.);
        Ok(())
    }
    #[test]
    fn pressure_intervals_retain_the_adjacent_float_search_support() -> Result<()> {
        let enumerate=|opening, range: &mut OpeningRange| {
            let mut bricks=Bricks { binding_limit:Some(1<<20), ..Bricks::new() };
            let mut nodes=BTreeSet::new();
            allocate_triangle([4;3],
                [Vec3::new(-100.,0.,-100.),Vec3::new(100.,0.,-100.),Vec3::new(0.,0.,100.)],
                Vec3::ZERO,&[Vec3::new(0.,200.,0.)],opening,&mut bricks,&mut nodes,range)?;
            Ok((bricks,nodes))
        };
        let selected=finest_fit(enumerate)?;
        // Reference the former adjacent-float search independently of the
        // intervals. Compare spatial identities, not just allocation counts.
        let (mut low,mut high)=(0f32,constants::PROBE_ANGULAR_SPACING);
        let mut reference=loop {
            match enumerate(high,&mut OpeningRange::default()) {
                Ok(field)=>break field,
                Err(error) if error.is::<MemoryPressure>()=>{low=high;high*=2.;},
                Err(error)=>return Err(error),
            }
        };
        while high.to_bits()-low.to_bits()>1 {
            let mid=f32::from_bits(low.to_bits()+(high.to_bits()-low.to_bits())/2);
            match enumerate(mid,&mut OpeningRange::default()) {
                Ok(field)=>{high=mid;reference=field;},
                Err(error) if error.is::<MemoryPressure>()=>low=mid,
                Err(error)=>return Err(error),
            }
        }
        assert_eq!(selected.0.cells,reference.0.cells);
        assert_eq!(selected.1,reference.1);
        Ok(())
    }
    #[test]
    fn opening_ranges_preserve_rounded_spatial_decisions() {
        for distance in [0.01,1.,3.,200.,1e20] {
            for side in [0.1,1.,16.,1024.] {
                for opening in [0.,0.001,0.249,0.5,1.,100.] {
                    let mut range=OpeningRange::default();
                    range.decision(opening,distance,side);
                    let accepts=distance*opening>=side;
                    assert_eq!(distance*f32::from_bits(range.low)>=side,accepts);
                    assert_eq!(distance*f32::from_bits(range.high)>=side,accepts);
                    if accepts && range.low>0 {assert!(distance*f32::from_bits(range.low-1)<side);}
                    if !accepts && range.high<f32::MAX.to_bits() {assert!(distance*f32::from_bits(range.high+1)>=side);}
                }
            }
        }
    }
    #[test]
    fn demand_growth_streams_and_retries_with_the_original_binding_limit() -> Result<()> {
        let mut initial=Field::from_cells(&vork_scene::ProbeVolume::default(),BTreeSet::from([[0,0,0]]),Instant::now())?;
        initial.binding_limit=Some(1<<20);
        let requests=(1..256).map(|x|[x*1000,0,0]).collect();
        let grown=initial.grow_bricks(&requests,0)?;
        assert!(grown.header.field[3]>0);
        assert_eq!(grown.binding_limit,initial.binding_limit);
        assert!(bindings_fit(grown.positions.len(),grown.bricks.len(),[4;3],1<<20));
        let level=grown.header.field[3];
        for key in requests {
            let cell=glam::IVec3::from_array(key).as_vec3()*3./2f32.powi(level as i32);
            let cell=cell.floor().as_ivec3();
            let brick=std::array::from_fn(|i|cell[i].div_euclid(3));
            let local=std::array::from_fn::<_,3,_>(|i|cell[i].rem_euclid(3) as u32);
            let corners=&grown.cells[&(level,brick)];
            for z in 0..2 {for y in 0..2 {for x in 0..2 {
                assert!(corners.contains(&[local[0]+x,local[1]+y,local[2]+z]),"every requested region retains a complete stencil");
            }}}
        }
        Ok(())
    }

    #[test]
    fn enumeration_checks_bindings_without_budget_telemetry()->Result<()> {
        let mut bricks=Bricks {binding_limit:Some(64<<10),..Bricks::new()};
        let mut nodes=BTreeSet::new();
        for x in 0..1000 {
            match mark_cell([4;3],glam::IVec3::new(x*6,0,0),&mut bricks,&mut nodes) {
                Err(error)=> {
                    assert!(error.is::<MemoryPressure>());
                    assert!(bricks.len()<1000,"admission waited until packing");
                    assert_eq!(bricks.available.get(),None);
                    return Ok(());
                },
                Ok(())=>{},
            }
        }
        panic!("native binding pressure was not detected during enumeration");
    }

    #[test]
    fn native_binding_capacity_is_independent_of_total_headroom() {
        let resolution = [4; 3];
        assert!(!bindings_fit(246027, 8192, resolution, 128 << 20));
        assert!(bindings_fit(246027, 8192, resolution, 1 << 30));
        assert!(bindings_fit(8192, 1024, resolution, 128 << 20));
    }

    #[test]
    fn computed_demand_retries_bindings_without_enumerating_the_clip()->Result<()> {
        let volume=vork_scene::ProbeVolume::default();
        let clip=std::array::from_fn(|i|Vec3::new(if i&1==0 {-2000.}else{2000.},if i&2==0 {-2000.}else{2000.},if i&4==0 {-2000.}else{2000.}));
        let mut initial=Field::place(&volume,&[],&[],&[Vec3::ZERO],0.,true,Some(clip),None)?;
        assert_eq!(initial.header.count(),8,"computed clip must bootstrap only one stencil");
        initial.binding_limit=Some(16<<20);
        let mut requests=BTreeSet::new();
        for x in 0..64 {for z in 0..64 {requests.insert([x*4,0,z*4]);}}
        requests.insert([-2000,2000,2000]);
        assert!(initial.grow_at_level(&requests,0,0,false).err().unwrap().is::<MemoryPressure>());
        let field=initial.grow(&requests,0)?;
        assert!(field.header.field[3]>0,"native pressure must retry at a coarser level");
        assert!(field_sizes(field.positions.len() as u64,field.bricks.len() as u64,[4;3]).fits(16<<20));
        let metadata=field.metadata();
        for world in [Vec3::splat(0.25),Vec3::new(252.25,0.25,252.25),Vec3::new(-1999.75,2000.25,2000.25)] {
            for normal in [Vec3::Y,-Vec3::Y,Vec3::X] {
                assert!(field.bricks.iter().any(|g| {
                    let p=world+normal*g.bias[0]+Vec3::Z*g.bias[1];
                    let origin=Vec3::from_slice(&g.origin[..3]);let side=g.spacing[0];
                    if (p/(side*3.)).floor()!=(origin/(side*3.)).floor() {return false;}
                    let base=((p-origin)/side).floor().as_ivec3().clamp(glam::IVec3::ZERO,glam::IVec3::splat(2));
                    (0..8).all(|i| {let q=base+glam::IVec3::new(i&1,(i>>1)&1,(i>>2)&1);let at=(q.x+4*(q.y+4*q.z)) as usize;metadata[g.field[1] as usize+at/4].field[at%4]!=u32::MAX})
                }),"native retry lost complete biased support at {world:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn demand_adds_only_complete_requested_stencils_and_preserves_the_frame() -> Result<()> {
        let volume=vork_scene::ProbeVolume::default();
        let frame=Frame {origin:Vec3::new(10.,20.,30.),rotation:glam::Quat::from_rotation_y(0.7),scale:0.25,relative:true,clip:None};
        let initial=Field::from_cells(&volume,BTreeSet::from([[0,0,0]]),Instant::now())?.in_world(frame);
        let requests=BTreeSet::from([[-2000,3000,4000]]);
        let grown=initial.grow(&requests,0)?;
        assert_eq!(grown.header.count(),initial.header.count()+8);
        assert_eq!(grown.header.field[3],0);
        assert_eq!(grown.frame,initial.frame);
        assert_eq!(grown.positions.iter().filter(|p|initial.positions.contains(p)).count(),initial.positions.len());
        let repeated=grown.grow(&requests,0)?;
        assert_eq!(repeated.metadata(),grown.metadata());
        Ok(())
    }

    #[test]
    fn resident_field_can_exceed_the_former_byte_ceiling() -> Result<()> {
        let mut bricks = Bricks::new();
        let mut nodes = BTreeSet::new();
        for x in 0..8192 {
            mark_cell([4;3], glam::IVec3::new(x*6,0,0), &mut bricks, &mut nodes)?;
        }
        let field = Field::pack(&vork_scene::ProbeVolume::default(),bricks,nodes,Instant::now())?;
        assert!(resident_bytes(field.positions.len(),field.bricks.len(),[4;3])>64<<20);
        assert_eq!(field.header.field[3],0,"capacity must not coarsen the lattice");
        assert_eq!(field.header.spacing[..3],[constants::PROBE_SPACING;3]);
        Ok(())
    }

    use super::*;
    #[test]
    fn open_terrain_keeps_near_detail_and_complete_far_support() -> Result<()> {
        let mut bricks = Bricks::new();
        let mut nodes = BTreeSet::new();
        let triangle = [
            Vec3::new(-10000., 0., -10000.),
            Vec3::new(10000., 0., -10000.),
            Vec3::new(0., 0., 10000.),
        ];
        let eyes = [Vec3::new(0., 2., 0.)];
        allocate_triangle([4; 3], triangle, Vec3::ZERO, &eyes,
            constants::PROBE_ANGULAR_SPACING, &mut bricks, &mut nodes, &mut OpeningRange::default())?;
        assert!(bricks.keys().any(|(level, _)| *level == 0));
        assert!(bricks.keys().any(|(level, _)| *level >= 8));
        let field = Field::pack(
            &vork_scene::ProbeVolume::default(),
            bricks,
            nodes,
            Instant::now(),
        )?;
        assert_eq!(field.header.spacing[..3], [constants::PROBE_SPACING; 3]);
        let metadata = field.metadata();
        for x in [-4000., -40., 0., 40., 4000.] {
            for z in [-4000., -40., 0., 40.] {
                let world = Vec3::new(x, 0., z);
                for normal in [Vec3::Y, -Vec3::Y, Vec3::X] {
                    // Mirror the shader's finest complete biased-stencil search.
                    let mut found = None;
                    for g in &field.bricks {
                        let p = world + normal * g.bias[0] + Vec3::Z * g.bias[1];
                        let side = g.spacing[0];
                        let key = (p / (side * 3.)).floor();
                        let origin = Vec3::from_slice(&g.origin[..3]);
                        if key != (origin / (side * 3.)).floor() {
                            continue;
                        }
                        let base = ((p - origin) / side)
                            .floor()
                            .as_ivec3()
                            .clamp(glam::IVec3::ZERO, glam::IVec3::splat(2));
                        let complete = (0..8).all(|i| {
                            let q = base + glam::IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
                            let at = (q.x + 4 * (q.y + 4 * q.z)) as usize;
                            metadata[g.field[1] as usize + at / 4].field[at % 4] != u32::MAX
                        });
                        if complete {
                            found = Some(g);
                            break;
                        }
                    }
                    let g = found.expect("every terrain receiver keeps a complete stencil");
                    if x == 0. && z == 0. {
                        assert_eq!(g.spacing[0], constants::PROBE_SPACING);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn shipped_profiles_disable_probe_capture() -> Result<()> {
        for name in ["std/previs", "std/draft", "std/final", "std/path"] {
            let profile = vork_scene::Profile::load(name)?;
            assert!(
                profile
                    .resolve(&Default::default())?
                    .integrator
                    .probe_volume()
                    .is_none(),
                "{name}"
            );
        }
        Ok(())
    }

    #[test]
    fn canonical_support_does_not_stretch_with_distant_geometry() -> Result<()> {
        let volume = vork_scene::ProbeVolume::default();
        let local = Field::from_cells(&volume, BTreeSet::from([[0, 0, 0]]), Instant::now())?;
        let distant = Field::from_cells(
            &volume,
            BTreeSet::from([[0, 0, 0], [1000, 0, 1000]]),
            Instant::now(),
        )?;
        assert_eq!(local.bricks[0].origin, distant.bricks[0].origin);
        assert_eq!(local.bricks[0].spacing, distant.bricks[0].spacing);
        assert_eq!(local.bricks[0].counts, distant.bricks[0].counts);
        assert_eq!(local.header.count() * 2, distant.header.count());
        assert_eq!(local.bricks[0].spacing[..3], [constants::PROBE_SPACING; 3]);
        Ok(())
    }
    #[test]
    fn capture_ray_and_emission_budgets_cannot_evict_resident_support() -> Result<()> {
        let mut volume = vork_scene::ProbeVolume::default();
        let low = Field::from_cells(
            &volume,
            BTreeSet::from([[0, 0, 0], [1, 0, 0]]),
            Instant::now(),
        )?;
        volume.rays_per_probe = 65536;
        let mut high = Field::from_cells(
            &volume,
            BTreeSet::from([[0, 0, 0], [1, 0, 0]]),
            Instant::now(),
        )?;
        high.header.visibility_policy[3] = 1;
        assert_eq!(low.positions, high.positions);
        assert_eq!(low.indices, high.indices);
        assert_eq!(low.header.count(), high.header.count());
        let binding_limit = 1 << 20;
        assert!(high.header.batch(binding_limit) < volume.rays_per_probe);
        assert!(
            u64::from(high.header.count())
                * u64::from(high.header.batch(binding_limit))
                * 16
                * high.header.trace_stride()
                <= binding_limit
        );
        Ok(())
    }
    #[test]
    fn receiver_cells_allocate_only_their_stencil_and_share_negative_boundaries() -> Result<()> {
        let res = [4; 3];
        let mut bricks = Bricks::new();
        let mut nodes = BTreeSet::new();
        mark_cell(res, glam::IVec3::splat(-1), &mut bricks, &mut nodes)?;
        let field = Field::pack(
            &vork_scene::ProbeVolume::default(),
            bricks,
            nodes,
            Instant::now(),
        )?;
        assert_eq!(
            field.header.count(),
            8,
            "one receiver cell needs eight vertices, not an entire brick"
        );
        assert!(
            field
                .positions
                .iter()
                .all(|g| g.origin[..3].iter().all(|&v| v == -1. || v == 0.))
        );
        let metadata = field.metadata();
        let start = field.bricks[0].field[1] as usize;
        assert_eq!(
            (0..64)
                .filter(|i| metadata[start + i / 4].field[i % 4] != u32::MAX)
                .count(),
            8
        );
        Ok(())
    }
    #[test]
    fn adjacent_bricks_capture_shared_vertices_once() -> Result<()> {
        let field = Field::from_cells(
            &vork_scene::ProbeVolume::default(),
            BTreeSet::from([[0, 0, 0], [1, 0, 0]]),
            Instant::now(),
        )?;
        assert_eq!(field.header.count(), 2 * 64 - 16);
        let metadata = field.metadata();
        for z in 0..4 {
            for y in 0..4 {
                let left = 3 + 4 * (y + 4 * z);
                let right = 4 * (y + 4 * z);
                let index = |brick: usize, local: usize| {
                    metadata[field.bricks[brick].field[1] as usize + local / 4].field[local % 4]
                };
                assert_eq!(index(0, left), index(1, right));
            }
        }
        assert!(
            resident_bytes(field.positions.len(), field.bricks.len(), [4; 3])
                < resident_bytes(128, 2, [4; 3])
        );
        Ok(())
    }
    #[test]
    fn triangle_brick_allocation_is_independent_of_triangulation() {
        let a = Vec3::new(-4., 0.5, -4.);
        let b = Vec3::new(4., 0.5, -4.);
        let c = Vec3::new(4., 0.5, 4.);
        let d = Vec3::new(-4., 0.5, 4.);
        for z in -3..3 {
            for y in -1..2 {
                for x in -3..3 {
                    let centre = Vec3::new(x as f32, y as f32, z as f32) * 3. + 1.5;
                    let half = Vec3::splat(1.5);
                    let first = triangle_box([a, b, c], centre, half)
                        || triangle_box([a, c, d], centre, half);
                    let split = triangle_box([a, b, d], centre, half)
                        || triangle_box([b, c, d], centre, half);
                    assert_eq!(first, split, "canonical brick {x}/{y}/{z}");
                }
            }
        }
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn automatic_frame_is_covariant_without_cpu_receiver_bounds() -> Result<()> {
        let r = crate::test_renderer()?;
        let scene:vork_scene::Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[
            {"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"clip_start":0.1,"clip_end":40}}},
            {"type":"camera","name":"scaled","material":{"program":"std/camera","inputs":{"clip_start":0.4,"clip_end":160}}}]}))?;
        let mut nodes=scene.nodes.iter().map(|node|crate::install::EvalNode {node:node.clone(),camera:None,parent:None,world:Mat4::IDENTITY,dynamic:false,embedded:None}).collect::<Vec<_>>();
        let views=nodes.iter().map(|n|(n.node.name.clone(),vec![scene.render.resolution])).collect();
        crate::camera_program::install(&r,&scene,&mut nodes,std::path::Path::new("."),&views)?;
        let camera=&nodes[0];let scaled=&nodes[1];
        let render=vork_scene::Render::default();
        let a=Frame::for_camera(&camera,Mat4::IDENTITY,&[],&[],&render);
        let b=Frame::for_camera(&scaled,Mat4::from_translation(Vec3::new(10.,-3.,7.)),&[],&[],&render);
        assert_eq!(b.scale,a.scale*4.);
        let moved=Frame::for_camera(&camera,Mat4::from_rotation_y(0.7)*Mat4::from_translation(Vec3::new(5.,1.,3.)),&[],&[],&render);
        assert_eq!(moved.scale,a.scale);
        Ok(())
    }

}
