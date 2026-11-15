//! Irradiance volume probes (render.retained indirect lighting; docs/design/lighting.md).
//! One installed volume is shared by every frame slot:
//! a sample re-traces it only when its lighting inputs change (`key`), and every
//! colour pass and hit kernel reads the result through group 2 (probes.slang).
//! Each update starts from zero and gathers `gi_diffuse_bounces` bounces, so the
//! volume depends only on the scene at that time, never on frame history.
use crate::*;
pub(crate) mod placement;
use placement::Field;

/// Stored total diffuse E/pi: local transport plus visible distant transport.
/// The capture writer projects channels in parallel; readers consume this typed
/// layout rather than treating it as local-only light or directional radiance.
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct TotalDiffuseIrradianceRecord {
    channels:[[f32;4];3],
    support:[f32;4],
}
const _: ()=assert!(std::mem::size_of::<TotalDiffuseIrradianceRecord>()==constants::PROBE_VEC4S as usize*16);

/// `ProbeGrid` in probes.slang.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct Grid {
    origin: [f32; 4],
    spacing: [f32; 4],
    counts: [u32; 4],
    bias: [f32; 4],
    pub(crate) visibility_policy: [u32; 4],
    field: [u32; 4],
}
impl Grid {
    /// One canonical brick. Resolution controls local record density, never
    /// the world extent of unrelated receivers.
    fn brick(volume:&vork_scene::ProbeVolume,key:[i32;3])->Self {
        let resolution=volume.resolution();let side=constants::PROBE_SPACING;
        let extent=Vec3::from_array(resolution.map(|n|n.saturating_sub(1).max(1) as f32))*side;
        let first=Vec3::from_array(key.map(|v|v as f32))*extent+Vec3::from_array(resolution.map(|n|if n==1 {0.5*side} else {0.}));
        Self {
            origin:[first.x,first.y,first.z,1.],spacing:[side,side,side,constants::PROBE_CAPTURE_DISTANCE],
            counts:[resolution[0],resolution[1],resolution[2],volume.rays_per_probe],
            bias:[constants::PROBE_NORMAL_BIAS*side,constants::PROBE_VIEW_BIAS*side,constants::PROBE_FACING_BIAS,constants::PROBE_VALIDITY_THRESHOLD],
            visibility_policy:[key[0] as u32,key[1] as u32,key[2] as u32,0],field:[0;4],
        }
    }
    pub(crate) fn count(&self) -> u32 {
        if self.field[0]!=0 {self.field[2]} else {self.counts[0] * self.counts[1] * self.counts[2]}
    }
    /// Rays traced per probe and dispatch: all of them unless the batch
    /// scratch would exceed available memory or a device binding limit.
    fn trace_stride(&self) -> u64 {if self.visibility_policy[3]!=0 {4} else {2}}
    fn batch(&self, capacity: u64) -> u32 {
        let lanes=(u64::from(self.count())*self.trace_stride()).max(u64::from(self.field[0])*2);
        let fit=capacity/(lanes*16);
        self.counts[3].min(fit.max(1).min(u64::from(u32::MAX)) as u32)
    }
}
/// Retained generation and its two readback ranges. The status word is in
/// Grid.field, not at the beginning of the dictionary header record.
pub(crate) struct Feedback {
    pub(crate) source: wgpu::Buffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) status_offset: u64,
}
/// Workgroups covering `threads` probe-update threads (probe_thread in probe_update.slang).
fn groups(threads: u64) -> [u32; 3] {
    let groups = threads.div_ceil(u64::from(constants::PROBE_GROUP));
    let width = u64::from(constants::PROBE_DISPATCH_WIDTH);
    [groups.min(width) as u32, groups.div_ceil(width) as u32, 1]
}
/// Record vec4s and depth-moment texels gathered per probe.
const GATHER_SLOTS:u64=(constants::PROBE_VEC4S+constants::PROBE_VISIBILITY_SIDE*constants::PROBE_VISIBILITY_SIDE) as u64;
// Three retained source-projection channels, independent of generation.
const SLOTS:u64=GATHER_SLOTS+3;

/// Group 2 entries of a probe reader from binding `first`: the final records,
/// depth moments and grid (probes.slang VORK_PROBE_BINDING). Hit kernels use 0.
// Storage permits sharing the shadow group with its ray-hit buffer array.
pub(crate) fn read_entries(first: u32, stages: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 3] {
    let storage = wgpu::BufferBindingType::Storage { read_only: true };
    [
        pipelines::buffer_entry(first, stages, storage),
        pipelines::buffer_entry(first + 1, stages, storage),
        pipelines::buffer_entry(first + 2, stages, wgpu::BufferBindingType::Storage {read_only:false}),
    ]
}
pub(crate) fn read_layout(d: &wgpu::Device) -> wgpu::BindGroupLayout {
    d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("probe volume read"),
        entries: &read_entries(0, wgpu::ShaderStages::COMPUTE),
    })
}
/// probe_update.slang group 0: the scene globals, TLAS, ray table, the traced
/// batch, gathered sums, outputs and the batch uniform (one dynamic offset per batch).
pub(crate) fn update_entries() -> Vec<wgpu::BindGroupLayoutEntry> {
    let compute = wgpu::ShaderStages::COMPUTE;
    let mut entries = pipelines::global_entries();
    entries.extend(rt::layout_entries(11, compute));
    let writable = wgpu::BufferBindingType::Storage { read_only: false };
    entries.extend((13..17).map(|binding| pipelines::buffer_entry(binding, compute, writable)));
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 17,
        visibility: compute,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
        count: None,
    });
    entries.push(pipelines::buffer_entry(18,compute,wgpu::BufferBindingType::Storage {read_only:true}));
    for e in &mut entries {
        e.visibility = compute;
    }
    entries
}

pub(crate) struct Volume {
    pub(crate) grid: Grid,
    #[cfg(test)]
    layout: Vec<Grid>,
    #[cfg(test)]
    placement_ms:f64,
    bounces: u32,
    /// Time-dependent geometry or materials (deformation, compute, programs,
    /// material tracks): every new sample time re-traces the volume.
    pub(crate) timed: bool,
    pub(crate) geometry_timed:bool,
    pub(crate) dependencies: vork_slang::ProgramDependencies,
    storage: crate::residency::Cache<Storage>,
    recipe: std::sync::Arc<StorageRecipe>,
    pub(crate) demand: std::sync::Arc<std::sync::Mutex<Demand>>,
    #[cfg(test)]
    batch_stride: u32,
    #[cfg(test)]
    batch_count: u32,
    #[cfg(test)]
    batch: u32,
    trace: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    choose:wgpu::ComputePipeline,
    directional_trace:wgpu::ComputePipeline,
    directional_gather:wgpu::ComputePipeline,
    filter:wgpu::ComputePipeline,
    update_layout: wgpu::BindGroupLayout,
    read_layout: wgpu::BindGroupLayout,
}
#[derive(Default)]
pub(crate) struct Demand {
    pub(crate) cells: std::collections::BTreeSet<[i32;3]>,
    pub(crate) capacity: u32,
}
pub(crate) struct Storage {
    recipe: std::sync::Arc<StorageRecipe>,
    pub(crate) grid_buffer: wgpu::Buffer,
    /// Two sets of batches at `batch_stride`: first ray, rays per probe and
    /// gathered slots. The first bounce gathers records and geometry moments;
    /// later bounces gather only the lighting records.
    batches: wgpu::Buffer,
    traced: wgpu::Buffer,
    sums: wgpu::Buffer,
    /// Records: [0] is final and read by consumers; bounces alternate with [1].
    irradiance: [wgpu::Buffer; 2],
    /// Depth moments: [0] is read by consumers, [1] is written by every bounce.
    visibility: [wgpu::Buffer; 2],
}
/// Native binding sizes, shared by placement admission and allocation. Counts
/// saturate so a proven lower bound can reject an unenumerable receiver region.
struct StorageSizes { records:u64, moments:u64, sums:u64, metadata:u64, traced:u64 }
impl StorageSizes {
    fn new(count:u64,bricks:u64,metadata:u64,trace_stride:u64,batch:u64)->Self {
        let records=count.saturating_mul(std::mem::size_of::<TotalDiffuseIrradianceRecord>() as u64);
        let moments=count.saturating_mul(constants::PROBE_VISIBILITY_SIDE.pow(2) as u64*8)
            .saturating_add(bricks.saturating_mul(u64::from(1+constants::PROBE_RADIANCE_SH_PAIRS+constants::PROBE_RADIANCE_TEXELS*constants::PROBE_RADIANCE_PAIRS)*8));
        let sums=count.saturating_mul(SLOTS).max(bricks.saturating_mul(u64::from(constants::PROBE_RADIANCE_SIDE.pow(2)*2+3))).saturating_mul(16);
        let traced=count.saturating_mul(trace_stride).max(bricks.saturating_mul(2)).saturating_mul(batch).saturating_mul(16);
        Self {records,moments,sums,metadata,traced}
    }
    fn fits(&self,limit:u64)->bool {[self.records,self.moments,self.sums,self.metadata,self.traced].into_iter().all(|bytes|bytes<=limit)}
    fn retained(&self)->u64 {self.records.saturating_add(self.moments).saturating_mul(2).saturating_add(self.sums).saturating_add(self.metadata)}
}
#[derive(Clone)]
struct StorageRecipe {
    field: Field,
    capacity: u32,
    linear: bool,
    batch: u32,
    batch_count: u32,
    batch_stride: u32,
    device: wgpu::Device,
    metadata: Vec<Grid>,
    words: Vec<u32>,
    records: u64,
    moments: u64,
    traced: u64,
    sums: u64,
}
impl StorageRecipe {
    fn new(d:&wgpu::Device,field:Field,capacity:u32,linear:bool) -> Result<Self> {
        let batch_stride=d.limits().min_uniform_buffer_offset_alignment.max(16);
        let mut field=field;
        let mut capacity=capacity;
        let limit=d.limits().max_storage_buffer_binding_size.min(d.limits().max_buffer_size);
        field.binding_limit=Some(limit);
        loop {
        let mut metadata=field.metadata();
        if field.demand {
            // Requests may stream across replays when a native binding cannot
            // address the whole dictionary. This changes scheduling, not support.
            capacity=capacity.min((limit/std::mem::size_of::<Grid>() as u64)
                .saturating_sub(metadata.len() as u64+1).min(u64::from(u32::MAX)) as u32).max(1);
            field.header.visibility_policy[2]=metadata.len() as u32;
            metadata[0]=field.header;
            metadata.push(Grid {field:[0,0,capacity,field.header.field[3]],..Grid::zeroed()});
            metadata.resize(metadata.len()+capacity as usize,Grid::zeroed());
        }
        let count=u64::from(field.header.count());
        let sizes=StorageSizes::new(count,u64::from(field.header.field[0]),(metadata.len()*std::mem::size_of::<Grid>()) as u64,field.header.trace_stride(),1);
        let StorageSizes {records,moments,sums,..}=sizes;
        let retained=sizes.retained();
        let batch=field.header.batch(crate::residency::available(&d).map_or(limit,|a|limit.min(a.saturating_sub(retained))));
        let batch_count=field.header.counts[3].div_ceil(batch);
        let sizes=StorageSizes::new(count,u64::from(field.header.field[0]),sizes.metadata,field.header.trace_stride(),u64::from(batch));
        let traced=sizes.traced;
        if !sizes.fits(limit) {
            // Demand growth and cache restoration obey the same native binding
            // capacity as initial placement, even without memory telemetry.
            field=field.coarsen()?;
            continue;
        }
        let words=Self::batch_words(field.header.counts[3],batch,batch_stride,linear);
        return Ok(Self {device:d.clone(),field,capacity,linear,batch,batch_count,batch_stride,metadata,words,records,moments,traced,sums});
        }
    }
    fn batch_words(rays:u32,batch:u32,batch_stride:u32,linear:bool) -> Vec<u32> {
        let batch_count=rays.div_ceil(batch);
        let mut words=vec![0;((4*batch_count+3)*batch_stride/4) as usize];
        for (set,slots) in [GATHER_SLOTS as u32,constants::PROBE_VEC4S as u32,constants::PROBE_VEC4S as u32,constants::PROBE_RADIANCE_SIDE.pow(2)].into_iter().enumerate() {
            for b in 0..batch_count {
                let at=((set as u32*batch_count+b)*batch_stride/4) as usize;
                words[at..at+4].copy_from_slice(&[b*batch,batch,slots,u32::from(set==1)*if linear {3}else{1}]);
            }
        }
        for level in 1..=3u32 {
            let at=((4*batch_count+level-1)*batch_stride/4) as usize;
            words[at..at+4].copy_from_slice(&[level,0,0,0]);
        }
        words
    }
    fn under_pressure(&self) -> Result<Self> {
        // Scratch is scheduling, not quality. Exhaust smaller trace batches
        // before changing the spatial field or its ray quadrature.
        if self.batch>1 {
            let mut next=self.clone();
            next.batch=(self.batch/2).max(1);
            next.batch_count=self.field.header.counts[3].div_ceil(next.batch);
            next.traced=self.traced/u64::from(self.batch)*u64::from(next.batch);
            next.words=Self::batch_words(self.field.header.counts[3],next.batch,self.batch_stride,self.linear);
            return Ok(next);
        }
        let mut field=self.field.coarsen()?;
        let mut capacity=(self.capacity/2).max(1);
        loop {
            if let Ok(recipe)=Self::new(&self.device,field.clone(),capacity,self.linear) {return Ok(recipe);}
            // Backend address limits can also constrain a pressure rebuild.
            // Request dictionaries can stream over replays; no miss is accepted.
            capacity=(capacity/2).max(1);
            field=field.coarsen()?;
        }
    }

    fn bytes(&self) -> u64 {
        (self.metadata.len()*std::mem::size_of::<Grid>()+self.words.len()*4) as u64
            + 2*(self.records+self.moments)+self.traced+self.sums
    }
    fn build(self:&std::sync::Arc<Self>) -> Storage {
        let d = &self.device;
        let buffer = |label, size| crate::gpu_memory::buffer(d, &wgpu::BufferDescriptor {
            label: Some(label), size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Storage {
            recipe: self.clone(),
            grid_buffer: init(d, "probe grid", bytemuck::cast_slice(&self.metadata), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC),
            batches: init(d, "probe batches", bytemuck::cast_slice(&self.words), wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST),
            traced: buffer("probe traced batch", self.traced),
            sums: buffer("probe gathered sums", self.sums),
            irradiance: [buffer("probe records", self.records), buffer("probe records (bounce)", self.records)],
            visibility: [buffer("probe depth moments", self.moments), buffer("probe depth moments (traced)", self.moments)],
        }
    }
    fn try_build(self:&std::sync::Arc<Self>) -> Option<Storage> {
        let d = &self.device;
        let buffer = |label, size| crate::gpu_memory::optional_buffer(d, &wgpu::BufferDescriptor {
            label: Some(label), size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Storage {
            recipe: self.clone(),
            grid_buffer: crate::gpu_memory::optional_buffer_init(d, "probe grid", bytemuck::cast_slice(&self.metadata), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC)?,
            batches: crate::gpu_memory::optional_buffer_init(d, "probe batches", bytemuck::cast_slice(&self.words), wgpu::BufferUsages::UNIFORM|wgpu::BufferUsages::COPY_DST)?,
            traced: buffer("probe traced batch", self.traced)?,
            sums: buffer("probe gathered sums", self.sums)?,
            irradiance: [buffer("probe records", self.records)?, buffer("probe records (bounce)", self.records)?],
            visibility: [buffer("probe depth moments", self.moments)?, buffer("probe depth moments (traced)", self.moments)?],
        })
    }
}
impl Storage {
    fn retire(&self) {
        for buffer in [&self.grid_buffer,&self.batches,&self.traced,&self.sums,&self.irradiance[0],&self.irradiance[1],&self.visibility[0],&self.visibility[1]] {buffer.destroy();}
    }
    pub(crate) fn read_bindings(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 3] {self.bindings(first, 0)}
    fn bindings(&self, first: u32, records: usize) -> [wgpu::BindGroupEntry<'_>; 3] {
        [entry(first, self.irradiance[records].as_entire_binding()),
         entry(first+1, self.visibility[0].as_entire_binding()),
         entry(first+2, self.grid_buffer.as_entire_binding())]
    }
}
/// The probe update's trace and gather pipelines and their layouts; `vork setup`
/// creates them through this constructor too.
pub(crate) fn pipelines(r: &Renderer, pipelines: &CachedPipelines, features:crate::forward_variant::FeatureMask, variants:&pipelines::VariantCompile) -> Result<(wgpu::ComputePipeline,wgpu::ComputePipeline,wgpu::ComputePipeline,wgpu::ComputePipeline,wgpu::ComputePipeline,wgpu::ComputePipeline,wgpu::BindGroupLayout,wgpu::BindGroupLayout)> {
    let d = &r.device;
    let update_layout = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("probe update"),
        entries: &update_entries(),
    });
    let read_layout = read_layout(d);
    let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("probe update layout"),
        bind_group_layouts: &[Some(&update_layout), Some(&pipelines.mat_layout), Some(&read_layout), Some(&rt::hits_layout(d))],
        immediate_size: 0,
    });
    let pipeline = |label, module: &wgpu::ShaderModule, entry| r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: r.cache.as_ref(),
    });
    let kernel=pipelines.drawn_kernel(pipelines::ProgramKernel::Probes {features,directional:false},variants,&r.compiler);
    let (module, entry) = pipelines.program_kernel(kernel)?;
    let trace = pipeline("probe volume trace", &module, entry);
    r.compiler.remember(pipelines.kernel_content_key(kernel));
    let gather = pipeline("probe volume gather", &native::shader_module(d, &native::PROBE_UPDATE_GATHER), native::PROBE_UPDATE_GATHER.entry);
    let kernel=pipelines.drawn_kernel(pipelines::ProgramKernel::Probes {features,directional:true},variants,&r.compiler);
    let (module,entry)=pipelines.program_kernel(kernel)?;
    let directional_trace=pipeline("local directional capture",&module,entry);
    r.compiler.remember(pipelines.kernel_content_key(kernel));
    let choose=pipeline("choose directional capture positions",&native::shader_module(d,&native::PROBE_UPDATE_CHOOSE),native::PROBE_UPDATE_CHOOSE.entry);
    let directional_gather=pipeline("gather directional captures",&native::shader_module(d,&native::PROBE_UPDATE_DIRECTIONAL_GATHER),native::PROBE_UPDATE_DIRECTIONAL_GATHER.entry);
    let filter=pipeline("filter directional captures",&native::shader_module(d,&native::PROBE_UPDATE_FILTER),native::PROBE_UPDATE_FILTER.entry);
    Ok((trace,gather,choose,directional_trace,directional_gather,filter,update_layout,read_layout))
}

impl Volume {
    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let storage=self.storage.get();
        let recipe=storage.as_ref().map_or(&self.recipe,|s|&s.recipe);
        let grid=recipe.field.header;
        serde_json::json!({
            "bricks": grid.field[0], "vertices": grid.count(),
            "max_level": grid.field[3], "rays_per_probe": grid.counts[3],
            "bounces": self.bounces, "time_dependent": self.timed,
            "storage_bytes": recipe.bytes(), "capture_batch": recipe.batch,
        })
    }

    pub(crate) fn new(r: &Renderer, pipelines: &CachedPipelines, mut field: Field, bounces: u32, timed: bool, exposure_samples:u32, animated_shutter:bool, dependencies: vork_slang::ProgramDependencies, geometry_timed:bool, linear:bool, features:crate::forward_variant::FeatureMask, variants:&pipelines::VariantCompile) -> Result<Self> {
        let d = &r.device;
        // A timed shutter must integrate the exposure's capture budget, rather
        // than multiply the entire bake by the number of camera time strata.
        let dynamic = timed && animated_shutter;
        let strata=if timed && !dynamic {exposure_samples.max(1)} else {1};
        field.header.counts[3]=field.header.counts[3].div_ceil(strata);
        // Zero selects the current frame's stratum count. Reserve the complete
        // closed-exposure ray allocation, and use its prefix for open exposures.
        field.header.visibility_policy[0]=if dynamic {0} else {strata};
        for brick in &mut field.bricks {brick.counts[3]=field.header.counts[3];}
        let demand = std::sync::Arc::new(std::sync::Mutex::new(Demand {capacity:field.header.count().max(1),..Default::default()}));
        let grid=field.header;
        let capacity=field.header.count().max(1);
        let (trace,gather,choose,directional_trace,directional_gather,filter,update_layout,read_layout) = self::pipelines(r, pipelines, features, variants)?;
        #[cfg(test)]
        let batch_stride = d.limits().min_uniform_buffer_offset_alignment.max(16);
        let recipe=std::sync::Arc::new(StorageRecipe::new(d,field.clone(),capacity,linear)?);
        let mut result=Self {
            grid,
            #[cfg(test)]
            layout: Vec::new(),
            #[cfg(test)]
            placement_ms:field.placement_ms,
            demand,
            bounces,
            timed,
            geometry_timed,
            dependencies,
            storage: crate::residency::Cache::new(&r.residency),
            recipe,
            #[cfg(test)]
            batch_stride,
            #[cfg(test)]
            batch_count: 0,
            #[cfg(test)]
            batch: 1,
            trace,
            gather,
            choose,directional_trace,directional_gather,filter,
            update_layout,
            read_layout,
        };
        result.apply_recipe();
        result.allocate(r)?;
        Ok(result)
    }
    pub(crate) fn feedback(&self) -> Option<Feedback> {
        let storage=self.storage();
        let offset=u64::from(storage.recipe.field.header.visibility_policy[2])*std::mem::size_of::<Grid>() as u64;
        (offset!=0).then(|| Feedback {source:storage.grid_buffer.clone(),offset,
            size:storage.recipe.metadata.len() as u64*std::mem::size_of::<Grid>() as u64-offset,
            status_offset:offset+std::mem::offset_of!(Grid,field) as u64})
    }
    pub(crate) fn collect_demand(demand: &std::sync::Mutex<Demand>, bytes: &[u8]) -> bool {
        let grids: &[Grid]=bytemuck::cast_slice(bytes);
        if grids[0].field[0]==0 {return false;}
        let mut demand=demand.lock().unwrap();
        for grid in &grids[1..] {
            if grid.field[0]==2 {demand.cells.insert(std::array::from_fn(|i|grid.counts[i] as i32));}
        }
        if grids[0].field[1]!=0 {demand.capacity=demand.capacity.max(grids[0].field[2].saturating_mul(2));}
        true
    }
    fn configure(&mut self, r:&Renderer, field:Field, capacity:u32) -> Result<()> {
        self.recipe=std::sync::Arc::new(StorageRecipe::new(&r.device,field,capacity,self.recipe.linear)?);
        self.apply_recipe();
        Ok(())
    }
    fn apply_recipe(&mut self) {
        self.grid=self.recipe.field.header;
        #[cfg(test)] {
            self.batch=self.recipe.batch;
            self.batch_count=self.recipe.batch_count;
            self.batch_stride=self.recipe.batch_stride;
            self.layout=self.recipe.metadata.clone();
        }
    }
    fn allocate(&mut self,r:&Renderer) -> Result<()> {
        self.allocate_with(r,StorageRecipe::try_build)
    }
    fn allocate_with(&mut self,r:&Renderer,mut build:impl FnMut(&std::sync::Arc<StorageRecipe>)->Option<Storage>) -> Result<()> {
        loop {
            if let Some(storage)=build(&self.recipe) {
                self.demand.lock().unwrap().capacity=self.recipe.capacity;
                self.storage.insert(storage,self.recipe.bytes());
                return Ok(());
            }
            r.errors.check()?;
            // optional_buffer has already evicted idle caches and retried the
            // allocator. Only verified native pressure reaches this branch.
            self.recipe=std::sync::Arc::new(self.recipe.under_pressure()?);
            self.apply_recipe();
        }
    }
    pub(crate) fn rebuild(&mut self, r:&Renderer) -> Result<()> {
        let previous=self.storage();
        let (field,capacity)={
            let demand=self.demand.lock().unwrap();
            (previous.recipe.field.grow_bricks(&demand.cells,previous.recipe.field.header.field[3])?,demand.capacity)
        };
        // All submitted users have been drained by redo_frame. Retire their
        // buffers before admission so the old generation is not double charged.
        if let Some(old)=self.storage.take() {old.retire();}
        let _=r.device.poll(wgpu::PollType::Poll);
        self.configure(r,field,capacity)?;
        self.allocate(r)?;
        self.demand.lock().unwrap().cells.clear();
        Ok(())
    }
    pub(crate) fn storage(&self) -> std::sync::Arc<Storage> {
        self.storage_with(StorageRecipe::try_build)
    }
    fn storage_with(&self,mut build:impl FnMut(&std::sync::Arc<StorageRecipe>)->Option<Storage>) -> std::sync::Arc<Storage> {
        self.storage.get_or_try_init(|| {
            let mut recipe=self.recipe.clone();
            loop {
                if let Some(storage)=build(&recipe) {self.demand.lock().unwrap().capacity=recipe.capacity;return Ok((storage,recipe.bytes()));}
                if crate::gpu_memory::is_lost(&recipe.device) {break;}
                match recipe.under_pressure() {Ok(next)=>recipe=std::sync::Arc::new(next),Err(_)=>break}
            }
            // Preserve the renderer's normal device-error boundary when even
            // the smallest representable support cannot be allocated.
            Ok((recipe.build(),recipe.bytes()))
        }).expect("infallible probe cache rebuild")
    }
    pub(crate) fn read_bind(&self, d: &wgpu::Device) -> wgpu::BindGroup {
        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe volume read"),
            layout: &self.read_layout,
            entries: &self.storage().read_bindings(0),
        })
    }
    /// A frame slot's update bindings: per written record buffer, its group 0
    /// (this slot's globals, TLAS and outputs) and group 2 (the other buffer).
    pub(crate) fn slot(&self, s: &InstalledScene, d: &wgpu::Device, view: &crate::view::ViewState, objects: &wgpu::Buffer,
        rays: &rt::FrameRays, material: &wgpu::BindGroup) -> Result<Slot> {
        let storage = self.storage();
        let table = &s.rays.as_ref().context("probes need the scene ray table")?.alpha;
        let lighting = &view.lighting;
        let binds = [0, 1].map(|write| {
            let mut entries = pipelines::ray_globals(s, &lighting.uniform, objects, &lighting.lights, &lighting.cluster_config, &lighting.cluster_lights, &lighting.analytic_light_tree);
            entries.extend([
                entry(11, rays.tlas.as_binding()),
                entry(12, table.as_entire_binding()),
                entry(13, storage.traced.as_entire_binding()),
                entry(14, storage.sums.as_entire_binding()),
                entry(15, storage.irradiance[write].as_entire_binding()),
                entry(16, storage.visibility[1].as_entire_binding()),
                entry(18,rays.emitter_weights.as_ref().map_or(table,|weights|&weights.buffer).as_entire_binding()),
                entry(17, wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &storage.batches, offset: 0, size: std::num::NonZeroU64::new(16),
                })),
            ]);
            let update = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe update"),
                layout: &self.update_layout,
                entries: &entries,
            });
            let read = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("probe update previous bounce"),
                layout: &self.read_layout,
                entries: &storage.bindings(0, 1 - write),
            });
            (update, read)
        });
        Ok(Slot { _storage: storage, binds, material: material.clone(), hits: rays.hits_bind(d) })
    }
    /// Seed captures must never consume a previous frame's retained lighting.
    pub(crate) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        let storage = self.storage();
        for buffer in &storage.irradiance {encoder.clear_buffer(buffer,0,None);}
        // Seed image lighting starts dark in both representations. Preserve the
        // separately retained geometry moments when geometry did not change.
        let offset=u64::from(storage.recipe.field.header.count())*constants::PROBE_VISIBILITY_SIDE.pow(2) as u64*8;
        for buffer in &storage.visibility {encoder.clear_buffer(buffer,offset,None);}
    }
    /// Publish the final bounce in buffer zero, starting from a dark volume.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, slot: &Slot, queries: &wgpu::QuerySet, geometry_changed:bool, exposure_samples:u32) {
        let storage = &slot._storage;
        let rays = if self.grid.visibility_policy[0] == 0 {
            self.grid.counts[3].div_ceil(exposure_samples.max(1))
        } else { self.grid.counts[3] };
        let batches = rays.div_ceil(storage.recipe.batch);
        let generations = self.bounces;
        let write = |bounce: u32| ((generations - 1 - bounce) % 2) as usize;
        encoder.clear_buffer(&storage.irradiance[1 - write(0)], 0, None);
        let span = timing::query("probes");
        for bounce in 0..generations {
            let (update, read) = &slot.binds[write(bounce)];
            let probes = u64::from(storage.recipe.field.header.count());
            // Every bounce traces the same geometry and ray directions. Only
            // outgoing radiance changes; depth moments from bounce zero remain valid.
            let slots = if bounce == 0 && geometry_changed { GATHER_SLOTS } else { u64::from(constants::PROBE_VEC4S as u32) };
            let batch_set = if bounce == 0 {if geometry_changed {0} else {2*storage.recipe.batch_count}} else {storage.recipe.batch_count};
            for batch in 0..batches {
                // Resident support and capture scratch are separate budgets. A
                // large sparse field can need many batches; keep each in its
                // own native compute body rather than one unbounded bounce.
                let first=bounce==0 && batch==0;
                let last=false; // Directional filtering closes the complete capture span.
                let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label:Some("probe capture batch"),
                    timestamp_writes:(first||last).then_some(wgpu::ComputePassTimestampWrites {
                        query_set:queries,beginning_of_pass_write_index:first.then_some(span),end_of_pass_write_index:last.then_some(span+1),
                    }),
                });
                pass.set_bind_group(1,&slot.material,&[]);
                pass.set_bind_group(2,read,&[]);
                pass.set_bind_group(3,&slot.hits,&[]);
                pass.set_bind_group(0, update, &[(batch_set + batch) * storage.recipe.batch_stride]);
                for (pipeline, threads) in [(&self.trace, probes * u64::from(storage.recipe.batch)), (&self.gather, probes * slots)] {
                    pass.set_pipeline(pipeline);
                    let [x, y, z] = groups(threads);
                    pass.dispatch_workgroups(x, y, z);
                }
            }
            // Geometry is fixed within an update: later bounces read the first's moments.
            if bounce == 0 && geometry_changed {
                encoder.copy_buffer_to_buffer(&storage.visibility[1],0,&storage.visibility[0],0,Some(u64::from(storage.recipe.field.header.count())*constants::PROBE_VISIBILITY_SIDE.pow(2) as u64*8));
            }
        }
        // Directional providers have their own density and radiance storage:
        // one free-space capture per brick, selected from validated vertices.
        // They consume the complete final diffuse field, never prior-frame gloss.
        let (update,read)=&slot.binds[1];
        let bind=|pass:&mut wgpu::ComputePass,offset| {
            pass.set_bind_group(0,update,&[offset]);pass.set_bind_group(1,&slot.material,&[]);
            pass.set_bind_group(2,read,&[]);pass.set_bind_group(3,&slot.hits,&[]);
        };
        {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("choose directional providers"),..Default::default()});
            pass.set_pipeline(&self.choose);bind(&mut pass,0);
            let [x,y,z]=groups(u64::from(storage.recipe.field.header.field[0]));pass.dispatch_workgroups(x,y,z);
        }
        for batch in 0..batches {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("directional capture batch"),..Default::default()});
            bind(&mut pass,(3*storage.recipe.batch_count+batch)*storage.recipe.batch_stride);
            for (pipeline,threads) in [(&self.directional_trace,u64::from(storage.recipe.field.header.field[0])*u64::from(storage.recipe.batch)),(&self.directional_gather,u64::from(storage.recipe.field.header.field[0])*u64::from(constants::PROBE_RADIANCE_SIDE.pow(2)))] {
                pass.set_pipeline(pipeline);let [x,y,z]=groups(threads);pass.dispatch_workgroups(x,y,z);
            }
        }
        for level in 1..=3u32 {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {label:Some("filter directional providers"),timestamp_writes:(level==3).then_some(wgpu::ComputePassTimestampWrites {query_set:queries,beginning_of_pass_write_index:None,end_of_pass_write_index:Some(span+1)})});
            pass.set_pipeline(&self.filter);bind(&mut pass,(4*storage.recipe.batch_count+level-1)*storage.recipe.batch_stride);
            let [x,y,z]=groups(u64::from(storage.recipe.field.header.field[0])*u64::from((constants::PROBE_RADIANCE_SIDE>>level).pow(2)));pass.dispatch_workgroups(x,y,z);
        }
        let offset=u64::from(storage.recipe.field.header.count())*constants::PROBE_VISIBILITY_SIDE.pow(2) as u64*8;
        encoder.copy_buffer_to_buffer(&storage.visibility[1],offset,&storage.visibility[0],offset,Some(storage.visibility[0].size()-offset));
    }
}
pub(crate) struct Slot {
    _storage: std::sync::Arc<Storage>,
    binds: [(wgpu::BindGroup, wgpu::BindGroup); 2],
    material: wgpu::BindGroup,
    hits: wgpu::BindGroup,
}

/// Native inputs probe rays see (program resources are appended separately): object poses and materials,
/// light emission (not camera-fitted shadow matrices), the world and, when
/// deformation, compute or material tracks animate, the sample time.
// Geometry/coverage inputs use the same submission-ordered tracker as lighting.
// Lights and World cannot change capture distances; camera/image/time are added
// only through the installed material dependency contract by the caller.
pub(crate) fn geometry_key(objects:&[Object],time:Option<f32>)->Vec<u8> {
    key(objects,&[],&Uniform::zeroed(),time)
}

pub(crate) fn key(objects: &[Object], lights: &[Light], uniform: &Uniform, time: Option<f32>) -> Vec<u8> {
    let mut key = Vec::new();
    for o in objects {
        key.extend_from_slice(bytemuck::bytes_of(&o.model));
        key.extend_from_slice(bytemuck::bytes_of(&o.opacity));
        key.extend_from_slice(bytemuck::bytes_of(&o.material_index));
        key.extend_from_slice(bytemuck::bytes_of(&o.metadata));
    }
    for l in lights {
        for part in [l.position, l.direction, l.radiance, l.emission, l.projection, [0., l.params[1], 0., 0.]] {
            key.extend_from_slice(bytemuck::bytes_of(&part));
        }
        // Area emitter edges (kind 3); a sun's carry the camera axis instead.
        if l.position[3] == 3. {
            key.extend_from_slice(bytemuck::bytes_of(&[l.emitter_u, l.emitter_v]));
        }
        key.extend_from_slice(bytemuck::cast_slice(&l.shadow_offsets));
    }
    key.extend_from_slice(bytemuck::bytes_of(&[uniform.radiance[0], uniform.radiance[1], uniform.radiance[2]]));
    key.extend_from_slice(bytemuck::bytes_of(&uniform.environment));
    key.extend_from_slice(bytemuck::bytes_of(&uniform.environment_rotation));
    key.extend_from_slice(bytemuck::bytes_of(&uniform.environment_color));
    key.extend_from_slice(bytemuck::bytes_of(&uniform.environment_links));
    if let Some(time) = time {
        key.extend_from_slice(&time.to_le_bytes());
    }
    key
}

/// Append only program-visible inputs to the existing volume key.
pub(crate) fn program_key(key: &mut Vec<u8>, uniform: &Uniform, dependencies: vork_slang::ProgramDependencies, image_revision: u64) {
    if dependencies.camera {
        key.extend_from_slice(bytemuck::bytes_of(&uniform.view));
        key.extend_from_slice(bytemuck::bytes_of(&uniform.vp));
    }
    if dependencies.images { key.extend_from_slice(&image_revision.to_le_bytes()); }
}

#[cfg(all(test, vork_private_tests))]
include!("probes_tests_private.rs");
