//! Structural validation shared by file loading and programmatic authoring.
use super::*;
use anyhow::{Context, Result, bail, ensure};

impl BufferType {
    pub fn width(&self) -> usize {
        match self {
            Self::Vec2 => 2,
            Self::Vec3 => 3,
            Self::Vec4 | Self::Uvec4 => 4,
            Self::Mat4 => 16,
            _ => 1,
        }
    }
}
impl Scene {
    pub(super) fn validate_resources(&self, nodes_by_name: &BTreeMap<&str, &Node>, require_camera:bool) -> Result<()> {
        let mut errors = Vec::new();
        errors.extend(self.geometry_stages.iter().enumerate().map(|(index,stage)| -> Result<()> {
            ensure!(!self.geometry_stages[..index].contains(stage),"geometry_stages[{index}] repeats {}; remove the duplicate stage",stage.name());
            Ok(())
        }).filter_map(Result::err));
        let lookup = |name: &str| -> Result<&Buffer> {
            let buffer=self.buffers.get(name).with_context(|| {
                diagnostics::unknown("buffer", name, self.buffers.keys().map(String::as_str))
            })?;
            ensure!(buffer.grid.is_none(), "grid buffer {name} cannot be used as numeric data");
            Ok(buffer)
        };
        errors.extend(self.buffers.iter().map(|(name,b)| -> Result<()> {
            if let Some(dims) = b.dims {
                ensure!(dims.iter().all(|&n| n > 0), "buffer {name}: dims must be positive");
                let count = dims.into_iter().try_fold(1u32, |n, d| n.checked_mul(d)).context("dense grid dimensions overflow")?;
                ensure!(count.checked_mul(b.element_type.width() as u32).is_some(), "buffer {name}: dense grid word count overflows");
                ensure!(b.grid.is_none(), "buffer {name}: dims cannot be combined with grid");
                ensure!(b.data.is_none() || b.file.is_none(), "buffer {name}: data and file are mutually exclusive");
                ensure!(b.file.is_some() || (b.offset == 0 && b.count.is_none()), "buffer {name}: offset/count require a file");
                ensure!(b.count.is_none_or(|n| n == count), "buffer {name}: count differs from dims product");
                ensure!(b.data.as_ref().is_none_or(|v| v.len() == count as usize), "buffer {name}: data length differs from dims product");
            }
            if let Some(grid) = &b.grid {
                ensure!(!grid.is_empty() && b.data.is_none() && b.offset == 0 && b.count.is_none(), "buffer {name}: grids require a name/file and no data/offset/count");
                ensure!(matches!(b.element_type, BufferType::F32 | BufferType::Vec3), "buffer {name}: grid voxel type must be f32 or vec3");
                ensure!(b.file.as_deref().is_some_and(|p| p.ends_with(".vdb") || p.ends_with(".nvdb")), "buffer {name}: grids require .vdb/.nvdb files");
            }
            ensure!(
                b.offset % 4 == 0,
                "buffer {name}: offset must be aligned to four bytes"
            );
            if let Some(data) = &b.data {
                ensure!(!data.is_empty(), "buffer {name}: empty data");
                ensure!(
                    b.offset == 0 && b.count.is_none(),
                    "buffer {name}: offset/count require a file"
                );
                fn flat<'a>(value: &'a serde_json::Value, out: &mut Vec<&'a serde_json::Value>) {
                    if let Some(a) = value.as_array() {
                        for v in a {
                            flat(v, out);
                        }
                    } else {
                        out.push(value);
                    }
                }
                for value in data {
                    let mut numbers = Vec::new();
                    flat(value, &mut numbers);
                    ensure!(
                        numbers.len() == b.element_type.width(),
                        "buffer {name}: element width mismatch"
                    );
                    for n in numbers {
                        let valid = match b.element_type {
                            BufferType::U32 | BufferType::Uvec4 => {
                                n.as_u64().is_some_and(|n| u32::try_from(n).is_ok())
                            }
                            BufferType::I32 => n.as_i64().is_some_and(|n| i32::try_from(n).is_ok()),
                            _ => n.as_f64().is_some_and(|n| (n as f32).is_finite()),
                        };
                        ensure!(valid, "buffer {name}: invalid scalar");
                    }
                }
            }
            Ok(())
        }).filter_map(Result::err));
        errors.extend(self.programs.iter().map(|(name,p)| -> Result<()> {
            ensure!(p.source_origin.is_none() || p.source.is_some(),"program {name}: source_origin requires inline source");
            ensure!(p.source_origin.as_ref().is_none_or(|origin|!origin.is_empty() && Path::new(origin).file_name().is_some()),"program {name}: source_origin must name a virtual source file");
            if p.is_compute() {
                ensure!(
                    p.schedule.is_some() && p.dispatch.is_some(),
                    "compute {name}: schedule and dispatch required"
                );
                ensure!(!p.bindings.is_empty(), "compute {name}: bindings required");
                ensure!(
                    p.dispatch.unwrap().threads().iter().all(|n| *n > 0),
                    "compute {name}: dispatch dimensions must be positive"
                );
                for (binding, buffer) in &p.bindings {
                    ensure!(
                        !binding.is_empty()
                            && binding
                                .bytes()
                                .enumerate()
                                .all(|(i, b)| b.is_ascii_alphabetic()
                                    || b == b'_'
                                    || (i > 0 && b.is_ascii_digit())),
                        "compute {name}: invalid binding identifier {binding}"
                    );
                    ensure!(lookup(buffer)?.grid.is_none(), "compute {name}: grid buffers are immutable volume fields");
                }
            } else {
                ensure!(
                    p.bindings.is_empty() && p.schedule.is_none() && p.dispatch.is_none(),
                    "program {name}: compute bindings/dispatch require a schedule"
                );
            }
            Ok(())
        }).filter_map(Result::err));
        let result = (|| -> Result<()> {
        if let Some(sim) = &self.simulation {
            ensure!(sim.t0.as_f32().is_finite() && sim.dt.as_f64().is_finite() && sim.dt.as_f64() > 0. && sim.dt.as_f32().is_finite() && sim.dt.as_f32() > 0. && sim.t0.as_f32().is_finite(), "simulation: finite t0 and positive finite dt required");
            ensure!(!sim.state.is_empty() && !sim.init.is_empty() && !sim.steps.is_empty(), "simulation: state, init and steps must be nonempty");
            let state: std::collections::BTreeSet<_> = sim.state.iter().collect();
            ensure!(state.len() == sim.state.len(), "simulation: duplicate state buffer");
            for buffer in &sim.state { lookup(buffer)?; }
            let passes: std::collections::BTreeSet<_> = sim.init.iter().chain(&sim.steps).collect();
            for pass in &passes {
                let p = self.programs.get(*pass).with_context(|| format!("simulation: {}", diagnostics::unknown("program", pass, self.programs.keys().map(String::as_str))))?;
                ensure!(p.schedule == Some(ComputeSchedule::Simulation), "simulation pass {pass}: schedule must be simulation");
                ensure!(p.bindings.values().all(|b| state.contains(b)), "simulation pass {pass}: every binding must be declared in state");
            }
            for (name, p) in &self.programs {
                if p.schedule == Some(ComputeSchedule::Simulation) {
                    ensure!(passes.contains(name), "simulation program {name} is absent from init/steps");
                } else if p.is_compute() {
                    ensure!(p.bindings.values().all(|b| !state.contains(b)), "compute {name}: simulation state has a single writer schedule");
                }
            }
        } else {
            ensure!(self.programs.values().all(|p| p.schedule != Some(ComputeSchedule::Simulation)), "simulation programs require scene.simulation");
        }
        Ok(())
        })();
        if let Err(error) = result {errors.push(error);}
        let computed: std::collections::BTreeSet<&str> = self
            .programs
            .values()
            .filter(|p| p.is_compute())
            .flat_map(|p| p.bindings.values().map(String::as_str))
            .collect();
        let immutable = |buffer: &str, consumer: &str| -> Result<()> {
            ensure!(lookup(buffer)?.dims.is_none(), "dense grid {buffer} cannot be used by install-time {consumer}");
            ensure!(
                !computed.contains(buffer),
                "compute-bound buffer {buffer} cannot be used by install-time {consumer}"
            );
            Ok(())
        };
        errors.extend(self.geometry.values().map(|geometry| -> Result<()> {
            let dynamic = !matches!(geometry.topology, Topology::Volume | Topology::Implicit) && (geometry.attributes.values().chain(geometry.widths.iter()).chain(geometry.ids.iter()).any(|b| computed.contains(b.as_str())) || geometry.count.is_some() || geometry.status.is_some());
            if dynamic {
                ensure!(matches!(geometry.topology, Topology::Triangles | Topology::Points), "computed geometry requires triangles or points");
                ensure!(geometry.subdivision.is_none() && geometry.face_vertex_counts.is_none(), "computed geometry requires fixed triangle topology without subdivision; place subdivision after the compute stage");
            }
            if let Some(ids) = &geometry.ids {
                ensure!(matches!(geometry.topology,Topology::Points),"geometry.ids requires points");
                let ids=lookup(ids)?;
                ensure!(ids.element_type==BufferType::U32 && ids.grid.is_none() && ids.dims.is_none(),"geometry.ids requires a numeric u32 buffer");
                if let Some(position)=geometry.attributes.get("POSITION") {
                    if let (Some(a),Some(b))=(ids.len(),lookup(position)?.len()) {ensure!(a==b,"point ids count must match POSITION");}
                }
            }
            if let Some(status) = &geometry.status {
                ensure!(matches!(geometry.topology, Topology::Triangles | Topology::Points | Topology::Implicit), "geometry.status requires triangles, points or implicit surfaces");
                let buffer = lookup(&status.buffer)?;
                ensure!(buffer.element_type == BufferType::U32 && buffer.grid.is_none() && buffer.dims.is_none() && buffer.len().is_none_or(|n| n == 1), "geometry.status requires a one-element numeric u32 buffer");
                ensure!(!status.message.trim().is_empty(), "geometry.status.message must describe the failure and remedy");
            }
            if let Some(count) = &geometry.count {
                ensure!(matches!(geometry.topology, Topology::Points), "geometry.count requires points");
                let buffer = lookup(count)?;
                ensure!(buffer.element_type == BufferType::U32 && buffer.len().is_none_or(|n| n == 1), "geometry.count requires a one-element u32 buffer");
            }
            for buffer in geometry.curve_vertex_counts.iter().chain(geometry.widths.iter()) {
                if !matches!(geometry.topology, Topology::Points) { immutable(buffer, "curve tessellation")?; }
                ensure!(lookup(buffer)?.grid.is_none() && lookup(buffer)?.dims.is_none(), "curve data cannot use grid buffers");
            }
            if let Some(buffer) = &geometry.indices {
                immutable(buffer, "indices")?;
            }
            for morph in geometry.morphs.values() {
                immutable(&morph.POSITION, "morph deltas")?;
                ensure!(lookup(&morph.POSITION)?.grid.is_none(), "morph deltas cannot use grid buffers");
                if let Some(buffer) = &morph.NORMAL {
                    immutable(buffer, "morph deltas")?;
                    ensure!(lookup(buffer)?.grid.is_none(), "morph deltas cannot use grid buffers");
                }
            }
            Ok(())
        }).filter_map(Result::err));
        let mut pending: Vec<&Node> = self.nodes.iter().collect();
        while let Some(node) = pending.pop() {
            let result = (|| -> Result<()> {
            pending.extend(node.children.iter());
            if let NodeKind::Mesh {
                instances, skin, ..
            } = &node.kind
            {
                if let Some(skin) = skin {
                    immutable(&skin.inverse_bind_matrices, "inverse bind matrices")?;
                }
                if let Some(instances) = instances {
                    if let Some(buffer) = instances.attributes.get("pass_index") {
                        immutable(buffer, "instance IDs")?;
                    }
                    if let Some(buffer) = instances.attributes.get("time_offset") {
                        immutable(buffer, "instance animation offsets")?;
                    }
                }
            }
            Ok(())
            })();
            if let Err(error) = result {errors.push(error);}
        }
        errors.extend(self.animation.iter().map(|layer| -> Result<()> {
            for track in layer.tracks.iter().flatten() {
                if let Times::Buffer(buffer) = &track.times {
                    immutable(buffer, "track times")?;
                    ensure!(lookup(buffer)?.grid.is_none(), "track times cannot use grid buffers");
                }
                if let Values::Buffer(buffer) = &track.values {
                    immutable(buffer, "track values")?;
                    ensure!(lookup(buffer)?.grid.is_none(), "track values cannot use grid buffers");
                }
            }
            Ok(())
        }).filter_map(Result::err));
        errors.extend(self.geometry.iter().map(|(name,g)| -> Result<()> {
            if matches!(g.topology, Topology::Implicit) {
                ensure!(g.attributes.len()==1 && g.indices.is_none() && g.material_ranges.is_empty() && g.morphs.is_empty() && g.subdivision.is_none() && g.face_vertex_counts.is_none() && g.crease_edge.is_empty() && g.widths.is_none() && g.curve_vertex_counts.is_none(), "geometry {name}: implicit requires exactly one grid and no mesh modifiers");
                let b=lookup(g.attributes.values().next().unwrap())?;
                ensure!(b.element_type==BufferType::F32 && b.dims.is_some_and(|d|d.iter().all(|&v|v>=2)), "geometry {name}: implicit requires f32 dims >= 2");
                return Ok(());
            }
            if matches!(g.topology, Topology::Volume) {
                ensure!(!g.attributes.is_empty(), "geometry {name}: volume requires named grid fields");
                ensure!(g.indices.is_none() && g.material_ranges.is_empty() && g.morphs.is_empty()
                    && g.face_vertex_counts.is_none() && g.crease_edge.is_empty()
                    && g.subdivision.is_none()
                    && g.curve_vertex_counts.is_none() && g.widths.is_none(),
                    "geometry {name}: volume fields cannot have mesh/curve data or modifiers");
                for buffer in g.attributes.values() {
                    ensure!(self.buffers.get(buffer).is_some_and(|b|b.is_volume_grid()), "geometry {name}: volume fields require grid buffers");
                }
                return Ok(());
            }
            for buffer in g.attributes.values().chain(g.indices.iter()) {
                ensure!(lookup(buffer)?.grid.is_none() && lookup(buffer)?.dims.is_none(), "geometry {name}: grid buffers require volume topology");
            }
            if matches!(g.topology, Topology::Curves) {
                let counts = g.curve_vertex_counts.as_ref().context("curves require curve_vertex_counts")?;
                let widths = g.widths.as_ref().context("curves require widths (diameter)")?;
                ensure!(lookup(counts)?.element_type == BufferType::U32, "curve_vertex_counts require u32");
                ensure!(lookup(widths)?.element_type == BufferType::F32, "widths require f32");
                ensure!(g.indices.is_none() && g.material_ranges.is_empty() && g.morphs.is_empty(),
                    "geometry {name}: curves do not support indices, material_ranges or morphs yet");
                ensure!(g.subdivision.is_none() && g.crease_edge.is_empty(), "curves do not support mesh modifiers or creases");
                ensure!(g.attributes.contains_key("POSITION"), "curves require POSITION");
                ensure!(!g.attributes.keys().any(|a| matches!(a.as_str(), "NORMAL"|"TANGENT"|"TEXCOORD_0"|"TEXCOORD_1"|"JOINTS_0"|"WEIGHTS_0")),
                    "curves generate normals and UVs; authored normal/tangent/UV and skin attributes are unsupported");
            } else if matches!(g.topology, Topology::Points) {
                let widths = g.widths.as_ref().context("points require widths (diameter)")?;
                ensure!(lookup(widths)?.element_type == BufferType::F32, "point widths require f32");
                ensure!(g.curve_vertex_counts.is_none() && g.indices.is_none() && g.material_ranges.is_empty() && g.morphs.is_empty(),
                    "geometry {name}: points do not support curve counts, indices, material_ranges or morphs");
                ensure!(g.subdivision.is_none() && g.crease_edge.is_empty(),
                    "points do not support mesh modifiers or creases");
                ensure!(g.attributes.contains_key("POSITION"), "points require POSITION");
                ensure!(!g.attributes.keys().any(|a| matches!(a.as_str(), "NORMAL"|"TANGENT"|"TEXCOORD_0"|"TEXCOORD_1"|"JOINTS_0"|"WEIGHTS_0")),
                    "points generate normals and UVs; authored normal/tangent/UV and skin attributes are unsupported");
            } else {
                ensure!(g.curve_vertex_counts.is_none() && g.widths.is_none(),
                    "curve_vertex_counts requires curves; widths requires curves or points topology");
            }
            for (attr, buffer) in &g.attributes {
                let b = lookup(buffer)?;
                if is_color_attribute(attr) {
                    ensure!(matches!(b.element_type, BufferType::Vec3 | BufferType::Vec4),
                        "geometry {name}: {attr} buffer has wrong type; expected Vec3 or Vec4");
                } else if let Some((_, expected)) = STANDARD_ATTRIBUTES.iter().find(|(n, _)| n == attr) {
                    ensure!(
                        b.element_type == *expected,
                        "geometry {name}: {attr} buffer has wrong type; expected {expected:?}"
                    );
                }
            }
            if let Some(index) = &g.indices {
                ensure!(
                    matches!(lookup(index)?.element_type, BufferType::U32),
                    "indices require u32"
                );
            }
            ensure!(g.subdivision.is_none() || g.material_ranges.is_empty(), "geometry {name}: subdivision requires one material (no material_ranges)");
            if let Some(sizes) = &g.face_vertex_counts {
                ensure!(
                    matches!(g.topology, Topology::Triangles),
                    "geometry {name}: polygon cages require triangle topology"
                );
                ensure!(
                    !sizes.is_empty() && sizes.iter().all(|n| *n >= 3),
                    "geometry {name}: face_vertex_counts require at least three corners per polygon"
                );
                ensure!(
                    g.material_ranges.is_empty(),
                    "geometry {name}: polygon cages cannot have triangle material_ranges; subdivide with std first"
                );
                let corners: u64 = sizes.iter().map(|n| u64::from(*n)).sum();
                let count = g
                    .indices
                    .as_ref()
                    .and_then(|i| self.buffers[i].len())
                    .or_else(|| {
                        g.indices
                            .is_none()
                            .then(|| {
                                g.attributes
                                    .get("POSITION")
                                    .and_then(|p| self.buffers[p].len())
                            })
                            .flatten()
                    });
                if let Some(count) = count {
                    ensure!(
                        corners == count as u64,
                        "geometry {name}: face_vertex_counts cover {corners} of {count} corners"
                    );
                }
            }
            let mut edges = std::collections::BTreeSet::new();
            let vertices = g
                .attributes
                .get("POSITION")
                .and_then(|p| self.buffers.get(p))
                .and_then(Buffer::len);
            for edge in &g.crease_edge {
                let [a, b] = edge.vertices;
                ensure!(
                    edge.crease.is_finite() && edge.crease >= 0.0,
                    "geometry {name}: edge crease sharpness must be finite and nonnegative"
                );
                ensure!(
                    a != b && vertices.is_none_or(|n| (a as usize) < n && (b as usize) < n),
                    "geometry {name}: crease endpoints require distinct in-range vertices"
                );
                ensure!(
                    edges.insert((a.min(b), a.max(b))),
                    "geometry {name}: duplicate crease edge"
                );
            }
            if !g.material_ranges.is_empty() {
                ensure!(
                    g.material_ranges
                        .iter()
                        .all(|range| range.start % 3 == 0 && range.count % 3 == 0),
                    "geometry {name}: material_ranges require triangle-aligned starts and counts"
                );
                // Ranges partition the index range: every triangle draws exactly once.
                let mut ranges: Vec<_> = g
                    .material_ranges
                    .iter()
                    .map(|x| (u64::from(x.start), u64::from(x.start) + u64::from(x.count)))
                    .collect();
                ranges.sort();
                let mut next = 0;
                for (start, end) in ranges {
                    ensure!(
                        start == next,
                        "geometry {name}: material_ranges must partition the index range; {} at index {next}",
                        if start < next { "overlap" } else { "gap" }
                    );
                    next = end;
                }
                let total = match &g.indices {
                    Some(index) => lookup(index)?.len(),
                    None => g
                        .attributes
                        .get("POSITION")
                        .and_then(|p| self.buffers.get(p)?.len()),
                };
                if let Some(total) = total {
                    ensure!(
                        next == total as u64,
                        "geometry {name}: material_ranges cover {next} of {total} indices; they must partition the index range"
                    );
                }
            }
            for morph in g.morphs.values() {
                for name in std::iter::once(&morph.POSITION).chain(morph.NORMAL.iter()) {
                    ensure!(
                        matches!(lookup(name)?.element_type, BufferType::Vec3),
                        "morph deltas require vec3"
                    );
                }
            }
            Ok(())
        }).filter_map(Result::err));
        // Camera materials must opt into a ray program before surface-input validation.
        let mut camera_nodes: Vec<_> = self.nodes.iter().collect();
        let mut camera_materials = std::collections::BTreeSet::new();
        while let Some(node) = camera_nodes.pop() {
            let result = (|| -> Result<()> {
            camera_nodes.extend(&node.children);
            if let NodeKind::Camera { material: Some(name), .. } = &node.kind {
                camera_materials.insert(name.as_str());
                if let Some(material) = self.materials.get(name) {
                    ensure!(material.program != "std/openpbr",
                        "camera {:?}: material {name} selects std/openpbr; camera materials require a camera program. Set program:\"std/camera\" beside inputs (e.g. lens), or omit camera.material for default optics. See `vork help cameras`", node.name);
                }
            }
            Ok(())
            })();
            if let Err(error) = result {errors.push(error);}
        }
        for (name, m) in &self.materials {
            if !self.programs.contains_key(&m.program)
                && let Some(message) = crate::libraries::removed_module(&m.program) {
                bail!("{message}");
            }
            let result = (|| -> Result<()> {
            if let Some(module) = crate::libraries::modules().find(|module| module.name == m.program) {
                ensure!(module.material, "material {name}: {} is a library-only module, not a material program", m.program);
            }
            ensure!(
                self.programs.contains_key(&m.program)
                    || crate::libraries::material_programs().any(|name| name == m.program),
                "{}",
                diagnostics::unknown(
                    "program",
                    &m.program,
                    self.programs.keys().map(String::as_str).chain(crate::libraries::material_programs())
                )
            );
            Ok(())
            })();
            if let Err(error) = result { errors.push(anyhow::anyhow!("{error:#}; material {name}: choose a material program (vork help programs --all)")); }
            for (key, value) in &m.inputs {
                let result = (|| -> Result<()> {
                // A rejected camera binding cannot interpret its inputs as a surface.
                if m.program == "std/openpbr" && !camera_materials.contains(name.as_str()) {
                    let p =
                        validate_openpbr_input(key).with_context(|| format!("material {name}"))?;
                    validate_openpbr_value(p, value)
                        .map_err(|issue| anyhow::anyhow!("material {name}: {issue}"))?;
                }
                match value {
                    Input::Geometry(GeometryBinding {geometry}) => {
                        ensure!(self.programs.get(&m.program).is_some_and(|program|program.is_compute()),
                            "material {name}: input {key}: Geometry requires a compute program; bind the collider on the geometry producer");
                        let target=nodes_by_name.get(geometry.as_str()).with_context(||format!("material {name}: input {key}: {}; name a native mesh object",diagnostics::unknown("mesh node",geometry,nodes_by_name.keys().copied())))?;
                        ensure!(matches!(target.kind,NodeKind::Mesh {..}),"material {name}: input {key}: geometry reference {geometry:?} requires a native mesh; use a named triangle collision proxy");
                    },
                    Input::Grid(GridBinding { grid }) => ensure!(
                        self.buffers.get(grid).is_some_and(|b| b.is_volume_grid()),
                        "material {name}: input {key}: {}",
                        diagnostics::unknown("grid buffer", grid,
                            self.buffers.iter().filter(|(_, b)| b.is_volume_grid()).map(|(n, _)| n.as_str()))
                    ),
                    Input::Aov(AovBinding { aov }) => ensure!(
                        self.render.aovs.iter().any(|a| &a.name == aov),
                        "material {name}: input {key}: {}",
                        diagnostics::unknown("render.aovs entry", aov, self.render.aovs.iter().map(|a| a.name.as_str()))
                    ),
                    _ => {}
                }
                if let Input::Texture(t) = value {
                    ensure!(
                        self.images.contains_key(&t.image),
                        "{}",
                        diagnostics::unknown(
                            "image",
                            &t.image,
                            self.images.keys().map(String::as_str)
                        )
                    );
                }
                Ok(())
                })();
                if let Err(error) = result {
                    errors.push(error);
                }
            }
        }
        fn nodes(scene: &Scene, list: &[Node]) -> Result<()> {
            let mut errors = Vec::new();
            for n in list {
                let result = (|| -> Result<()> {
                let t = &n.transform;
                ensure!(
                    t.translation
                        .iter()
                        .chain(&t.resolved_rotation())
                        .chain(&t.scale)
                        .all(|x| x.is_finite())
                        && glam::Quat::from_array(t.resolved_rotation()).length_squared()
                            > constants::NONDEGENERATE_EPSILON,
                    "node {:?}: transform must be finite with a nonzero rotation",
                    n.name
                );
                if let Some(AimTarget::Point(point)) = &n.target {
                    ensure!(point.iter().all(|x|x.is_finite()), "node {:?}: target point must be finite",n.name);
                }
                scene.validate_node_time(n)?;
                match &n.kind {
                    NodeKind::Mesh {
                        geometry,
                        materials,
                        instances,
                        skin,
                        morph_weights,
                        ..
                    } => {
                        let g = scene.geometry.get(geometry).with_context(|| {
                            diagnostics::unknown(
                                "geometry",
                                geometry,
                                scene.geometry.keys().map(String::as_str),
                            )
                        })?;
                        ensure!(
                            morph_weights.keys().all(|name| g.morphs.contains_key(name)),
                            "mesh {:?}: unknown morph weight name",
                            n.name
                        );
                        if matches!(g.topology, Topology::Volume | Topology::Implicit) {
                            ensure!(skin.is_none() && morph_weights.is_empty(), "volume and implicit geometry cannot have skin or morph weights");
                        }
                        let joints = g.attributes.get("JOINTS_0");
                        let weights = g.attributes.get("WEIGHTS_0");
                        if skin.is_some() {
                            ensure!(
                                joints.is_some() && weights.is_some(),
                                "mesh {:?}: skin requires JOINTS_0 and WEIGHTS_0 attributes",
                                n.name
                            );
                        } else if !matches!(g.topology, Topology::Volume | Topology::Implicit) {
                            ensure!(
                                joints.is_none() && weights.is_none(),
                                "mesh {:?}: joints/weights require a skin",
                                n.name
                            );
                        }
                        for m in materials {
                            if let Some(material) = scene.materials.get(m) {
                                for (input, value) in &material.inputs {
                                    if let Input::Attribute(binding) = value {
                                        let vertex = g.attributes.get(&binding.attribute);
                                        let instance = instances.as_ref().and_then(|i| i.attributes.get(&binding.attribute));
                                        ensure!(vertex.is_none() || instance.is_none(), "attribute {:?} exists on both geometry and instances", binding.attribute);
                                        let buffer = vertex.or(instance).with_context(|| format!(
                                            "mesh {:?}: material {m} input {input}: missing attribute {:?}", n.name, binding.attribute))?;
                                        let source = scene.buffers.get(buffer).with_context(|| diagnostics::unknown(
                                            "attribute buffer", buffer, scene.buffers.keys().map(String::as_str)))?;
                                        ensure!(source.grid.is_none() && source.dims.is_none(), "surface attribute bindings cannot read volume grids; use a medium program");
                                        let ty = source.element_type;
                                        ensure!(
                                            matches!(
                                                ty,
                                                BufferType::F32
                                                    | BufferType::Vec2
                                                    | BufferType::Vec3
                                                    | BufferType::Vec4
                                            ),
                                            "attribute {} requires f32 or vec2/3/4",
                                            binding.attribute
                                        );
                                    }
                                }
                            }
                            ensure!(
                                scene.materials.contains_key(m),
                                "{}",
                                diagnostics::unknown(
                                    "material",
                                    m,
                                    scene
                                        .materials
                                        .keys()
                                        .map(String::as_str)
                                )
                            );
                        }
                        ensure!(
                            g.material_ranges
                                .iter()
                                .all(|r| (r.material_index as usize) < materials.len().max(1)),
                            "mesh {:?}: material_ranges material_index exceeds its {} materials", n.name, materials.len().max(1)
                        );
                        if let Some(i) = instances {
                            if let Some(count) = &i.count {
                                ensure!(
                                    scene
                                        .buffers
                                        .get(count)
                                        .is_some_and(|b| matches!(b.element_type, BufferType::U32)),
                                    "instance count requires a u32 buffer"
                                );
                                ensure!(
                                    scene.programs.values().any(|p| p.is_compute()
                                        && p.bindings.values().any(|b| b == count)),
                                    "instance count requires a compute binding"
                                );
                            }
                            ensure!(!i.attributes.is_empty(), "instances.attributes must contain at least one buffer column (TRANSLATION vec3, ROTATION vec4 xyzw, SCALE vec3, or a custom attribute); columns must have equal lengths");
                            let count = i.attributes.values().filter_map(|name| scene.buffers.get(name).and_then(|b| b.len())).next();
                            for (attribute, b) in &i.attributes {
                                let buffer = scene.buffers.get(b).with_context(|| {
                                    diagnostics::unknown(
                                        "instance buffer",
                                        b,
                                        scene.buffers.keys().map(String::as_str),
                                    )
                                })?;
                                let expected = match attribute.as_str() {
                                    "TRANSLATION" | "SCALE" => Some(BufferType::Vec3),
                                    "ROTATION" => Some(BufferType::Vec4),
                                    "time_offset" => Some(BufferType::F32),
                                    "pass_index" => Some(BufferType::U32), _ => None,
                                };
                                if let Some(expected) = expected {
                                    ensure!(expected == buffer.element_type, "instances.attributes.{attribute}: buffer {b} has type {:?}; expected {:?}{}", buffer.element_type, expected, if attribute == "ROTATION" { " (xyzw)" } else { "" });
                                }
                                if let (Some(count), Some(len)) = (count, buffer.len()) {
                                    ensure!(
                                        len == count,
                                        "instances.attributes.{attribute}: buffer {b} has {len} elements but another instance column has {count}"
                                    );
                                }
                            }
                        }
                        if let Some(s) = skin {
                            ensure!(
                                scene
                                    .buffers
                                    .get(&s.inverse_bind_matrices)
                                    .is_some_and(|b| matches!(b.element_type, BufferType::Mat4)),
                                "inverse_bind_matrices requires mat4"
                            );
                        }
                    }
                    NodeKind::Asset {
                        file, materials, ..
                    } => {
                        ensure!(!file.is_empty(), "asset {:?}: file is empty", n.name);
                        for m in materials.values() {
                            ensure!(
                                scene.materials.contains_key(m),
                                "{}",
                                diagnostics::unknown(
                                    "material",
                                    m,
                                    scene
                                        .materials
                                        .keys()
                                        .map(String::as_str)
                                )
                            );
                        }
                    }
                    NodeKind::Camera { material, .. } => {
                        if let Some(material) = material {
                            ensure!(scene.materials.contains_key(material), "camera {:?}: unknown material {material}", n.name);
                        }
                    }
                    NodeKind::Light {
                        shape,
                        spot_size,
                        spot_blend,
                        color,
                        temperature,
                        energy: intensity,
                        image,
                        medium,
                        shadowed_by,
                        material,
                        angle,
                        radius,
                        size: width,
                        size_y: height,
                        ..
                    } => {
                        ensure!(
                            color.is_none_or(|c| c.iter().all(|v| v.is_finite() && *v >= 0.))
                                && intensity.is_none_or(|v| v.is_finite() && v >= 0.),
                            "invalid light radiance"
                        );
                        ensure!(
                            temperature.is_none_or(|t| (constants::LIGHT_TEMPERATURE_MIN..=constants::LIGHT_TEMPERATURE_MAX).contains(&t)),
                            "light {:?}: temperature must be {}..{} kelvin (Blender's Light.temperature range)",
                            n.name, constants::LIGHT_TEMPERATURE_MIN, constants::LIGHT_TEMPERATURE_MAX
                        );
                        ensure!(
                            temperature.is_none() || !matches!(shape, LightShape::Environment),
                            "light {:?}: temperature applies to sun, point, spot and area lights, not environment",
                            n.name
                        );
                        ensure!(
                            angle.is_finite()
                                && *angle >= 0.
                                && radius.is_finite()
                                && *radius >= 0.
                                && width.is_finite()
                                && *width > 0.
                                && height.is_none_or(|h| h.is_finite() && h > 0.),
                            "invalid light dimensions"
                        );
                        ensure!(matches!(shape, LightShape::Spot) || (*spot_size == constants::spot_size() && *spot_blend == constants::spot_blend()),
                            "light {:?}: spot_size/spot_blend apply only to spot lights", n.name);
                        ensure!(spot_blend.is_finite() && (0. ..=1.).contains(spot_blend), "spot_blend must be 0..1");
                        ensure!(spot_size.is_finite() && *spot_size > 0. && *spot_size < std::f32::consts::PI, "spot_size must be a full angle between 0 and pi radians");
                        ensure!(
                            *angle < std::f32::consts::PI,
                            "light {:?}: angle must be below pi",
                            n.name
                        );
                        ensure!(
                            matches!(shape, LightShape::Sun) || *angle == constants::sun_angle(),
                            "light {:?}: angle applies only to sun lights",
                            n.name
                        );
                        ensure!(
                            matches!(shape, LightShape::Rectangle | LightShape::Ellipse)
                                || (*width == constants::area_dimension()
                                    && height.is_none_or(|h| h == constants::area_dimension())),
                            "light {:?}: size/size_y apply only to rectangle or ellipse lights",
                            n.name
                        );
                        ensure!(
                            matches!(shape, LightShape::Point | LightShape::Spot) || *radius == 0.,
                            "light {:?}: radius applies only to point and spot lights",
                            n.name
                        );
                        if let Some(i) = image {
                            ensure!(
                                scene.images.contains_key(i),
                                "light image reference is invalid"
                            );
                        }
                        ensure!(!matches!(shape,LightShape::Environment) || shadowed_by.is_none(),"environment shadowed_by belongs on a declared light");
                        ensure!(!matches!(shape,LightShape::Environment) || image.is_none(),"environment images are ordinary program inputs");
                        if let Some(name)=image {
                            if !matches!(shape,LightShape::Environment) {
                                let asset=&scene.images[name];
                                ensure!(asset.file.as_ref().is_some_and(|file| std::path::Path::new(file).extension().is_some_and(|ext|ext.eq_ignore_ascii_case("ies"))), "local light.image requires an IES asset");
                            }
                        }
                        if let Some(p) = material {
                            ensure!(
                                matches!(shape, LightShape::Environment),
                                "light.material applies only to environment lights"
                            );
                            ensure!(
                                image.is_none(),
                                "environment image and material are mutually exclusive"
                            );
                            ensure!(
                                scene.materials.contains_key(p),
                                "unknown environment material {p}"
                            );
                        }
                        if let Some(m) = medium {
                            ensure!(
                                matches!(shape, LightShape::Environment),
                                "medium applies only to environment lights"
                            );
                            ensure!(
                                scene.materials.contains_key(m),
                                "{}",
                                diagnostics::unknown(
                                    "medium material",
                                    m,
                                    scene.materials.keys().map(String::as_str)
                                )
                            );
                        }
                    }
                    _ => {}
                }
                Ok(())
                })();
                if let Err(error) = result {errors.push(error);}
                if let Err(error) = nodes(scene, &n.children) {errors.push(error);}
            }
            diagnostics::joined(errors)
        }
        if let Err(error) = nodes(self, &self.nodes) {errors.push(error);}
        if let Err(error) = self.validate_targets(nodes_by_name) {errors.push(error);}
        // A canvas has no camera.
        if let Err(error) = self.validate_render_selection(&self.main_render(), nodes_by_name,require_camera && !self.is_canvas()) {errors.push(error);}
        diagnostics::joined(errors)
    }
}
impl Scene {
    /// Rule 12 at open: `active` windows are ordered, `time` constants are in range,
    /// and `loop` or a negative speed has an end to traverse (A7.4). The one
    /// source's length is known only at install; several or none are refused here.
    pub(super) fn validate_node_time(&self, node: &Node) -> Result<()> {
        if matches!(
            &node.kind,
            NodeKind::Mesh {
                instances: Some(_),
                ..
            }
        ) {
            ensure!(
                self.audio_sources(node).is_empty(),
                "node {:?}: instanced sound is not built yet",
                node.name
            );
        }
        if let Some(audio) = &node.audio {
            ensure!(
                audio.gain.is_finite(),
                "node {:?}: audio.gain must be finite",
                node.name
            );
            ensure!(
                audio.file.as_ref().is_none_or(|f| !f.is_empty()),
                "node {:?}: audio.file must not be empty",
                node.name
            );
        }
        if let Some((a, b)) = node.active.window() {
            let Active::Window([start, end]) = &node.active else { unreachable!() };
            ensure!(start.as_f64().is_finite() && end.as_f64().is_finite() && a < b, "node {:?}: active: [start, end) needs start < end", node.name);
        }
        if let Some(time) = &node.time {
            if let Some(line) = time.refusal() { anyhow::bail!("node {:?}: {line}", node.name); }
            if (time.looping || time.reverse()) && time.duration.is_none() {
                let sources = self.media_sources(node);
                ensure!(sources.len() + self.audio_sources(node).len() == 1, "node {:?}: time: {} write time.duration", node.name,
                    if sources.is_empty() { "loop and speed < 0 need an end;" } else { "several sources have no one length;" });
            }
        }
        Ok(())
    }
    /// Unique sound files in a subtree, for the node clock's default span.
    pub fn audio_sources(&self, node: &Node) -> std::collections::BTreeSet<String> {
        fn visit(node: &Node, out: &mut std::collections::BTreeSet<String>) {
            if let Some(file) = node.audio.as_ref().and_then(|a| a.file.as_ref()) {
                out.insert(file.clone());
            }
            for child in &node.children {
                visit(child, out);
            }
        }
        let mut out = std::collections::BTreeSet::new();
        visit(node, &mut out);
        out
    }
    /// Movie and frame-sequence images sampled by materials or paints of `node` and its subtree.
    pub fn media_sources(&self, node: &Node) -> std::collections::BTreeSet<String> {
        let mut out = std::collections::BTreeSet::new();
        fn visit(scene: &Scene, node: &Node, out: &mut std::collections::BTreeSet<String>) {
            out.extend(scene.node_media(node));
            for child in &node.children { visit(scene, child, out); }
        }
        visit(self, node, &mut out);
        out
    }
    /// The movies and frame sequences `node` itself shows through its materials: a mesh's, an
    /// asset's overrides, a camera's, an environment light's radiance and medium, or a path's paints (rule 7).
    pub fn node_media(&self, node: &Node) -> Vec<String> {
        let materials: Vec<&String> = match &node.kind {
            NodeKind::Mesh { materials, .. } => materials.iter().collect(),
            NodeKind::Asset { materials, .. } => materials.values().collect(),
            NodeKind::Camera { material, .. } => material.iter().collect(),
            NodeKind::Light { material, medium, .. } => material.iter().chain(medium).collect(),
            _ => Vec::new(),
        };
        let mut out = Vec::new();
        let paints: Vec<&Paint> = match &node.kind {
            NodeKind::Path { fill, stroke, .. } => vec![fill, stroke],
            NodeKind::Text { fill, stroke, content, .. } => {
                let mut paints = vec![fill, stroke];
                if let Content::Runs(runs) = content {
                    paints.extend(runs.iter().flat_map(|run| run.fill.iter().chain(&run.stroke)));
                }
                paints
            }
            _ => Vec::new(),
        };
        for paint in paints {
            if let Paint::Image(binding) = paint {
                if self.images.get(&binding.image).and_then(|image| image.file.as_deref()).is_some_and(is_media_file) && !out.contains(&binding.image) {
                    out.push(binding.image.clone());
                }
            }
        }
        if matches!(node.kind, NodeKind::Path { .. } | NodeKind::Text { .. })
            && self.images.values().any(|image| image.file.as_deref().is_some_and(is_media_file)) {
            for name in self.keyed_images(node) {
                if self.images.get(&name).and_then(|image| image.file.as_deref()).is_some_and(is_media_file) && !out.contains(&name) {
                    out.push(name);
                }
            }
        }
        for material in materials.into_iter().filter_map(|name| self.materials.get(name)) {
            for input in material.inputs.values() {
                if let Input::Texture(binding) = input {
                    if self.images.get(&binding.image).and_then(|image| image.file.as_deref()).is_some_and(is_media_file) && !out.contains(&binding.image) {
                        out.push(binding.image.clone());
                    }
                }
            }
        }
        out
    }
    /// One image under two clocks: a movie shown under a node clock is not also read on scene
    /// time by a compositor, the scene's or a render image's.
    pub(super) fn validate_media_clocks(&self) -> Result<()> {
        let mut readers = BTreeMap::new();
        for image in self.compositor_image_reads(self.compositor()) { readers.entry(image).or_insert_with(|| "composition.compositor".to_string()); }
        for (name, source) in &self.images {
            for image in source.render.iter().flat_map(|_| self.compositor_image_reads(source.compositor())) {
                readers.entry(image).or_insert_with(|| format!("images.{name}.composition.compositor"));
            }
        }
        fn visit(scene: &Scene, nodes: &[Node], clock: Option<&str>, readers: &BTreeMap<String, String>) -> Result<()> {
            for node in nodes {
                let clock = if node.has_clock() { Some(node.name.as_str()) } else { clock };
                if let Some(owner) = clock {
                    if let Some((image, reader)) = scene.node_media(node).into_iter().find_map(|image| readers.get(&image).map(|reader| (image, reader))) {
                        anyhow::bail!("image {image:?} is shown under node {owner:?}'s clock and read by {reader} on scene time; give the compositor its own image");
                    }
                }
                visit(scene, &node.children, clock, readers)?;
            }
            Ok(())
        }
        if readers.is_empty() { return Ok(()); }
        visit(self, &self.nodes, None, &readers)
    }
    pub(super) fn validate_render(&self, r: &Render, nodes: &BTreeMap<&str, &Node>) -> Result<()> {
        self.validate_render_selection(r,nodes,true)
    }
    fn validate_render_selection(&self,r:&Render,nodes:&BTreeMap<&str,&Node>,require_camera:bool)->Result<()> {
        let mut names=std::collections::BTreeSet::new();
        for aov in &r.aovs {
            ensure!(!aov.name.is_empty() && aov.name.bytes().all(|b|b.is_ascii_alphanumeric() || b==b'_' || b==b'-') && names.insert(&aov.name),"AOV name {:?} must be unique, nonempty and use only ASCII letters, digits, _ and -",aov.name);
            ensure!(!constants::OUTPUT_CHANNELS.contains(&aov.name.as_str()),
                "AOV name {:?} is reserved: it names a built-in output channel and EXR layer ({}); rename the AOV",
                aov.name, constants::OUTPUT_CHANNELS.join(", "));
        }
        if let Some(name) = &r.camera {
            validate_camera(name, nodes)?;
        } else {
            fn count(nodes: &[Node]) -> usize {
                nodes
                    .iter()
                    .map(|n| {
                        usize::from(matches!(n.kind, NodeKind::Camera { .. })) + count(&n.children)
                    })
                    .sum()
            }
            let cameras = count(&self.nodes);
            ensure!(
                !require_camera || cameras <= 1 || self.animation.iter().any(|layer| layer.blend == Blend::Replace
                    && layer.tracks.iter().flatten().any(|track| track.path == ["render", "camera"])),
                "a scene with {cameras} cameras requires render.camera"
            );
        }
        if let Some(root) = &r.root {
            ensure!(
                nodes.contains_key(root.as_str()),
                "render.root: unknown node {root}"
            );
        }
        ensure!(
            (1..=constants::MAX_RENDER_SAMPLES).contains(&r.settings.samples)
                && r.resolution.iter().all(|x| *x > 0)
                && r.fps.as_f32().is_finite()
                && r.fps > crate::Time::ZERO
                && r.duration.as_f32().is_finite()
                && r.duration.as_f32() > 0.,
            "render requires positive dimensions/time and samples in 1..=4096"
        );
        if let Some(m) = &r.motion_blur {
            ensure!(m.shutter.is_finite() && m.shutter >= 0., "render.motion_blur.shutter must be nonnegative frames");

        }
        let (_, display) = self.view_material()?;
        ensure!(self.programs.contains_key(&display.program) || crate::libraries::material_programs().any(|name| name == display.program),
            "composition.view: {}", diagnostics::unknown("program", &display.program,
                self.programs.keys().map(String::as_str).chain(crate::libraries::material_programs())));
        let mut earlier = std::collections::BTreeSet::new();
        for (index, entry) in r.compositor.iter().enumerate() {
            let name = entry.material();
            let material = self.materials.get(name).with_context(|| format!("composition.compositor[{index}]: unknown material {name}"))?;
            entry.extent(r.resolution)?;

            for input in material.inputs.values() {
                if let Input::Step(binding) = input {
                    ensure!(earlier.contains(binding.step.as_str()), "composition.compositor[{index}]: step {} must name an earlier entry", binding.step);
                }
            }
            if let Some(name) = entry.name() {
                ensure!(!name.is_empty() && earlier.insert(name), "composition.compositor[{index}]: empty or duplicate step name {name}");
            }
        }
        if let Background::Color(c) = &r.background {
            ensure!(
                c.iter().all(|v| v.is_finite() && *v >= 0.),
                "render.background colour must be finite and nonnegative"
            );
        }
        Ok(())
    }
}
pub(super) fn validate_camera(name: &str, nodes: &BTreeMap<&str, &Node>) -> Result<()> {
    ensure!(
        nodes
            .get(name)
            .is_some_and(|n| matches!(n.kind, NodeKind::Camera { .. }))
            || name
                .split_once('/')
                .is_some_and(|(asset, child)| !child.is_empty()
                    && nodes
                        .get(asset)
                        .is_some_and(|n| matches!(n.kind, NodeKind::Asset { .. }))),
        "{}",
        diagnostics::unknown(
            "camera node",
            name,
            nodes
                .iter()
                .filter(|(_, n)| matches!(n.kind, NodeKind::Camera { .. }))
                .map(|(name, _)| *name)
        )
    );
    Ok(())
}
/// The OpenPBR parameter a std/openpbr input or track names.
pub fn validate_openpbr_input(name: &str) -> Result<&'static OpenPbrParam> {
    if name == "base_diffuse_roughness" {
        bail!("base_diffuse_roughness is not supported; remove base_diffuse_roughness.");
    }
    if matches!(name, "transmission_scatter" | "transmission_scatter_anisotropy") {
        bail!("{name} is not supported; remove {name} and use a volume material.");
    }
    let p = OpenPbrParam::find(name).with_context(|| match principled_spelling(name) {
        Some(meant) => format!(
            "unknown OpenPBR parameter {name:?}; did you mean {meant:?}? (Principled and glTF names are not inputs)"
        ),
        None => diagnostics::unknown("OpenPBR parameter", name, OpenPbrParam::supported().map(|p| p.name)),
    })?;
    ensure!(p.supported, "{name} is not a std/openpbr input; a surface program sets Closure.{name}");
    Ok(p)
}
/// The OpenPBR name for a common Principled or glTF spelling (error text only, never accepted).
fn principled_spelling(name: &str) -> Option<&'static str> {
    let stem = name.strip_suffix("_factor").unwrap_or(name);
    let known = [
        ("color", "base_color"),
        ("metallic", "base_metalness"),
        ("roughness", "specular_roughness"),
        ("ior", "specular_ior"),
        ("emissive", "emission_color"),
        ("emissive_strength", "emission_luminance"),
        ("emission_strength", "emission_luminance"),
        ("opacity", "geometry_opacity"),
        ("alpha", "geometry_opacity"),
        ("transmission", "transmission_weight"),
        ("normal", "geometry_normal"),
        ("anisotropy", "specular_roughness_anisotropy"),
        ("sheen", "fuzz_weight"),
        ("clearcoat", "coat_weight"),
    ];
    known
        .iter()
        .find(|(k, _)| *k == stem)
        .map(|(_, v)| *v)
        .or_else(|| {
            let renamed = stem.replace("sheen", "fuzz").replace("clearcoat", "coat");
            OpenPbrParam::find(&renamed).map(|p| p.name)
        })
}
/// A material input failure retains its authored field path and the failed constraint.
#[derive(Debug)]
struct InputIssue {
    path: String,
    reason: InputReason,
}
#[derive(Debug)]
enum InputReason {
    Expected(&'static str),
    Length { actual: usize, shape: &'static str },
    NonFinite(f32),
    Range { value: f32, min: f32, max: f32, open_min: bool, open_max: bool },
}
impl std::fmt::Display for InputIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: ", self.path)?;
        match self.reason {
            InputReason::Expected(expected) => write!(f, "expected {expected}"),
            InputReason::Length { actual, shape } => write!(f, "expected {shape} triple (3 components), got {actual}"),
            InputReason::NonFinite(value) => write!(f, "{value} must be finite"),
            InputReason::Range { value, min, max, open_min, open_max } => write!(
                f, "{value} outside {}{min},{max}{}",
                if open_min { "(" } else { "[" }, if open_max { ")" } else { "]" }),
        }
    }
}
/// Constants and both binding factors use the same shape and range checks.
fn validate_openpbr_value(p: &OpenPbrParam, value: &Input) -> std::result::Result<(), InputIssue> {
    let issue = |path: &str, reason| InputIssue { path: path.into(), reason };
    let scalar = |path: &str, x: f32| {
        if !x.is_finite() {
            Err(issue(path, InputReason::NonFinite(x)))
        } else if !p.allows(x) {
            Err(issue(path, InputReason::Range { value: x, min: p.min, max: p.max,
                open_min: p.open_min, open_max: p.open_max }))
        } else { Ok(()) }
    };
    let numeric = |path: &str, number: Option<f32>, vector: Option<&[f32]>| {
        match p.kind {
            OpenPbrKind::Float => scalar(path, number.ok_or_else(|| issue(path, InputReason::Expected("a number")))?),
            OpenPbrKind::Color3 => {
                let v = vector.ok_or_else(|| issue(path, InputReason::Expected("an RGB triple")))?;
                if v.len() != 3 { return Err(issue(path, InputReason::Length { actual: v.len(), shape: "RGB" })); }
                for (i, &x) in v.iter().enumerate() { scalar(&format!("{path}[{i}]"), x)?; }
                Ok(())
            }
            // Normal-map factors scale XY and have no parameter range.
            OpenPbrKind::Vector3 => {
                if let Some(v) = vector {
                    if v.len() != 3 { return Err(issue(path, InputReason::Length { actual: v.len(), shape: "XYZ" })); }
                    for (i, &x) in v.iter().enumerate() {
                        if !x.is_finite() { return Err(issue(&format!("{path}[{i}]"), InputReason::NonFinite(x))); }
                    }
                    return Ok(());
                }
                let x = number.ok_or_else(|| issue(path, InputReason::Expected("a normal-map scale number or XYZ triple")))?;
                if x.is_finite() { Ok(()) } else { Err(issue(path, InputReason::NonFinite(x))) }
            }
            OpenPbrKind::Bool => Err(issue(path, InputReason::Expected("a boolean"))),
        }
    };
    let expected = match p.kind {
        OpenPbrKind::Bool => "a boolean",
        OpenPbrKind::Float => "a number, texture binding or attribute binding",
        OpenPbrKind::Color3 => "an RGB triple, texture binding or attribute binding",
        OpenPbrKind::Vector3 => "a finite tangent-space XYZ triple or a texture binding",
    };
    let factor = match value {
        Input::Bool(_) if p.kind == OpenPbrKind::Bool => return Ok(()),
        Input::Number(x) if p.kind == OpenPbrKind::Float => return numeric(p.name, Some(*x), None),
        Input::Vector(v) if p.kind == OpenPbrKind::Color3 => return numeric(p.name, None, Some(v)),
        Input::Vector(v) if p.kind == OpenPbrKind::Vector3 => return numeric(p.name, None, Some(v)),
        Input::Texture(_) if p.kind != OpenPbrKind::Bool => return Ok(()),
        Input::Attribute(t) if matches!(p.kind, OpenPbrKind::Float | OpenPbrKind::Color3) => {
            if t.attribute.is_empty() {
                return Err(issue(&format!("{}.attribute", p.name), InputReason::Expected("a nonempty attribute name")));
            }
            if p.kind == OpenPbrKind::Float && matches!(t.channel, Some(Channel::Rgb)) {
                return Err(issue(&format!("{}.channel", p.name), InputReason::Expected("a scalar channel (r, g, b or a)")));
            }
            &t.factor
        }
        _ => return Err(issue(p.name, InputReason::Expected(expected))),
    };
    match factor {
        None => Ok(()),
        Some(Factor::Number(x)) => numeric(&format!("{}.factor", p.name), Some(*x), None),
        Some(Factor::Vector(v)) => numeric(&format!("{}.factor", p.name), None, Some(v)),
    }
}

#[cfg(test)]
mod hemisphere_tests {
    use super::*;

    #[test]
    fn diffuse_roughness_stays_refused_until_its_brdf_is_implemented() {
        let error=validate_openpbr_input("base_diffuse_roughness").err().expect("unshaded diffuse roughness accepted").to_string();
        assert_eq!(error,"base_diffuse_roughness is not supported; remove base_diffuse_roughness.");
    }

    #[test]
    fn transmission_scattering_names_are_refused_without_spelling_suggestions() {
        for name in ["transmission_scatter", "transmission_scatter_anisotropy"] {
            let error=validate_openpbr_input(name).err().expect("unshaded scattering parameter accepted").to_string();
            assert_eq!(error,format!("{name} is not supported; remove {name} and use a volume material."));
        }
    }

    #[test]
    fn supported_openpbr_descriptors_accept_documented_bindings() {
        for p in OpenPbrParam::supported() {
            assert_eq!(validate_openpbr_input(p.name).unwrap().name, p.name);
            let value = match p.kind {
                OpenPbrKind::Bool => serde_json::json!(p.default[0] != 0.),
                OpenPbrKind::Float => serde_json::json!(p.default[0]),
                OpenPbrKind::Color3 => serde_json::json!(p.default),
                OpenPbrKind::Vector3 => serde_json::json!({"image":"normal"}),
            };
            let input: Input = crate::v2::from_test_value(value).unwrap();
            assert!(validate_openpbr_value(p, &input).is_ok(), "{}", p.name);
            if p.kind == OpenPbrKind::Vector3 {
                assert!(validate_openpbr_value(p, &Input::Vector(vec![0., 0., 1.])).is_ok());
            }
        }
    }

    #[test]
    fn numeric_normal_inputs_retain_structured_failures() {
        let p = OpenPbrParam::find("geometry_normal").unwrap();
        for (input, path, finite) in [
            (Input::Vector(vec![0., f32::INFINITY, 1.]), "geometry_normal[1]", true),
            (Input::Vector(vec![0., 1.]), "geometry_normal", false),
        ] {
            let issue = validate_openpbr_value(p, &input).unwrap_err();
            assert_eq!(issue.path, path);
            if finite { assert!(matches!(issue.reason, InputReason::NonFinite(_))); }
            else { assert!(matches!(issue.reason, InputReason::Length { actual: 2, shape: "XYZ" })); }
        }

    }

    #[test]
    fn material_input_errors_identify_binding_factor_and_component() {
        let validate = |value| {
            let scene: Scene = crate::v2::from_test_value(serde_json::json!({
                "version":2,"nodes":[{"type":"camera"}],
                "materials":{"m":{"program":"std/openpbr","inputs":{"subsurface_color":value}}}
            })).unwrap();
            scene.validate()
        };
        for (factor, expected) in [
            (serde_json::json!([1.6,1.6,1.2]), "subsurface_color.factor[0]: 1.6 outside [0,1]"),
            (serde_json::json!([0.6,0.6]), "subsurface_color.factor: expected RGB triple (3 components), got 2"),
            (serde_json::json!(0.6), "subsurface_color.factor: expected an RGB triple"),
        ] {
            let value = serde_json::json!({"attribute":"COLOR_0","factor":factor});
            let error = validate(value).unwrap_err().to_string();
            assert_eq!(error, format!("material m: {expected}"));
        }
        validate(serde_json::json!({"attribute":"COLOR_0","factor":[0.6,0.6,0.2]})).unwrap();
        assert_eq!(validate(serde_json::json!([0.6,1.6,0.2])).unwrap_err().to_string(),
            "material m: subsurface_color[1]: 1.6 outside [0,1]");
    }

    #[test]
    fn material_binding_validation_preserves_descriptor_contracts() {
        for p in OpenPbrParam::supported() {
            for binding in ["image", "attribute"] {
                let factor = match p.kind {
                    OpenPbrKind::Float => serde_json::json!(p.default[0]),
                    OpenPbrKind::Color3 => serde_json::json!(p.default),
                    OpenPbrKind::Vector3 => serde_json::json!(-2.),
                    OpenPbrKind::Bool => serde_json::json!(1.),
                };
                for with_factor in [false, true] {
                    let mut value = serde_json::json!({binding:"source"});
                    if with_factor && binding=="attribute" { value["factor"] = factor.clone(); }
                    let input = crate::v2::from_test_value(value).unwrap();
                    let accepted = p.kind != OpenPbrKind::Bool
                        && !(binding == "attribute" && p.kind == OpenPbrKind::Vector3);
                    assert_eq!(validate_openpbr_value(p, &input).is_ok(), accepted,
                        "{} {binding} factor={with_factor}", p.name);
                }
            }
        }
        let p = OpenPbrParam::find("base_weight").unwrap();
        for (value, path) in [
            (serde_json::json!({"attribute":""}), "base_weight.attribute"),
            (serde_json::json!({"attribute":"COLOR_0","channel":"rgb"}), "base_weight.channel"),
            (serde_json::json!({"attribute":"COLOR_0","factor":2.}), "base_weight.factor"),
        ] {
            let input = crate::v2::from_test_value(value).unwrap();
            assert_eq!(validate_openpbr_value(p, &input).unwrap_err().path, path);
        }
        for p in OpenPbrParam::supported().filter(|p| p.kind == OpenPbrKind::Float) {
            for value in [p.min, p.max, f32::NAN, f32::INFINITY] {
                let result = validate_openpbr_value(p, &Input::Number(value));
                assert_eq!(result.is_ok(), p.allows(value), "{}: {value}", p.name);
                if !value.is_finite() {
                    assert!(matches!(result.unwrap_err().reason, InputReason::NonFinite(_)));
                }
            }
        }
    }

    #[test]
    fn openpbr_constants_obey_parameter_ranges() {
        for inputs in [serde_json::json!({"transmission_weight": 7.}),
            serde_json::json!({"transmission_weight": 1e39}),
            serde_json::json!({"base_color": [5., -1., 1e39]})] {
            let scene: Scene = crate::v2::from_test_value(serde_json::json!({
                "version":2,"nodes":[{"type":"camera"}],"materials":{"m":{"program":"std/openpbr","inputs":inputs}}
            })).unwrap();
            assert!(scene.validate().is_err(), "{inputs}");
        }
    }

    #[test]
    fn object_focus_overrides_unused_manual_distance() {
        let scene: Scene = crate::v2::from_test_value(serde_json::json!({
            "version":2,"nodes":[{"type":"camera","focus_object":"subject","material":{"program":"std/camera","inputs":{"lens":50.,"focus_distance":0.01,"use_dof":true}}},
                {"type":"node","name":"subject","translation":[0.,0.,-2.]}]
        })).unwrap();
        scene.validate().unwrap();
    }

    #[test]
    fn zero_scale_is_valid_but_nonfinite_trs_and_zero_rotation_are_not() {
        let mut scene: Scene = crate::v2::from_test_value(serde_json::json!({
            "version":2,"nodes":[{"type":"node","scale":[0,1,1]},{"type":"camera"}]})).unwrap();
        scene.validate().unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            scene.nodes[0].transform.scale = [value, 1., 1.];
            assert!(scene.validate().is_err());
            scene.nodes[0].transform.scale = [0., 1., 1.];
            scene.nodes[0].transform.translation = [value, 0., 0.];
            assert!(scene.validate().is_err());
            scene.nodes[0].transform.translation = [0.; 3];
        }
        // Rotations are normalized; rounded quaternions such as 0.707 stay valid, zero length does not.
        scene.nodes[0].transform.rotation = Some([0.,0.707,0.,0.707]);
        scene.validate().unwrap();
        scene.nodes[0].transform.rotation = Some([0.,0.,0.,0.]);
        assert!(scene.validate().is_err());
    }

    #[test]
    fn node_transforms_are_flat_and_rotation_has_two_forms() {
        // The glTF fields sit directly on the node; there is no transform wrapper.
        assert!(crate::v2::from_test_value::<Scene>(serde_json::json!({
            "version":2,"nodes":[{"type":"node","transform":{"translation":[0,0,1]}}]})).is_err());
        let parse = |rotation: serde_json::Value| crate::v2::from_test_value::<Scene>(serde_json::json!({
            "version":2,"nodes":[{"type":"node","rotation":rotation}]}));
        let euler = parse(serde_json::json!({"euler":[0.3,-0.4,1.2]})).unwrap().nodes[0].transform.resolved_rotation();
        let expected = (glam::Quat::from_rotation_z(1.2) * glam::Quat::from_rotation_y(-0.4) * glam::Quat::from_rotation_x(0.3)).to_array();
        assert!(euler.iter().zip(expected).all(|(a, b)| (a - b).abs() < 1e-6), "{euler:?}");
        let blender = parse(serde_json::json!({"euler":[std::f32::consts::FRAC_PI_2,0,0]})).unwrap().nodes[0].transform.matrix();
        assert!((blender.transform_vector3(glam::Vec3::Y) - glam::Vec3::Z).length() < 1e-6, "X is applied about the parent's X");
        for bad in [serde_json::json!([0,0,0]), serde_json::json!({"euler":[0,0]}), serde_json::json!({"euler":[0,0,0],"order":"ZYX"})] {
            let error = parse(bad).unwrap_err().to_string();
            assert!(error.contains("euler"), "{error}");
        }
    }

    #[test]
    fn geometry_status_requires_scalar_u32_and_named_failure() {
        let value=serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "geometry":{"g":{"attributes":{"POSITION":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
                "status":{"buffer":{"type":"u32","data":[0]},"message":"solver exhausted capacity; reduce contact distance"}}}});
        Scene::from_json(value.to_string().as_bytes()).unwrap().validate().unwrap();
        for (buffer,message) in [
            (serde_json::json!({"type":"f32","data":[0]}),"failure"),
            (serde_json::json!({"type":"u32","data":[0,0]}),"failure"),
            (serde_json::json!({"type":"u32","data":[0]})," "),
        ] {
            let mut invalid=value.clone();
            invalid["geometry"]["g"]["status"]=serde_json::json!({"buffer":buffer,"message":message});
            assert!(Scene::from_json(invalid.to_string().as_bytes()).and_then(|s|s.validate()).is_err());
        }
    }

    #[test]
    fn display_is_a_material_and_legacy_tracks_are_rejected() {
        let input = serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "materials":{"grade":{"program":"std/agx","inputs":{"exposure":0}}},
            "composition":{"space":"3d","view":"grade"},
            "animation":[{"tracks":[{"path":["materials","grade","inputs","exposure"],"keys":[[0,0],[1,2]]}]}]});
        let scene: Scene = crate::v2::from_test_value(input.clone()).unwrap();
        scene.validate().unwrap();
        for property in ["view_transform", "exposure", "gamma", "dither_intensity"] {
            let mut invalid = input.clone();
            invalid["animation"][0]["tracks"][0]["path"] = serde_json::json!(["render",property]);
            let invalid: Scene = crate::v2::from_test_value(invalid).unwrap();
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn target_points_are_fixed_world_vec3_values() {
        let mut input: Scene=crate::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera","target":[0,0,0],"translation":[0,0,5]}]})).unwrap();
        input.validate().unwrap();
        let copy:Scene=crate::v2::from_test_value(serde_json::to_value(&input).unwrap()).unwrap();
        copy.validate().unwrap();
        input.nodes[0].target=Some(AimTarget::Point([f32::NAN,0.,0.]));
        assert!(input.validate().unwrap_err().to_string().contains("target point must be finite"));
    }
    #[test]
    fn authored_programs_can_use_retired_dotted_names() {
        for name in ["std.particles", "post.white_balance"] {
            let input = serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
                "programs":{name:{"source":"import vork; struct Inputs {}; import std.view; ViewResult view(ViewContext c, Inputs i) { return finish_view(float3(1, 0, 0),c,1.); }"}},
                "composition":{"space":"3d","view":{"program":name}}});
            Scene::from_json(input.to_string().as_bytes()).unwrap().validate().unwrap();
        }
    }

    #[test]
    fn removed_library_programs_refuse_with_replacement() {
        for (name, replacement) in [("post/white_balance", "std.white_balance"), ("post.white_balance", "std.white_balance"),
            ("std/particles", "fluid.particles"), ("std.particles", "fluid.particles")] {
            let scene:Scene=crate::v2::from_test_value(serde_json::json!({"version":2,
                "nodes":[{"type":"camera"}],"materials":{"unused":{"program":name}}})).unwrap();
            let error = scene.validate().unwrap_err().to_string();
            assert_eq!(error.lines().count(), 1, "{error}");
            assert!(error.contains(replacement) && error.contains("vork help upgrade"), "{error}");
        }
    }

    #[test]
    fn library_modules_are_not_material_programs() {
        for module in crate::libraries::modules().filter(|module| !module.material) {
            let input: Scene = crate::v2::from_test_value(serde_json::json!({"version":2,
                "nodes":[{"type":"camera"}],"materials":{"test":{"program":module.name}}})).unwrap();
            let error = input.validate().unwrap_err().to_string();
            assert!(error.contains("library-only module"), "{}: {error}", module.name);
        }
    }

    #[test]
    fn independent_buffer_and_material_errors_are_reported_together() {
        let scene: Scene = crate::v2::from_test_value(serde_json::json!({"version":2,
            "nodes":[{"type":"camera"}],
            "buffers":{"empty":{"type":"f32","data":[]},"unaligned":{"type":"f32","data":[1],"offset":2}},
            "materials":{"first":{"program":"unknown-first"},"second":{"program":"std/color"}}})).unwrap();
        let error = scene.validate().unwrap_err().to_string();
        for name in ["buffer empty", "buffer unaligned", "unknown-first", "library-only module"] {
            assert!(error.contains(name), "{error}");
        }
    }

    #[test]
    fn targets_compose_authored_and_animated_rotation_and_environment_accepts_null_shadow() {
        for kind in ["node", "camera", "light"] {
            let mut node = serde_json::json!({"name":"aimed","type":kind,"target":"subject","translation":[0,0,4]});
            if kind == "light" { node["shape"] = serde_json::json!("point"); }
            let scene = |node: serde_json::Value| serde_json::json!({"version":2,"render":{"camera":"camera"},"nodes":[
                {"type":"camera","name":"camera"},{"type":"node","name":"subject"},node]});
            let input: Scene = crate::v2::from_test_value(scene(node.clone())).unwrap();
            input.validate().unwrap();
            let mut animated = scene(node.clone());
            animated["animation"] = serde_json::json!([{"tracks":[{"path":["nodes","aimed","rotation"],"keys":[[0,[0,0,0,1]]]}]}]);
            let animated: Scene = crate::v2::from_test_value(animated).unwrap();
            animated.validate().unwrap();
            crate::v2::from_test_value::<Scene>(serde_json::to_value(input).unwrap()).unwrap().validate().unwrap();
            for rotation in [[0.,0.,0.,1.], [0.,1.,0.,0.]] {
                node["rotation"] = serde_json::json!(rotation);
                let input: Scene = crate::v2::from_test_value(scene(node.clone())).unwrap();
                input.validate().unwrap();
            }
            node["rotation"] = serde_json::Value::Null;
            assert!(crate::v2::from_test_value::<Scene>(scene(node)).is_err());
        }
        let input: Scene = crate::v2::from_test_value(serde_json::json!({"version":2,"nodes":[
            {"type":"camera"},{"type":"light","shape":"environment","shadowed_by":[]}]})).unwrap();
        assert!(input.validate().unwrap_err().to_string().contains("declared light"));
    }

    #[test]
    fn default_material_is_implicit_and_names_are_ordinary() {
        let mut json = serde_json::json!({"version":2,
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p"},"material_ranges":[{"material_index":0,"start":0,"count":3}]}},
            "nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"mesh","geometry":"g"}]});
        let validate = |json: serde_json::Value| crate::v2::from_test_value::<Scene>(json).unwrap().validate();
        validate(json.clone()).unwrap();
        json["nodes"][1]["materials"] = serde_json::json!([]);
        validate(json.clone()).unwrap();
        json["nodes"][1]["materials"] = serde_json::json!(["std/default"]);
        assert!(validate(json.clone()).is_err());
        json["nodes"][1]["materials"] = serde_json::json!(["plain"]);
        json["materials"] = serde_json::json!({"plain":{"program":"std/openpbr"}});
        validate(json).unwrap();
    }


    #[test]
    fn per_light_shadow_is_rejected() {
        let json = serde_json::json!({"version":2,"nodes":[{"type":"light","name":"sun","shape":"sun","shadow":{"method":"ray"}}]});
        assert!(crate::v2::from_test_value::<Scene>(json).is_err());
    }


}


#[cfg(test)]
mod orthographic_sensor_tests {
    use super::*;
    #[test]
    fn camera_without_program_reports_camera_context() {
        for name in ["eye", ""] {
            let value = serde_json::json!({"version":2,"nodes":[{"type":"camera","name":name,"material":{"inputs":{"lens":35}}}]});
            let scene = crate::v2::from_test_value::<Scene>(value).unwrap();
            let error = format!("{:#}", scene.validate().unwrap_err());
            assert!(error.contains("camera") && error.contains("std/camera"), "{error}");
            assert!(!error.contains("unknown OpenPBR"), "{error}");
        }
    }
    #[test]
    fn orthographic_cameras_use_fit_and_reject_old_names() -> Result<()> {
        for fit in ["auto", "vertical", "horizontal"] {
            let value = serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":4,"fit":fit}}}]});
            crate::v2::from_test_value::<Scene>(value)?.validate()?;
        }
        for (key, fit) in [("sensor_fit", "vertical")] {
            let value = serde_json::json!({"version":2,"nodes":[{"type":"camera",key:fit,"material":{"program":"std/camera","inputs":{"projection":"ortho","ortho_scale":4}}}]});
            assert!(crate::v2::from_test_value::<Scene>(value).is_err(), "{key}={fit}");
        }
        let mut old = serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}]});
        old["nodes"][0]["sensor"] = serde_json::json!([36,24]);
        assert!(crate::v2::from_test_value::<Scene>(old).is_err());
        Ok(())
    }

}

#[cfg(test)]
mod group_alignment_tests {
    use super::*;
    #[test]
    fn groups_partition_whole_triangles_at_validation() -> Result<()> {
        let mut json = serde_json::json!({"version":2,"nodes":[{"type":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0],[0,0,0],[1,0,0],[0,1,0]]}},
            "geometry":{"g":{"attributes":{"POSITION":"p"},"material_ranges":[{"start":0,"count":3,"material_index":0},{"start":3,"count":3,"material_index":0}]}}});
        crate::v2::from_test_value::<Scene>(json.clone())?.validate()?;
        json["geometry"]["g"]["material_ranges"][0]["count"] = serde_json::json!(4);
        json["geometry"]["g"]["material_ranges"][1]["start"] = serde_json::json!(4);
        json["geometry"]["g"]["material_ranges"][1]["count"] = serde_json::json!(2);
        let error = crate::v2::from_test_value::<Scene>(json)?
            .validate()
            .unwrap_err();
        assert!(error.to_string().contains("triangle-aligned"));
        Ok(())
    }
    #[test]
    fn a_clocked_movie_read_by_a_compositor_is_refused() -> Result<()> {
        // Named follow-up "one image under two clocks": the compositor reads scene time.
        let scene = |time: serde_json::Value, read: &str| -> Result<Scene> {
            Ok(crate::v2::from_test_value(serde_json::json!({"version":2,
                "nodes":[{"type":"camera","name":"cam","material":{"program":"std/camera","inputs":{"projection":"persp"}}},
                    {"name":"plate","time":time,"children":[{"name":"screen","type":"mesh","geometry":"g","materials":["screen"]}]}],
                "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[0,1,0]]}},
                "geometry":{"g":{"attributes":{"POSITION":"p"}}},
                "images":{"ramp":{"file":"ramp.mp4"},"ramp_scene":{"file":"ramp.mp4"}},
                "programs":{"overlay":{"source":"import vork; struct Inputs {Image plate;}; float4 composite(CompositeContext c, Inputs i) {return compositor_sample(i.plate,c.uv);}"}},
                "materials":{"screen":{"inputs":{"base_color":{"image":"ramp"}}},"comp":{"program":"overlay","inputs":{"plate":{"image":read}}}},
                "composition":{"space":"3d","compositor":["comp"]},"render":{}}))?)
        };
        let error = scene(serde_json::json!({"offset":1}), "ramp")?.validate().unwrap_err();
        assert_eq!(format!("{error:#}"), "image \"ramp\" is shown under node \"plate\"'s clock and read by composition.compositor on scene time; give the compositor its own image");
        scene(serde_json::json!({"offset":1}), "ramp_scene")?.validate()?;
        scene(serde_json::Value::Null, "ramp")?.validate()?;
        Ok(())
    }
}

#[cfg(all(test, vork_private_tests))]
include!("validate_compute_validation_tests_private.rs");

#[cfg(test)]
mod environment_program_tests {
    use super::*;
    #[test]
    fn environment_program_references_and_sources() {
        let obsolete = serde_json::json!({"version":2,"nodes":[{"type":"camera"},
            {"type":"light","shape":"environment","program":"daylight"}]});
        let error = crate::v2::from_test_value::<Scene>(obsolete).unwrap_err().to_string();
        assert!(error.contains("program"), "{error}");

        let mut value = serde_json::json!({"version":2,"nodes":[{"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}},{"type":"light","name":"world","shape":"environment","material":"daylight"}],"materials":{"daylight":{"program":"std/sky","inputs":{}}}});
        let check = |value: &serde_json::Value| {
            crate::v2::from_test_value::<Scene>(value.clone())
                .unwrap()
                .validate()
        };
        check(&value).unwrap();
        value["nodes"][1]["image"] = serde_json::json!("hdr");
        value["images"] = serde_json::json!({"hdr":{"file":"sky.hdr"}});
        assert!(
            check(&value)
                .unwrap_err()
                .to_string()
                .contains("ordinary program inputs")
        );
        value["nodes"][1].as_object_mut().unwrap().remove("image");
        value["nodes"][1]["material"] = serde_json::json!("missing");
        assert!(
            check(&value)
                .unwrap_err()
                .to_string()
                .contains("unknown environment material")
        );
        value["nodes"][1]["material"] = serde_json::json!("daylight");
        value["nodes"][1]["shape"] = serde_json::json!("point");
        assert!(
            check(&value)
                .unwrap_err()
                .to_string()
                .contains("only to environment")
        );
    }
}

#[cfg(test)]
mod cage_tests {
    use super::*;
    #[test]
    fn polygon_counts_and_crease_endpoints_are_validated() {
        let mut value = serde_json::json!({
            "version":2,"nodes":[{"type":"camera","name":"camera","material":{"program":"std/camera","inputs":{"projection":"persp"}}}],
            "buffers":{"p":{"type":"vec3","data":[[0,0,0],[1,0,0],[1,1,0],[0,1,0]]},
                "i":{"type":"u32","data":[0,1,2,3]}},
            "geometry":{"cage":{"attributes":{"POSITION":"p"},"indices":"i","face_vertex_counts":[4],
                "crease_edge":[{"vertices":[0,1],"crease":0.5}]}}
        });
        let check = |v: serde_json::Value| crate::v2::from_test_value::<Scene>(v).unwrap().validate();
        check(value.clone()).unwrap();
        value["geometry"]["cage"]["face_vertex_counts"] = serde_json::json!([3]);
        assert!(check(value.clone()).is_err());
        value["geometry"]["cage"]["face_vertex_counts"] = serde_json::json!([4]);
        value["geometry"]["cage"]["crease_edge"][0]["vertices"] = serde_json::json!([0,4]);
        assert!(check(value.clone()).is_err());
        value["geometry"]["cage"]["crease_edge"][0]["vertices"] = serde_json::json!([0,1]);
        value["geometry"]["cage"]["crease_edge"][0]["crease"] = serde_json::json!(10.);
        check(value.clone()).unwrap();
        value["geometry"]["cage"]["crease_edge"][0]["crease"] = serde_json::json!(-0.1);
        assert!(check(value).is_err());
    }
}


#[cfg(test)]
mod simulation_tests {
    use super::*;
    #[test]
    fn simulation_rejects_invalid_clocks_passes_and_shared_writers() -> Result<()> {
        let base=serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "buffers":{"state":{"type":"u32","data":[0]}},
            "programs":{"step":{"source":"struct Inputs {}; void compute(ComputeContext c,Inputs i) {}","schedule":"simulation","dispatch":1,"bindings":{"state":"state"}}},
            "simulation":{"t0":0,"dt":0.01,"seed":7,"state":["state"],"init":["step"],"steps":["step"]}});
        let valid=|v:serde_json::Value|->Result<()> {crate::v2::from_test_value::<Scene>(v)?.validate()};
        valid(base.clone())?;
        for dt in [0.,-0.01,1e-100,1e100] {
            let mut v=base.clone();v["simulation"]["dt"]=serde_json::json!(dt);assert!(valid(v).is_err());
        }
        for key in ["state","init","steps"] {
            let mut v=base.clone();v["simulation"][key]=serde_json::json!(["missing"]);assert!(valid(v).is_err());
            let mut v=base.clone();v["simulation"][key]=serde_json::json!([]);assert!(valid(v).is_err());
        }
        let mut v=base.clone();v["programs"]["other"]=v["programs"]["step"].clone();v["programs"]["other"]["schedule"]="frame".into();assert!(valid(v).is_err());
        let mut v=base;v.as_object_mut().unwrap().remove("simulation");assert!(valid(v).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod dense_grid_tests {
    use super::*;
    #[test]
    fn dense_fields_stay_on_volume_geometry() -> Result<()> {
        let mut value = serde_json::json!({"version":2,"nodes":[{"type":"camera"}],
            "buffers":{"d":{"type":"vec3","dims":[1,1,3]}},
            "geometry":{"g":{"topology":"volume","attributes":{"density":"d"}}}});
        let scene: Scene = crate::v2::from_test_value(value.clone())?;
        scene.validate()?;
        value["geometry"]["g"]["topology"] = "triangles".into();
        value["geometry"]["g"]["attributes"] = serde_json::json!({"POSITION":"d"});
        let scene: Scene = crate::v2::from_test_value(value)?;
        assert!(scene.validate().unwrap_err().to_string().contains("grid buffers require volume topology"));
        Ok(())
    }
    #[test]
    fn dense_grid_shape_type_and_storage_contract() -> Result<()> {
        let base=serde_json::json!({"version":2,"nodes":[{"type":"camera"}],"buffers":{"d":{"type":"f32","dims":[2,3,4]}}});
        let check=|v|->Result<Scene>{let s:Scene=crate::v2::from_test_value(v)?;s.validate()?;Ok(s)};
        assert_eq!(check(base.clone())?.buffers["d"].len(),Some(24));
        for value in [serde_json::json!([0,3,4]),serde_json::json!([u32::MAX,2,4])] {let mut v=base.clone();v["buffers"]["d"]["dims"]=value;assert!(check(v).is_err());}
        for (key,value) in [("count",serde_json::json!(24)),("data",serde_json::json!([0]))] {let mut v=base.clone();v["buffers"]["d"][key]=value;assert!(check(v).is_err());}
        let mut file = base.clone();
        file["buffers"]["d"]["file"] = "field.bin".into();
        assert_eq!(check(file.clone())?.buffers["d"].len(), Some(24));
        file["buffers"]["d"]["offset"] = 16.into();
        file["buffers"]["d"]["count"] = 24.into();
        check(file.clone())?;
        for (key, value) in [("count", serde_json::json!(23)), ("data", serde_json::json!(vec![0; 24])), ("grid", serde_json::json!("density")), ("offset", serde_json::json!(3))] {
            let mut invalid = file.clone(); invalid["buffers"]["d"][key] = value;
            assert!(check(invalid).is_err());
        }
        for kind in ["f32","i32","u32","vec2","vec3","vec4","uvec4","mat4"] {
            let mut v=base.clone();v["buffers"]["d"]["type"]=kind.into();
            assert_eq!(check(v.clone())?.buffers["d"].len(),Some(24));
            v["geometry"]=serde_json::json!({"g":{"topology":"volume","attributes":{"density":"d"}}});
            assert_eq!(check(v).is_ok(),matches!(kind,"f32"|"vec3"));
        }
        Ok(())
    }
}
