//! InstalledScene: the immutable GPU scene built once per shot.
use super::*;

/// Attribute normalized resources to every authored slot, preserving sharing.
pub(crate) fn medium_use_sites(nodes: &[Node], media: &std::collections::BTreeSet<&str>) -> Vec<String> {
    fn visit(nodes: &[Node], parent: &str, media: &std::collections::BTreeSet<&str>, labels: &mut Vec<String>) {
        for (index, node) in nodes.iter().enumerate() {
            let path = format!("{parent}[{index}]");
            let label = format!("{path} ({:?})", node.name);
            let mut slot = |property: String, material: &str| {
                if media.contains(material) {
                    labels.push(format!("{label}.{property} (medium material {material:?})"));
                }
            };
            match &node.kind {
                NodeKind::Mesh { materials, .. } => for (index, material) in materials.iter().enumerate() {
                    slot(format!("materials[{index}]"), material);
                },
                NodeKind::Asset { materials, .. } => for (name, material) in materials {
                    slot(format!("materials[{name:?}]"), material);
                },
                NodeKind::Light { medium: Some(material), .. } => slot("medium (world medium)".into(), material),
                _ => {},
            }
            visit(&node.children, &format!("{path}.children"), media, labels);
        }
    }
    let mut labels = Vec::new();
    visit(nodes, "nodes", media, &mut labels);
    labels
}

/// Resolve delivered bindings before choosing providers. Reflection identifies
/// medium stages; native ranges and selected glTF scenes identify uses. This is
/// the single admission decision for effects with no delivery evaluator, not an
/// authored-scene rewrite. Approximate surface/environment providers remain live.
#[derive(Default, Clone, Copy)]
pub(crate) struct TransportFacts {pub media:bool,pub hair:bool}

pub(crate) fn admit_delivery_effects(r:&Renderer,scene:&Scene,base:&Path,allow_media:bool)->Result<(std::collections::BTreeMap<String,programs::Stages>, TransportFacts)> {
    let stages=programs::mesh_stages(r,scene,base)?;
    let mut hair_materials=std::collections::BTreeSet::new();
    let mut abi=programs::abi();abi.aov_count=scene.render.aovs.len();
    let compiler=vork_slang::Compiler::discover();
    let animated=materials::animated_inputs(scene)?;
    // Compile every authored surface program once, in parallel, before the
    // serial admission loop below consumes the results. Retain them directly:
    // cache persistence can be disabled or refused, so prewarming alone would
    // compile twice in that case.
    let mut sources=std::collections::BTreeMap::new();
    for (name,stage) in &stages {
        let material=&scene.materials[name];
        if !stage.surface || material.program=="std/openpbr" {continue;}
        if !sources.contains_key(&material.program) {sources.insert(material.program.clone(),programs::material_source(scene,base,&material.program)?);}
    }
    let unique:Vec<_>=sources.values().collect();
    let compiled=programs::parallel_map(&unique,|(source,origin)| programs::compile(r,&compiler,&abi,source,origin))?;
    let compiled:std::collections::BTreeMap<_,_>=sources.keys().zip(compiled).collect();
    for (name,stage) in &stages {
        let material=&scene.materials[name];
        if !stage.surface || material.program=="std/openpbr" {continue;}
        let program=&compiled[&material.program];
        if !program.capabilities.hair {continue;}
        let values=material.inputs.iter().map(|(k,v)|Ok((k.clone(),serde_json::to_value(v)?)))
            .collect::<Result<std::collections::BTreeMap<_,_>>>()?;
        let dynamic=animated.get(name).cloned().unwrap_or_default();
        let words=program.inputs.effect_words(&values,&dynamic)?;
        // Hair is a closure type; only its bound scattering effect needs a path owner.
        if program.effect_probe()?.capabilities(&words).physical_hair {hair_materials.insert(name.clone());}
    }
    let (facts, _) = bound_effects(scene, base, &stages, allow_media, &hair_materials)?;
    Ok((stages,facts))
}

/// Source admission and installation use the same stage and binding decision.
/// No surface callback needs compilation to refuse this unsupported combination.
pub(crate) fn admit_stage_effects(scene:&Scene,base:&Path,stages:&std::collections::BTreeMap<String,programs::Stages>)->Result<()> {
    if !scene.materials.iter().any(|(name,material)| stages.get(name).is_some_and(|stage|stage.surface) && material.program!="std/openpbr") { return Ok(()); }
    let (_, media) = bound_effects(scene, base, stages, true, &Default::default())?;
    if let Some(material) = media.first() {
        bail!("material {material}: fog with custom surface callbacks is not supported; remove its medium stage or use std/openpbr for surface materials");
    }
    Ok(())
}

fn bound_effects(scene:&Scene,base:&Path,stages:&std::collections::BTreeMap<String,programs::Stages>,allow_media:bool,hair_materials:&std::collections::BTreeSet<String>)->Result<(TransportFacts,std::collections::BTreeSet<String>)> {
    let mut refused=std::collections::BTreeSet::new();
    let mut environments=Vec::new();
    let mut documents=std::collections::HashMap::new();
    let mut facts=TransportFacts::default();
    let mut media=std::collections::BTreeSet::new();
    fn visit(nodes:&[Node],scene:&Scene,base:&Path,stages:&std::collections::BTreeMap<String,programs::Stages>,
        documents:&mut std::collections::HashMap<PathBuf,gltf::Document>,environments:&mut Vec<String>,refused:&mut std::collections::BTreeSet<String>,next:&mut usize,allow_media:bool,facts:&mut TransportFacts,hair_materials:&std::collections::BTreeSet<String>,media:&mut std::collections::BTreeSet<String>)->Result<()> {
        for node in nodes {
            let index=*next;*next+=1;
            let label=format!("node {index} ({:?})",node.name);
            let mut medium=|slot:String,name:&str| {
                if stages.get(name).is_some_and(|stage|stage.medium) { facts.media=true; media.insert(name.to_owned()); }
                facts.hair |= hair_materials.contains(name);
                if !allow_media && stages.get(name).is_some_and(|stage|stage.medium) {
                    refused.insert(format!("{label}.{slot}: participating media are unavailable in delivery (material {name:?}); bake the medium into an image plate and composite it, or render the medium externally"));
                }
            };
            match &node.kind {
                NodeKind::Mesh {geometry,..} => {
                    let geometry=&scene.geometry[geometry];
                    let bindings=MaterialBindings::compile(scene,node,None,stages)?;
                    let slots:std::collections::BTreeSet<_>=if geometry.material_ranges.is_empty() {[0].into_iter().collect()}
                        else {geometry.material_ranges.iter().filter(|range|range.count>0).map(|range|range.material_index as usize).collect()};
                    for slot in slots {if let Some(name)=bindings.select(Some(slot)) {medium(format!("materials[{slot}]"),name);}}
                    facts.media |= matches!(geometry.topology,vork_scene::v2::Topology::Volume);
                    if (!allow_media || scene.delivery) && matches!(geometry.topology,vork_scene::v2::Topology::Volume) {
                        refused.insert(format!("{label}: stage 1 fog does not support grid volume topology; use a homogeneous closed convex triangle carrier, or render the volume externally and composite an image plate"));
                    }
                }
                NodeKind::Asset {file,materials,..} if crate::splats::is_file(file) => {
                    for name in materials.values() {medium("materials".into(),name);}
                }
                NodeKind::Asset {file,materials,..} if materials.values().any(|name|stages[name].medium || hair_materials.contains(name)) => {
                    // Only medium remaps need primitive reachability here. An
                    // ordinary asset contributes no unavailable effect and keeps
                    // its single normal installation read.
                    let path=base.join(file);
                    if !documents.contains_key(&path) {documents.insert(path.clone(),crate::import::open(&path).with_context(||format!("read installed material bindings from {}",path.display()))?.document);}
                    let doc=&documents[&path];
                    let bindings=MaterialBindings::compile(scene,node,Some(doc),stages)?;
                    for slot in imported_material_slots(doc)? {
                        if let Some(name)=bindings.select(slot) {medium(format!("materials[{slot:?}]"),name);}
                    }
                }
                NodeKind::Light {shape:vork_scene::LightShape::Environment,medium,..} => {
                    facts.media |= medium.is_some();
                    media.extend(medium.iter().cloned());
                    environments.push(label.clone());
                    if let Some(name)=medium.as_ref().filter(|_|!allow_media) {
                        refused.insert(format!("{label}.medium: participating media are unavailable in delivery (world material {name:?}); bake the medium into an image plate and composite it, or render the medium externally"));
                    }
                }
                _=>{}
            }
            visit(&node.children,scene,base,stages,documents,environments,refused,next,allow_media,facts,hair_materials,media)?;
        }
        Ok(())
    }
    visit(&scene.nodes,scene,base,stages,&mut documents,&mut environments,&mut refused,&mut 0,allow_media,&mut facts,hair_materials,&mut media)?;
    if let Some(first)=environments.first() {
        for extra in &environments[1..] {
            refused.insert(format!("{extra}: additional environment lights are unavailable; combine its contribution with {first} in one environment program or precomposite the environment maps"));
        }
    }
    ensure!(refused.is_empty(),"unsupported effects:\n{}",refused.into_iter().collect::<Vec<_>>().join("\n"));
    Ok((facts,media))
}

/// Authored material names bound to a mesh's material slots (glTF material
/// indices or native geometry groups); unbound imported slots keep glTF materials.
struct MaterialBindings {
    slots: Vec<Option<String>>,
    default: Option<String>,
    /// Program instances of the bound non-OpenPBR materials.
    programs: std::collections::BTreeMap<String, std::sync::Arc<programs::Instance>>,
}
impl MaterialBindings {
    fn compile(input: &Scene, source: &Node, doc: Option<&gltf::Document>, stages: &std::collections::BTreeMap<String, programs::Stages>) -> Result<Self> {
        let splats=matches!(&source.kind,NodeKind::Asset {file,..} if crate::splats::is_file(file));
        let mut slots = vec![None; doc.map_or(usize::from(splats), |doc| doc.materials().len())];
        let mut default = None;
        match &source.kind {
            NodeKind::Mesh { materials, .. } => {
                slots = materials.iter().cloned().map(Some).collect();
                if slots.is_empty() { slots.push(None); }
            }
            NodeKind::Asset { materials, .. } => {
                default = materials.get("*").cloned();
                slots.fill(default.clone());
                let mut specific = vec![false; slots.len()];
                for (key, material) in materials.iter().filter(|(key, _)| key.as_str() != "*") {
                    let targets: Vec<_> = if let Some(index) = key
                        .strip_prefix('#')
                        .and_then(|index| index.parse::<usize>().ok())
                    {
                        ensure!(index < slots.len(), "unknown imported material index {key}");
                        vec![index]
                    } else {
                        let targets: Vec<_> = doc
                            .context("asset material bindings require a glTF document")?
                            .materials()
                            .filter(|candidate| candidate.name() == Some(key.as_str()))
                            .map(|material| material.index().unwrap())
                            .collect();
                        ensure!(!targets.is_empty(), "unknown imported material {key:?}");
                        targets
                    };
                    for slot in targets {
                        ensure!(
                            !specific[slot],
                            "multiple remaps target imported material #{slot}"
                        );
                        specific[slot] = true;
                        slots[slot] = Some(material.clone());
                    }
                }
            }
            _ => {}
        }
        let mut programs = std::collections::BTreeMap::new();
        for name in slots.iter().flatten().chain(default.iter()) {
            if stages.get(name).is_none_or(|s| s.surface)
                && !programs.contains_key(name)
                && let Some(program) = programs::instance(input, name)?
            {
                programs.insert(name.clone(), std::sync::Arc::new(program));
            }
        }
        Ok(Self {
            slots,
            default,
            programs,
        })
    }
    fn select(&self, slot: Option<usize>) -> Option<&str> {
        match slot.and_then(|slot| self.slots.get(slot)) {
            Some(name) => name.as_deref(),
            None => self.default.as_deref(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct ImplicitField {
    pub source: String,
    pub statuses: Vec<(String, wgpu::Buffer, u64)>,
    pub dims: [u32;3],
    pub acceleration: std::sync::Arc<crate::implicit_acceleration::Acceleration>,
    pub spans: Vec<(wgpu::Buffer, buffers::AttributeSpan)>,
}
#[derive(Clone)]
pub(crate) struct Draw {
    pub(crate) splats: Option<std::sync::Arc<crate::splats::Geometry>>,
    pub(crate) implicit: Option<ImplicitField>,
    pub(crate) sort_order: usize,
    pub(crate) vertices: wgpu::Buffer,
    pub(crate) attributes: wgpu::Buffer,
    pub(crate) layout: crate::geometry::VertexLayout,
    /// Bit n: the geometry has TEXCOORD_n (material taps may read only those).
    pub(crate) uv_sets: u8,
    pub(crate) exit_geometry: Option<std::sync::Arc<crate::geometry::Geometry>>,
    pub(crate) exit_bvh: Option<wgpu::Buffer>,
    /// Draws of meshes that own a path solid: this draw's triangles' edges a
    /// closed mesh would not have (passes::path::open_edges); None where the
    /// geometry is not retained.
    pub(crate) open_edges: Option<std::sync::Arc<crate::passes::path::OpenEdges>>,
    pub(crate) indices: wgpu::Buffer,
    /// Immutable topology for validated BLAS refits; shared across frame slots.
    pub(crate) refit_indices: Option<wgpu::Buffer>,
    /// Shared static page; evaluated-layout and compute geometry own their buffers.
    pub(crate) geometry_page: Option<usize>,
    pub(crate) vertex_offset: u64,
    pub(crate) index_offset: u64,
    pub(crate) count: u32,
    /// Vertices addressed by this draw's indices (ray BLAS input range).
    pub(crate) vertex_count: u32,
    /// Alpha-tested draws of ray scenes: base-colour UV and vertex-colour alpha per
    /// index, for ray candidates. Released once install has built the alpha table.
    pub(crate) ray_corners: Option<std::sync::Arc<Vec<[f32; 5]>>>,
    pub(crate) material: usize,
    pub(crate) material_slot: Option<usize>,
    pub(crate) object: u32,
    pub(crate) instances: std::ops::Range<u32>,
    pub(crate) sort_stride: usize,
    pub(crate) local: Mat4,
    pub(crate) gltf_node: (usize, String),
    pub(crate) gltf_path: Vec<(usize, String)>,
    pub(crate) primitive_start: u32,
    pub(crate) center: Vec3,
    pub(crate) bounds: (Vec3, Vec3),
    pub(crate) generated_bounds: (Vec3, Vec3),
    pub(crate) computed: bool,
    pub(crate) deform: Option<Deform>,
    pub(crate) displacement: Option<std::sync::Arc<displacement::Mesh>>,
    /// Matching complete evaluation inputs may share a frame-private range.
    pub(crate) displacement_leader: Option<usize>,
}
impl Draw {
    pub(crate) fn rasterized(&self) -> bool {self.count>0 || self.computed}
    pub(crate) fn evaluated(&self) -> bool {self.computed || self.deform.is_some() || self.displacement.is_some()}
    pub(crate) fn evaluation_leader(&self) -> Option<usize> {
        self.displacement_leader.or_else(|| self.deform.as_ref().map(|d| d.leader))
    }
    pub(crate) fn attribute_offset(&self) -> u64 {
        self.vertex_offset / self.layout.vertex_stride() * self.layout.attribute_stride()
    }
}
#[derive(Clone)]
pub(crate) struct EvalNode {
    pub(crate) camera: Option<crate::camera_program::State>,
    pub(crate) node: Node,
    pub(crate) parent: Option<usize>,
    pub(crate) world: Mat4,
    pub(crate) dynamic: bool,
    pub(crate) embedded: Option<(usize, usize, usize, Option<f32>)>,
}
#[derive(Clone)]
pub(crate) struct MeshRoot {
    pub(crate) node: usize,
    pub(crate) local: Mat4,
    pub(crate) time_offset: f32,
    pub(crate) instance_index: Option<u32>,
    pub(crate) imported_visibility: u32,
    pub(crate) imported_pose: Option<(usize, usize)>,
}
impl MeshRoot {
    /// Placed by the same node, local transform and time offset as `other`.
    pub(crate) fn same_placement(&self, other: &Self) -> bool {
        self.node == other.node && self.local == other.local && self.time_offset == other.time_offset
    }
    pub(crate) fn world(&self, worlds: &[Mat4]) -> Mat4 {
        worlds[self.node] * self.local
    }
}

/// Structural choices fixed for one install, independent of other installs.
pub struct InstallOptions {
    /// Host loops must refuse promotion before allocating path views.
    pub(crate) raster_only: bool,
    /// Internal lowering declaration; scene authoring is exposed separately.
    pub(crate) geometry_stages: Option<Vec<crate::geometry_chain::Kind>>,
    /// Initial requested scene time in seconds, used for pose and camera initialization.
    pub initial_time: vork_time::Request,
    /// Per-install render settings; None uses the scene's render settings.
    pub render: Option<vork_scene::Render>,
    /// CPU planes frames may read back. The graph prunes readback, and its
    /// pixel buffer, when nothing is read; frames may request only these planes.
    pub readback: crate::ReadbackOutputs,
    /// Internal delivery intent; library installs retain GPU colour by default.
    pub(crate) gpu_color: bool,
    /// The display the view's result is encoded for: the render call's `output.display`. A world inside a
    /// canvas keeps the default, sRGB: it renders for the canvas, whose encoding that is.
    pub(crate) display: crate::output_device::Target,
    /// The view also keeps its result floating point (dithered, unquantized and unclamped) in the frame's
    /// staging planes: a world inside a canvas hands the canvas that picture.
    pub(crate) view_picture: bool,
    /// Initial frame storage; collection can request growth through `FrameRedo`.
    pub reservations: crate::Reservations,
    /// The pictures of the scene's canvas nodes, by image name, split from the scene before it installs.
    pub(crate) canvases: std::collections::BTreeMap<String, crate::canvas_world::Picture>,
    pub(crate) variants: pipelines::VariantCompile,
    /// Internal transport fixtures only; delivery drivers retain release admission.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_participating_media: bool,
    /// Scattering continuation budget for analytic path-medium fixtures.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_volume_bounces: u32,
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_convex_media: bool,
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_force_general_media: bool,
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_convex_work_limit: Option<u32>,
    /// Qualification-only cap; absent from scene content and release policy.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_medium_batch_rows: Option<u32>,
}
impl Default for InstallOptions {
    fn default() -> Self {
        Self {
            raster_only: false,
            geometry_stages: None,
            initial_time: vork_time::Request::Numeric(0.),
            render: None,
            readback: crate::ReadbackOutputs {
                float: crate::FloatOutputs::ALL,
                ..crate::ReadbackOutputs::ALL
            },
            gpu_color: true,
            display: Default::default(),
            view_picture: false,
            reservations: Default::default(),
            canvases: Default::default(),
            variants: Default::default(),
            #[cfg(any(test, feature = "test-support"))]
            test_participating_media: cfg!(test),
            #[cfg(any(test, feature = "test-support"))]
            test_volume_bounces: 0,
            #[cfg(any(test, feature = "test-support"))]
            test_convex_media: false,
            #[cfg(any(test, feature = "test-support"))]
            test_force_general_media: false,
            #[cfg(any(test, feature = "test-support"))]
            test_convex_work_limit: None,
            #[cfg(any(test, feature = "test-support"))]
            test_medium_batch_rows: None,
        }
    }
}

pub struct InstalledScene {
    pub(crate) geometry_stages: Vec<crate::geometry_chain::Kind>,
    pub(crate) has_catchers: bool,
    pub(crate) footage: std::collections::BTreeMap<String, crate::footage::Installed>,
    pub(crate) file_images: std::collections::BTreeMap<String, wgpu::TextureView>,
    pub(crate) reservations: crate::Reservations,
    pub(crate) sampling: crate::sampling::Parameters,
    /// Shared exposure error word, protected by the normal submission lock.
    pub(crate) displacement: Option<displacement::Pipeline>,
    pub(crate) shadows: crate::residency::Cache<crate::shadows::Persistent>,
    /// Colour history shared by every frame slot for `integrator.aa(): temporal`.
    pub(crate) compute: Option<compute::Compute>,
    pub(crate) dense: Option<crate::dense::Dense>,
    pub(crate) computed_geometry: Vec<Option<crate::computed_geometry::Mesh>>,
    pub(crate) compositor: std::collections::HashMap<String, compositor::Program>,
    pub(crate) image_order: Vec<String>,
    pub(crate) main_outputs: compositor::Dependencies,
    pub(crate) main_subset: Option<crate::render_images::Subset>,
    pub(crate) capture_controls: bool,
    /// CPU planes this install may read back.
    pub(crate) readback: crate::ReadbackOutputs,
    pub(crate) animation: std::sync::Arc<animation::Plan>,
    pub(crate) pipelines: std::sync::Arc<CachedPipelines>,
    pub(crate) plan: crate::plan::ScenePlan,
    pub(crate) graph: crate::graph::Graph,
    pub(crate) transients: crate::graph::Transients,
    pub(crate) path: Option<crate::passes::path::Pipeline>,
    pub(crate) reflections: Option<pipelines::Drawn<wgpu::ComputePipeline>>,
    /// Decoupled shading kernels and the camera's draw table.
    pub(crate) decoupled: Option<crate::passes::decoupled::Install>,
    pub(crate) software_geometry: Option<crate::passes::decoupled::RasterGeometry>,
    pub(crate) gi: Option<pipelines::Drawn<wgpu::ComputePipeline>>,
    /// The irradiance volume when the scene enables probes.
    pub(crate) probes: Option<crate::probes::Volume>,
    pub(crate) fibre: Option<crate::fibre_transport::Persistent>,
    pub(crate) geometry_pipelines:
        std::collections::HashMap<pipelines::GeometryKey, pipelines::Drawn>,
    pub(crate) vertex_defaults: wgpu::Buffer,
    pub(crate) environment: environment::Environment,
    pub(crate) spectral: bool,
    pub(crate) propagation_base: u32,
    pub(crate) camera_programs: Vec<crate::camera_program::Instance>,
    pub(crate) surface_dependencies: Vec<vork_slang::ProgramDependencies>,
    pub(crate) links: lights::links::Links,
    pub(crate) medium_base: u32,
    /// Environment program input words: the first program words after the records.
    pub(crate) environment_base: u32,
    pub(crate) has_media: bool,
    pub(crate) medium_certified_times: Option<[f32;2]>,
    pub(crate) emitter_profiles: std::collections::HashMap<usize,u32>,
    pub(crate) assets: Vec<Asset>,
    pub(crate) scene: Scene,
    pub(crate) render: vork_scene::Render,
    pub(crate) nodes: Vec<EvalNode>,
    pub(crate) targets: crate::targets::Plan,
    pub(crate) dynamic_objects: Vec<usize>,
    pub(crate) roots: Vec<MeshRoot>,
    pub(crate) object_ids: std::sync::Arc<serde_json::Value>,
    pub(crate) camera: usize,
    pub(crate) light_plan: LightPlan,
    pub(crate) draws: Vec<Draw>,
    pub(crate) implicit_fields: Vec<ImplicitField>,
    /// Static acceleration structures when the scene chooses a ray technique.
    pub(crate) rays: Option<crate::rt::Rays>,
    pub(crate) draw_groups: Vec<std::ops::Range<usize>>,
    pub(crate) materials: Vec<Material>,
    /// The scene material table: records, texture views and samplers.
    pub(crate) material_table: materials::MaterialTable,
    pub(crate) checked_shader_capacity: bool,
    pub(crate) objects: Vec<Object>,
    pub(crate) display_program: crate::display_program::Stage,
    /// Unique raster vertex and attribute buffer bytes; excludes indices and deformation workspace.
    pub geometry_vertex_bytes: u64,
    pub install_time: Duration,
    pub pipeline_ms: f64,
    pub environment_install_ms: f64,
    pub material_bind_ms: f64,
    pub asset_decode_ms: f64,
    pub texture_upload_ms: f64,
    pub texture_cache_hits: usize,
}
// A screen capture has no interior scene-boundary provider. Refuse possible
// contents rather than render a picture that drops them. Bounds are conservative:
// hardware transport or path is the complete provider when these overlap.
/// Material slots reachable from the same glTF scene used by GeometryUpload.
/// Other document scenes and unreferenced meshes do not create scene programs.
fn imported_material_slots(doc:&gltf::Document)->Result<std::collections::BTreeSet<Option<usize>>> {
    fn visit(node:gltf::Node,slots:&mut std::collections::BTreeSet<Option<usize>>) {
        if let Some(mesh)=node.mesh() {for p in mesh.primitives() {slots.insert(p.material().index());}}
        for child in node.children() {visit(child,slots);}
    }
    let scene=doc.default_scene().or_else(||doc.scenes().next()).context("glTF has no scene")?;
    let mut slots=std::collections::BTreeSet::new();
    for node in scene.nodes() {visit(node,&mut slots);}
    Ok(slots)
}

fn refuse_screen_contents(draws:&[Draw],roots:&[MeshRoot],included:&[bool],objects:&[Object],materials:&[Material],records:&[materials::MaterialRecord])->Result<()> {
    let in_view=|object:u32|included[roots[object as usize].node];
    let visible=draws.iter().filter(|draw|draw.instances.clone().any(in_view));
    ensure!(!visible.clone().any(|draw|materials[draw.material].thin_transmission),
        "thin-sheet refraction needs scene boundary transport; use ray refraction on a ray-query capable adapter, or std/path (screen refraction cannot represent rough transmission and rear interfaces)");
    if !visible.clone().any(|draw|materials[draw.material].solid_transmission) {return Ok(());}
    ensure!(!visible.clone().any(|draw|materials[draw.material].solid_transmission
        && records[draw.material].screen_exit_needs_color(draw.layout.color)),
        "solid glTF surface colour from COLOR_0 or a base-colour texture needs exit material evaluation; use ray refraction or std/path (screen refraction is unsupported); see vork help upgrade");
    let identities=crate::passes::path::mesh_identities(draws,roots)?;
    let placed:Vec<_>=visible.flat_map(|draw|draw.instances.clone().filter(|&object|in_view(object)).map(move |object|(draw,object))).map(|(draw,object)| {
        let model=objects[object as usize].model;
        (draw,object,crate::shadow_envelope::transformed(model,draw.bounds))
    }).collect();
    for &(glass,object,(lo,hi)) in &placed {
        if !materials[glass.material].solid_transmission {continue;}
        for &(_,other,(other_lo,other_hi)) in &placed {
            if objects[other as usize].metadata[1]&constants::RAY_TRANSMISSION==0 {continue;}
            if identities[object as usize]==identities[other as usize] {continue;}
            if lo.cmplt(other_hi).all() && other_lo.cmplt(hi).all() {
                anyhow::bail!("solid glass contents need scene boundary transport; use a ray-query capable adapter with static placements, or std/path (screen refraction cannot represent overlapping contents)");
            }
        }
    }
    Ok(())
}

/// A light whose emission reaches receivers through shadow rays. Environment lighting never does; an extracted sun is its own node.
fn shadowed_light(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Light { shape, .. } if !matches!(shape, vork_scene::LightShape::Environment)) && kind.casts_shadows()
}

impl InstalledScene {
    /// The frame's passes and the scene's programs, to name a GPU failure. A
    /// lost device reports only the failed submission, which holds the whole frame.
    pub(crate) fn gpu_work(&self) -> String {
        let passes: Vec<_> = self.graph.passes().iter().map(|p| format!("{p:?}")).collect();
        let mut work = format!("passes {}", passes.join(", "));
        if !self.scene.programs.is_empty() {
            work += &format!("; scene programs {}", self.scene.programs.keys().cloned().collect::<Vec<_>>().join(", "));
        }
        work
    }
    pub(crate) fn retain_shadows(&self, r: &Renderer) -> Result<std::sync::Arc<crate::shadows::Persistent>> {
        self.shadows.get_or_try_init(|| {
            let shadows = crate::shadows::Persistent::new(self, r)?;
            let bytes = shadows.bytes();
            Ok((shadows, bytes))
        })
    }
    pub(crate) fn shadows(&self) -> std::sync::Arc<crate::shadows::Persistent> {
        self.shadows.get().expect("installed shadows")
    }
    pub(crate) fn new(
        r: &Renderer,
        scene: Scene,
        base: &Path,
        effects: EffectPlan,
        background_pipelines: bool,
        automatic_probes: bool,
        options: &InstallOptions,
        opened: &crate::footage::Opened,
    ) -> Result<Self> {
        std::thread::scope(|scope| Self::new_scoped(r, scene, base, effects, background_pipelines, automatic_probes, options, opened, scope))
    }

    fn new_scoped<'scope>(
        r: &'scope Renderer,
        scene: Scene,
        base: &Path,
        mut effects: EffectPlan,
        _background_pipelines: bool,
        automatic_probes: bool,
        options: &InstallOptions,
        opened: &crate::footage::Opened,
        scope: &'scope std::thread::Scope<'scope, '_>,
    ) -> Result<Self> {
        let InstallOptions { readback, ref variants, reservations, .. } = *options;
        let geometry_stages = options.geometry_stages.clone().unwrap_or_else(||scene.geometry_stages.iter().copied().map(crate::geometry_chain::Kind::from_scene).collect());
        for (index,kind) in geometry_stages.iter().enumerate() {
            ensure!(!geometry_stages[..index].contains(kind),"geometry stages[{index}] repeats {kind:?}; remove the duplicate built-in stage");
        }
        let compositor = compositor::install(r, &scene, base, opened)?;
        let (mut output_dependencies, _) = compositor::dependencies(&compositor, scene.compositor());
        if !options.gpu_color && !readback.color && !readback.color16 && !readback.float.color {output_dependencies=Default::default();}
        output_dependencies.depth |= readback.depth_ids || readback.float.depth;
        output_dependencies.ids |= readback.depth_ids;
        output_dependencies.normals |= readback.normals || readback.float.normals;
        output_dependencies.motion |= readback.motion || readback.float.motion;
        output_dependencies.aov_output = readback.aovs && !scene.render.aovs.is_empty();
        output_dependencies.aovs |= output_dependencies.aov_output;
        output_dependencies.catcher |= readback.aovs && crate::catcher::has_catchers(&scene);
        // Captures retain their own scene producers. File images need no scene passes.
        output_dependencies.images &= scene.images.values().any(|i|i.render.is_some());
        let (allocation_dependencies, planned_image_order)=compositor::image_plan(&scene,&compositor,output_dependencies)?;
        crate::render_images::require_supported_roots(&scene, &planned_image_order)?;
        effects.outputs = Some(allocation_dependencies);
        effects.denoise &= allocation_dependencies.beauty;
        if !allocation_dependencies.beauty && !allocation_dependencies.catcher {
            effects.indirect=false; effects.ray_indirect=false; effects.reflections=false;
            effects.probes=false; effects.fibre=false; effects.subsurface=false; effects.denoise=false;
        }
        let input = &scene;
        if !input.render.aovs.is_empty() {
            if output_dependencies.beauty || output_dependencies.aovs || output_dependencies.catcher {
                crate::aov::check_storage(&r.device.limits(), &input.render)?;
            }
            for name in &planned_image_order {
                let render=input.images[name].resolved_render(&input.render).expect("a rendered image");
                let (d,_)=compositor::dependencies(&compositor,&render.compositor);
                if d.beauty || d.aovs || d.catcher {
                    crate::aov::check_storage(&r.device.limits(), &render)?;
                }
            }
        }
        let display_program = crate::display_program::prepare(r, &scene, base, options.display, opened, options.view_picture)?.install(r)?;
        let mut render = scene.main_render();
        let environment_start = Instant::now();
        let scene_inputs=allocation_dependencies.beauty || allocation_dependencies.controls()
            || allocation_dependencies.aovs || allocation_dependencies.catcher;
        if !scene_inputs {effects.implicit=false;}
        let mut environment = if scene_inputs {environment::Environment::new_in(
            r, environment_node::Settings::from_scene(&scene)?, &scene, base, opened,
        )?} else {environment::Environment::unused(r)};
        let environment_install_ms = environment_start.elapsed().as_secs_f64() * 1000.;
        // Preorder: an authored node's index is its position in the scene tree walk.
        let mut nodes = Vec::new();
        fn flatten(input: &[Node], parent: Option<usize>, nodes: &mut Vec<EvalNode>) {
            for n in input {
                let id = nodes.len();
                let dynamic = parent.is_some_and(|p| nodes[p].dynamic);
                let world =
                    parent.map_or(Mat4::IDENTITY, |p| nodes[p].world) * n.transform.matrix();
                let mut node = n.without_children();
                if let NodeKind::Light {
                    shape, energy: intensity, ..
                } = &mut node.kind
                {
                    intensity.get_or_insert(shape.default_energy());
                }
                nodes.push(EvalNode { camera: None,
                    node,
                    parent,
                    world,
                    dynamic,
                    embedded: None,
                });
                flatten(&n.children, Some(id), nodes);
            }
        }
        flatten(&scene.nodes, None, &mut nodes);
        let propagation=crate::propagation::prepare(r,input,base)?;
        let buffer_data = buffers::SceneBuffers::decode(input, base)?;
        let mut animation = animation::Plan::compile_in(input, base, Some(&buffer_data.data), opened)?;
        let camera_views=crate::camera_program::views(input,&nodes,&animation,&options.initial_time);
        let mut camera_programs=crate::camera_program::install(r,input,&mut nodes,base,&camera_views)?;
        let spectral=crate::camera_program::scene_spectral(r,input,base)?;
        let camera_native=(spectral || !camera_programs.is_empty()).then(||std::sync::Arc::new(crate::camera_program::native(&camera_programs)));
        environment.node = environment.settings.source;
        let mut links = lights::links::Links::new(&nodes)?;
        if render.settings.integrator.probe_volume().is_some() {
            ensure!(environment.node.is_none_or(|world| links.illuminates(world).all),
                "retained indirect lighting cannot represent environment illuminates receiver links: the aggregate field mixes World and local sources; omit illuminates or choose --profile std/path");
        }
        for node in &mut nodes {
            if let NodeKind::Mesh {
                geometry,
                morph_weights,
                ..
            } = &mut node.node.kind
            {
                for name in input.geometry[geometry.as_str()].morphs.keys() {
                    morph_weights.entry(name.clone()).or_insert(0.);
                }
            }
        }
        let mut node_names: std::collections::HashMap<String, usize> = nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| !node.node.name.is_empty())
            .map(|(index, node)| (node.node.name.clone(), index))
            .collect();
        let curve_sides = crate::curves::projected_sides(input, &nodes, &buffer_data, &animation, &camera_views)?;
        buffers::admit_points(input, &buffer_data, &curve_sides)?;
        let mut embedded = Vec::new();
        let mut imported_camera_materials=vork_scene::v2::Table::new();
        let pipeline_start = Instant::now();
        let mut program_material_names = std::collections::BTreeSet::new();
        for node in nodes.iter().filter(|_|scene_inputs) {
            match &node.node.kind {
                NodeKind::Mesh { materials, .. } => program_material_names.extend(materials.iter()),
                NodeKind::Asset { materials, .. } => {
                    program_material_names.extend(materials.values())
                }
                _ => {}
            }
        }
        let stages = programs::mesh_stages(r, input, base)?;
        let mut decoded = std::collections::HashMap::new();
        let mut image_decoders = materials::ImageDecoders::default();
        // Imported glTF keeps its factors and texture taps in the shared material table.
        let image_order = planned_image_order.clone();
        let image_slots = image_order.iter().enumerate().map(|(slot,name)|(name.clone(),slot)).collect();
        let mut textures = TextureInstaller::new(r, &image_slots);
        textures.opened = opened.clone();
        textures.canvases = options.canvases.clone();
        textures.fps = render.fps;
        let mut imported_materials=std::collections::BTreeMap::new();
        for node in &nodes {
            let NodeKind::Asset {file,..}=&node.node.kind else {continue;};
            if crate::splats::is_file(file) {continue;}
            let asset_key=format!("asset:{file}");
            if !decoded.contains_key(&asset_key) {
                let ((doc,buffers,images),keys)=materials::load_cached(&base.join(file),&r.disk_cache,&mut image_decoders,r.device.limits().max_texture_dimension_2d)?;
                decoded.insert(asset_key.clone(),std::rc::Rc::new(((Some(doc),buffers,images),None,keys)));
            }
            let ((doc,_,images),_,keys)=&**decoded.get(&asset_key).unwrap();
            let doc=doc.as_deref().unwrap();
            if !scene_inputs {continue;}
            let bindings=MaterialBindings::compile(input,&node.node,Some(doc),&stages)?;
            let reachable=imported_material_slots(doc)?;
            for source in std::iter::once(None).chain(doc.materials().map(Some)) {
                let slot=source.as_ref().and_then(|m|m.index());
                let key=(file.clone(),slot);
                if !reachable.contains(&slot) || bindings.select(slot).is_some() || imported_materials.contains_key(&key) {continue;}
                let material=materials::install_gltf(&mut textures,doc,source.as_ref(),images,keys)?;
                imported_materials.insert(key,material);
            }
        }
        let native_program_materials = program_material_names
            .iter()
            .filter(|name| stages[name.as_str()].surface)
            .map(|name| Ok(programs::instance(input, name)?.map(|p| (name.as_str(), p))))
            .filter_map(Result::transpose)
            .collect::<Result<Vec<_>>>()?;
        let attribute_names = if scene_inputs {crate::attributes::names(input)?} else {Vec::new()};
        let attributes=attribute_names.len() as u32;
        let required_locations=vork_slang::registry::VISIBILITY_LENS_LOCATION+1+attributes.saturating_sub(8);
        let limits=r.device.limits();
        let raster_attributes=nodes.iter().any(|node|matches!(&node.node.kind,NodeKind::Mesh {..})
            || matches!(&node.node.kind,NodeKind::Asset {file,..} if !crate::splats::is_file(file)));
        // The most named attributes both checks below admit on this device; up
        // to eight take no inter-stage check. Refusals state it (vork help limits).
        let fit=(limits.max_vertex_attributes+8).saturating_sub(vork_slang::registry::VISIBILITY_LENS_LOCATION+1)
            .min(limits.max_inter_stage_shader_variables.saturating_sub(14).max(8));
        ensure!(!raster_attributes || required_locations <= limits.max_vertex_attributes,
            crate::gpu_memory::device_limit("material attribute vertex layout", format_args!("{required_locations} locations ({attributes} named attributes)"), format_args!("max_vertex_attributes={}; this device fits {fit} named attributes on meshes",limits.max_vertex_attributes), "reduce named material attributes"));

        if attributes>8 && raster_attributes {
            let varying_locations=14+attributes;
            ensure!(varying_locations<=limits.max_inter_stage_shader_variables,
                crate::gpu_memory::device_limit("material attribute pipeline", format_args!("{varying_locations} inter-stage locations ({attributes} named attributes)"), format_args!("max_inter_stage_shader_variables={}; this device fits {fit} named attributes on meshes",limits.max_inter_stage_shader_variables), "reduce named material attributes"));
        }
        let programs = programs::install(r, input, native_program_materials.iter().map(|(name, p)| (*name, p)), base)?;
        let mut physical_hair_materials = std::collections::HashSet::new();
        for name in &program_material_names {
            if let Some(instance) = programs::instance(input, name)? {
                if programs.as_ref().is_some_and(|p| p.is_physical_hair(&instance)) {
                    physical_hair_materials.insert(name.as_str());
                }
            }
        }
        let computed_attribute_sources: std::collections::HashSet<_> = input.programs.values()
            .filter(|program| program.is_compute()).flat_map(|program| program.bindings.values()).collect();
        let animated_inputs = materials::animated_inputs(input)?;
        // One installed capability summary drives both graph reservation and
        // material validity. Program outputs need no sentinel input names.
        let mut transmission_materials=std::collections::BTreeSet::new();
        let mut spatial_subsurface = false;
        let splat_materials:std::collections::HashSet<_>=nodes.iter().filter_map(|node|match &node.node.kind {
            NodeKind::Asset {file,materials,..} if crate::splats::is_file(file)=>Some(materials.values().map(String::as_str)),
            _=>None,
        }).flatten().collect();
        for name in &program_material_names {
            if !stages[*name].surface { continue; }
            if let Some(instance) = programs::instance(input, name)? {
                let capabilities=programs.as_ref().context("material program was not compiled")?.capabilities(&instance);
                if splat_materials.contains(name.as_str()) {crate::splats::material_capabilities(&capabilities)?;}
                spatial_subsurface|=capabilities.spatial_subsurface;
                if capabilities.transmission {transmission_materials.insert(*name);}
            } else {
                let record=materials::openpbr_values(input, Some(name), animated_inputs.get(*name).unwrap_or(&Default::default()), |_| Ok(0))?;
                if splat_materials.contains(name.as_str()) {crate::splats::material_capabilities(&record.capabilities)?;}
                spatial_subsurface|=record.spatial_subsurface();
                if record.transmissive {transmission_materials.insert(*name);}
            }
        }

        // Shader-to-RGB results are final (surface_lighting::unreplaced):
        // their diffuse and reflection payloads are zero. Keep the original
        // surface's probe and deferred connection lighting for the lit input,
        // but install no screen replacement consumers when every surface is lit.
        let only_lit=nodes.iter().all(|node|match &node.node.kind {
            NodeKind::Mesh {materials,..}=>!materials.is_empty() && materials.iter().all(|name| {
                !stages[name].surface || programs::instance(input,name).ok().flatten().is_some_and(|instance|
                    programs.as_ref().is_some_and(|p|p.capabilities(&instance).lit))
            }),
            NodeKind::Asset {..}=>false,
            _=>true,
        });
        // Refraction's first reflected event also consumes projection.z; keep
        // that transport mode intact whenever a material may transmit.
        if only_lit && transmission_materials.is_empty() && !effects.path {
            effects.indirect=false;
            effects.ray_indirect=false;
            effects.reflections=false;
            render.settings.integrator.raster_mut().unwrap().reflections=vork_scene::Reflections::Environment;
        }

        // Solid scattering needs mesh chords. Any transmitting caster needs RGB
        // straight connections for every receiver; scalar depth maps cannot
        // represent its closure throughput. Both use the shared ray providers.
        // Without a shadowed light no receiver issues a light connection, so
        // the caster needs neither provider. Assets may import lights.
        let material_spatial_subsurface=spatial_subsurface;
        let spatial_subsurface=material_spatial_subsurface && !effects.path;
        let transmitting_casters=!effects.path
            && nodes.iter().any(|node| shadowed_light(&node.node.kind) || matches!(node.node.kind, NodeKind::Asset{..}))
            && nodes.iter().enumerate().any(|(id,node)| {
            if buffers::node_visibility_mask(&nodes,id)&constants::RAY_SHADOW==0
                && !crate::plan::shadow_visibility_animated(&nodes,id,&animation) {return false;}
            match &node.node.kind {
                NodeKind::Mesh{materials,..}=>materials.iter().any(|name|transmission_materials.contains(name)),
                NodeKind::Asset{..}=>true,
                _=>false,
            }
        });
        let program_buffers = input
            .programs
            .values()
            .any(|p| p.is_compute())
            .then_some(()).or_else(|| (input.buffers.values().any(|b| b.dims.is_some()) || input.geometry.values().any(|g|crate::computed_geometry::dynamic(input,g))).then_some(()))
            .filter(|_| scene_inputs)
            .map(|_| buffers::ProgramBuffers::install(r, input, &buffer_data))
            .transpose()?;
        let packed_predecessors = nodes.iter().filter_map(|node| match &node.node.kind {
            NodeKind::Mesh {geometry,materials,..} if materials.iter().any(|name|stages.get(name).is_some_and(|stage|stage.displacement))=>Some(geometry.clone()),
            _=>None,
        }).collect();
        let prepared_compute=if scene_inputs {compute::prepare_with_packed(r,input,base,program_buffers.as_ref(),&nodes.iter().map(|node|Some(&node.node)).collect::<Vec<_>>(),&packed_predecessors)?} else {Default::default()};
        let compute_trace=prepared_compute.values().any(|p|!p.artifact.queries.is_empty());
        let mut query_identity=compute_trace.then(blake3::Hasher::new);
        if let Some(hash)=&mut query_identity {
            hash.update(&serde_json::to_vec(input)?);
            for (name,data) in &buffer_data.data {hash.update(name.as_bytes());hash.update(&(data.words.len() as u64).to_le_bytes());hash.update(bytemuck::cast_slice(&data.words));}
        }

        let program_trace=programs.as_ref().is_some_and(|p|p.dependencies().trace);
        effects.trace=program_trace;
        ensure!(!program_trace || !input.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit)), "trace does not yet support implicit geometry; use evaluated triangle geometry");
        ensure!(!program_trace || !crate::splats::present(input), "trace does not support Gaussian splats; remove trace from the program or the splat asset from the scene");
        let mut promoted_transport=spatial_subsurface || transmitting_casters;
        if promoted_transport {
            let raster=render.settings.integrator.raster_mut().unwrap();
            if spatial_subsurface {raster.subsurface_method=Some(vork_scene::v2::SubsurfaceMethod::Thickness);}
            // Transmitting casters require RGB ray connections.
            if transmitting_casters {raster.shadow_method=vork_scene::ShadowMethod::Ray;}
        }
        effects.subsurface = effects.raster_diffusion(spatial_subsurface);
        effects.fibre = !effects.path && programs.as_ref().is_some_and(|p|p.has_physical_hair());

        let mut medium = if scene_inputs {medium::install(r, input, environment.settings.medium.as_deref(), base, &stages)?} else {None};

        #[cfg(any(test, feature = "test-support"))]
        let qualified_media = input.delivery || options.test_convex_media;
        #[cfg(not(any(test, feature = "test-support")))]
        let qualified_media = true;
        if qualified_media && medium.is_some() {
            admit_stage_effects(input,base,&stages)?;
            ensure!(!stages.values().any(|stage| stage.medium && stage.surface), "stage 1 fog does not support combined surface/medium programs; overlap an independent fog carrier with std/openpbr glass");
            if let Some(medium) = medium.as_mut() {
                ensure!(medium.native.homogeneous || environment.settings.medium.as_deref().is_some_and(|world|medium.certified_world_exponential(world)), "stage 1 fog does not support varying finite media or arbitrary world density; use homogeneous coefficient programs for carriers and homogeneous or certified exponential-height world fog");
                // Select a boundary representation after geometry is available; neither
                // representation imposes shape/count admission.
                medium.native.convex = false;
                #[cfg(any(test, feature = "test-support"))]
                if options.test_convex_media {medium.native.convex=true;}
                medium.native.work_limit = Some(131072);
                #[cfg(any(test, feature = "test-support"))]
                if options.test_convex_media || options.test_convex_work_limit.is_some() {medium.native.work_limit = options.test_convex_work_limit;}
            }
        }

        ensure!(medium.is_none() || effects.path,"internal error: participating media require complete-camera path transport");
        if effects.path && let Some(error)=medium.as_ref().and_then(|m|m.native.sampled_error.as_ref()) {return Err(crate::failure::scene(anyhow::Error::new(crate::medium::SampledRefusal(error.clone()))));}
        let raster_samples = raster_samples(&scene.render);
        let mut light_shadow_rays=crate::rt::installed_shadow_rays(&scene,&nodes);
        let mut rays = crate::rt::uses_rays(&scene) || light_shadow_rays || promoted_transport || program_trace || compute_trace;
        // Decoded mesh emitters can request volume rays after geometry upload.
        // Reserve the native input usage while scattering is possible; the
        // decoded source decision still owns actual ray/topology installation.
        let geometry_rays=rays || (medium.as_ref().is_some_and(|m|!m.native.scattering_free)
            && r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY));
        // Device preflight treats every unknown program/asset as transmissive.
        // Installation already knows authored surface capabilities; a medium,
        // environment or opaque program must not specialize surface connections.
        let connection_transmission = rays && (!transmission_materials.is_empty()
            || input.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit))
            || nodes.iter().any(|node|matches!(node.node.kind,NodeKind::Asset{..})));
        let mut ray_refraction = crate::rt::ray_refraction(&scene);
        let mut ray_shadows = light_shadow_rays || ray_refraction || effects.probes || promoted_transport || program_trace || compute_trace;
        let field_controls = !effects.path && input.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit));
        let mut slang = crate::slang_forward::SlangScene {
                program_trace,
                bounded_secondary_compilation: render.settings.bounded_secondary_compilation,
                approximate_field_boundaries: render.settings.approximate_field_boundaries && field_controls,
                propagation: propagation.clone(),
                camera: camera_native,
                environment: environment.primary.clone(),
                programs: programs.as_ref().and_then(|p| p.native.clone()),
                aov_count: input.render.aovs.len(),
                attributes: attribute_names.len() as u32,
                textures: 1, samplers: 1,
                ray_refraction,
                connection_transmission, surface_coverage:false, attribute_coverage:false, shadow_capacity:0,
                no_shadow_containment: false, no_boundary_transactions: false, no_points: false, gaussians: false, unit_ior: false, unit_boundary_ior: false, no_gltf_tint: false, no_film: false,
                probes: effects.probes,
                fibre: effects.fibre,
                denoise: effects.denoise,
                field_controls,
                software_raster: r.adapter_info.device_type == wgpu::DeviceType::Cpu,
                medium: medium.as_ref().map(|m|std::sync::Arc::new(m.native.clone())),
            };
        // Program input words follow the material records in one buffer.
        let mut material_words = environment.primary.as_ref().map_or_else(Vec::new,|p|p.words.clone());
        let propagation_offset=material_words.len();
        if let Some(p)=&propagation {material_words.extend_from_slice(&p.words);}
        for camera in &mut camera_programs {camera.offset=material_words.len();material_words.extend_from_slice(&camera.words);}
        let mut program_word_lengths = std::collections::HashMap::<u32, usize>::new();
        let mut hair_geometry_words = std::collections::HashMap::<String, u32>::new();
        let mut word_offsets = std::collections::HashMap::<(u32, Vec<u32>), u32>::new();
        let mut pipeline_ms = pipeline_start.elapsed().as_secs_f64() * 1000.;
        let mut asset_decode_ms = 0.;
        let image_order = planned_image_order.clone();
        let mut views: Vec<_> = std::iter::once(render.clone()).chain(image_order.iter()
            .map(|name| input.images[name].resolved_render(&render).expect("a rendered image"))).map(|mut view| {
                if let Some(raster)=view.settings.integrator.raster_mut() {
                    if spatial_subsurface {raster.subsurface_method=Some(vork_scene::v2::SubsurfaceMethod::Thickness);}
                    if transmitting_casters {raster.shadow_method=vork_scene::ShadowMethod::Ray;}
                }
                view
            }).collect();
        let mut assets = Vec::new();
        let mut draws: Vec<Draw> = Vec::new();
        let mut materials = Vec::new();
        let mut objects = Vec::new();
        let mut roots = Vec::new();
        let mut source_cache = animation::SourceCache::default();
        let mut animation_sources = std::collections::HashMap::<String, Asset>::new();
        let mut material_ranges = std::collections::HashMap::<String, (Vec<usize>, usize)>::new();
        let mut material_ids = std::collections::HashMap::new();
        // Record index of each authored material name, shared by all its uses.
        let mut native_ids = std::collections::BTreeMap::<String, usize>::new();
        // Install static geometry in shared, bounded vertex/index pages. Draw order and
        // local indices remain unchanged; each draw keeps its byte offsets.
        let mut geometry_pages = Vec::new();
        let placeholder = init(
            &r.device,
            "geometry placeholder",
            &[0; 96],
            wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE,
        );
        let mut splat_clouds=std::collections::HashMap::<String,std::sync::Arc<vork_splat::Cloud>>::new();
        let mut geometry_ranges = std::collections::HashMap::<
            (String, bool, Option<usize>),
            std::ops::Range<usize>,
        >::new();
        for (node_id, node) in nodes.iter().enumerate() {
            if !scene_inputs && matches!(node.node.kind,NodeKind::Mesh {..}) {continue;}
            let (native_mesh, asset_source) = match &node.node.kind {
                NodeKind::Mesh {
                    geometry,
                    materials,
                    ..
                } => (Some((geometry, materials)), None),
                NodeKind::Asset {
                    file,
                    materials,
                    nodes: overrides,
                } => (None, Some((file, materials, overrides))),
                _ => continue,
            };
            let object = objects.len() as u32;
            objects.push(Object::new(node.world));
            let decode_start = Instant::now();
            let (key, material_key) = match (native_mesh, asset_source) {
                (Some((geometry, materials)), _) => (
                    format!("mesh:{geometry}"),
                    format!("mesh:{geometry}:{}", serde_json::to_string(materials)?),
                ),
                (_, Some((path, materials, _))) => (
                    format!("asset:{path}"),
                    format!("asset:{path}:{}", serde_json::to_string(materials)?),
                ),
                _ => unreachable!(),
            };
            let is_splat=asset_source.is_some_and(|(path,..)|crate::splats::is_file(path));
            let stream_layout = native_mesh.is_some_and(|(name,_)| crate::computed_geometry::dynamic(input, &input.geometry[name.as_str()]));
            let evaluated_layout=stream_layout || match (native_mesh,asset_source) {
                (Some((_,materials)),_)=>materials.iter().any(|name|stages.get(name).is_some_and(|s|s.displacement)),
                (_,Some((_,materials,_)))=>materials.values().any(|name|stages.get(name).is_some_and(|s|s.displacement)),
                _=>false,
            };
            ensure!(!is_splat || !evaluated_layout,"Gaussian splats do not support displacement programs; transform their asset node or replace the surface material");
            let key=if evaluated_layout {format!("{key}:evaluated")} else {key};
            let loaded = if let Some(loaded) = decoded.get(&key) {
                std::rc::Rc::clone(loaded)
            } else {
                let (loaded, geometry, image_keys) = if is_splat {
                    let (path,..)=asset_source.unwrap();
                    let cloud=crate::splats::read(&base.join(path),crate::residency::available(&r.device).unwrap_or(u64::MAX))?;
                    splat_clouds.insert(key.clone(),std::sync::Arc::new(cloud));
                    ((None,std::sync::Arc::new(vec![]),vec![]),None,Vec::new())
                } else if let Some((path, ..)) = asset_source {
                    let path = base.join(path);
                    let (loaded, keys) = materials::load_cached(
                        &path,
                        &r.disk_cache,
                        &mut image_decoders,
                        r.device.limits().max_texture_dimension_2d,
                    )
                        .with_context(|| format!("load {}", path.display()))?;
                    (
                        {
                            let (doc, buffers, images) = loaded;
                            (Some(doc), buffers, images)
                        },
                        None,
                        keys,
                    )
                } else {
                    let (name, _) = native_mesh.unwrap();
                    let geometry = buffers::geometry_v2(
                        &buffer_data,
                        &input.geometry[name.as_str()],
                        &attribute_names,
                        curve_sides.get(name.as_str()).copied().unwrap_or(8),
                        Some(&r.disk_cache),
                    )?;
                    let geometry = crate::subdivision::prepare(r, &buffer_data, &input.geometry[name.as_str()], geometry)?;
                    ((None, std::sync::Arc::new(vec![]), vec![]), Some(geometry), Vec::new())
                };
                if let Some(hash)=&mut query_identity {
                    if let Some(doc)=&loaded.0 {hash.update(&serde_json::to_vec(doc.as_json())?);}
                    for buffer in loaded.1.iter() {hash.update(&(buffer.0.len() as u64).to_le_bytes());hash.update(&buffer.0);}
                }
                let loaded = std::rc::Rc::new((loaded, geometry, image_keys));
                decoded.insert(key.clone(), std::rc::Rc::clone(&loaded));
                loaded
            };
            let ((doc, buffers, _), geometry, _) = &*loaded;
            if native_mesh.is_some_and(|(_, names)| names.iter().any(|name| physical_hair_materials.contains(name.as_str()))) {
                if let Some(geometry) = geometry.as_ref().filter(|g| g.vertices.first().is_some_and(|v| v.TANGENT[3].abs() >= 2.)) {
                    let offset = if let Some(&offset) = hair_geometry_words.get(&key) { offset } else {
                        let offset = u32::try_from(material_words.len())?;
                        material_words.extend(crate::hair::geometry_words(geometry)?);
                        hair_geometry_words.insert(key.clone(), offset);
                        offset
                    };
                    objects[object as usize].material_index[1] = offset;
                    objects[object as usize].material_index[2] = 1;
                }
            }
            if let Some((_, _, overrides)) = asset_source.filter(|_|!is_splat) {
                let doc = doc.as_ref().context("asset requires a glTF document")?;
                let mut targets = std::collections::HashSet::new();
                for key in overrides.keys() {
                    let indices: Vec<_> = if let Some(index) =
                        key.strip_prefix('#').and_then(|s| s.parse::<usize>().ok())
                    {
                        doc.nodes()
                            .filter(|node| node.index() == index)
                            .map(|node| node.index())
                            .collect()
                    } else {
                        doc.nodes()
                            .filter(|node| node.name() == Some(key.as_str()))
                            .map(|node| node.index())
                            .collect()
                    };
                    ensure!(
                        indices.len() == 1,
                        "asset override {key:?} must name exactly one imported node or #index"
                    );
                    ensure!(
                        targets.insert(indices[0]),
                        "multiple overrides target imported node #{}",
                        indices[0]
                    );
                }
            }
            asset_decode_ms += decode_start.elapsed().as_secs_f64() * 1000.;
            let asset_id = (asset_source.is_some() && !is_splat).then_some(assets.len());
            if let (Some(asset), Some((_, _, overrides))) = (asset_id, asset_source) {
                lights::import(
                    doc.as_ref().context("asset requires a glTF document")?,
                    node_id,
                    asset,
                    node.world,
                    &node.node.name,
                    overrides,
                    &mut embedded, &mut imported_camera_materials,
                )?;
            }
            let native_deformation = match &node.node.kind {
                NodeKind::Mesh { geometry, skin, .. } => {
                    skin.is_some() || !input.geometry[geometry.as_str()].morphs.is_empty()
                }
                _ => false,
            };
            let native_binding = if native_deformation {
                Some(buffers::native_binding(
                    input,
                    &nodes,
                    &node_names,
                    node_id,
                    &buffer_data,
                )?)
            } else {
                None
            };
            if let Some((path, ..)) = asset_source.filter(|_|!is_splat) {
                let doc = doc.as_ref().context("asset requires a glTF document")?;
                // Clip layers bound to this asset, by plan layer index.
                let clips: Vec<_> = animation.clips_for_asset(&node.node.name).collect();
                let animation_key = format!("{path}:{}", serde_json::to_string(&clips)?);
                let template = if let Some(template) = animation_sources.get(&animation_key) {
                    template.clone()
                } else {
                    let template =
                        Asset::cached(doc, buffers, &clips, base, path, &mut source_cache)?;
                    animation_sources.insert(animation_key, template.clone());
                    template
                };
                assets.push(template.register_native(
                    &mut animation,
                    doc,
                    &node.node.name,
                    asset_id.expect("glTF asset registered"),
                    asset_source.unwrap().2,
                )?);
            }
            // Imported cameras and their animation remain live without mesh
            // consumers; material installation and geometry uploads do not.
            if !scene_inputs {continue;}
            let asset_animated = if native_mesh.is_some() || is_splat {
                native_deformation
            } else {
                assets[asset_id.expect("glTF asset registered")].animated()
            };
            let (material_map, default_material) = if let Some(range) =
                material_ranges.get(&material_key)
            {
                range.clone()
            } else {
                let bindings = MaterialBindings::compile(input, &node.node, doc.as_deref(), &stages)?;
                if let Some(doc) = &doc {
                    for primitive in doc.meshes().flat_map(|m| m.primitives()) {
                        if let Some(material) = bindings
                            .select(primitive.material().index())
                            .and_then(|name| input.materials.get(name))
                        {
                            for (input_name,value) in &material.inputs {
                                if let vork_scene::v2::Input::Attribute(binding) = value {
                                    let accessor=primitive.attributes().find(|(semantic,_)|semantic.to_string()==binding.attribute).map(|(_,a)|a)
                                        .with_context(||format!("asset material input {input_name}: missing attribute {:?}",binding.attribute))?;
                                    let width=if material.program=="std/openpbr" {
                                        match vork_scene::OpenPbrParam::find(input_name).context("OpenPBR attribute input")?.kind {
                                            vork_scene::OpenPbrKind::Float=>1,vork_scene::OpenPbrKind::Color3=>3,_=>anyhow::bail!("attribute requires a numeric destination")
                                        }
                                    } else {
                                        let instance=programs::instance(input,bindings.select(primitive.material().index()).unwrap())?.context("attribute program instance")?;
                                        programs.as_ref().and_then(|p|p.attribute_width(&instance,input_name)).context("attribute bindings require Attribute<T> Inputs")?
                                    };
                                    attributes::validate(binding,width,accessor.dimensions().multiplicity() as u32)?;
                                }
                            }
                        }
                    }
                }
                if is_splat {
                    if let Some(name)=bindings.select(Some(0)) {
                        let material=&input.materials[name];
                        for (input_name,value) in &material.inputs {
                            let vork_scene::v2::Input::Attribute(binding)=value else {continue;};
                            let source=crate::splats::attribute(&splat_clouds[&key].metadata,&binding.attribute)?>>24;
                            let width=if material.program=="std/openpbr" {
                                match vork_scene::OpenPbrParam::find(input_name).context("OpenPBR attribute input")?.kind {
                                    vork_scene::OpenPbrKind::Float=>1,vork_scene::OpenPbrKind::Color3=>3,_=>anyhow::bail!("attribute requires a numeric destination"),
                                }
                            } else {
                                let instance=programs::instance(input,name)?.context("attribute program instance")?;
                                programs.as_ref().and_then(|p|p.attribute_width(&instance,input_name)).context("attribute bindings require Attribute<T> Inputs")?
                            };
                            attributes::validate(binding,width,source)?;
                        }
                    }
                }
                // Tracks write one material's record, so animated records are never shared.
                let mut retain = |material: materials::PendingMaterial, shared: bool| {
                    if !shared {
                        materials.push(material);
                        return materials.len() - 1;
                    }
                    *material_ids.entry(material.key()).or_insert_with(|| {
                        let index = materials.len();
                        materials.push(material);
                        index
                    })
                };
                let mut make_material = |source: Option<&gltf::Material<'_>>,
                                         slot: Option<usize>|
                 -> Result<usize> {
                    let Some(name) = bindings.select(slot) else {
                        let material = if doc.is_some() {
                            let key=(asset_source.unwrap().0.clone(),source.and_then(|m|m.index()));
                            imported_materials[&key].clone()
                        } else {
                            materials::install_openpbr(&mut textures, input, None, base, &Default::default())?
                        };
                        return Ok(retain(material, true));
                    };
                    if let Some(&index) = native_ids.get(name) {
                        return Ok(index);
                    }
                    let index = if !stages[name].surface {
                        retain(materials::PendingMaterial::medium_boundary(), false)
                    } else if let Some(program) = bindings.programs.get(name) {
                        let programs = programs
                            .as_ref()
                            .context("material program was not compiled")?;
                        // Program images pack their table words into the input words.
                        let mut image_words = std::collections::BTreeMap::new();
                        for image in &program.images {
                            let (view, sampler) = textures.program_image(image, base)?;
                            image_words.insert(image.input.clone(), textures.slot(view, sampler));
                        }
                        let (id, words) = programs.pack(program, &image_words, input, base)
                            .with_context(|| format!("materials.{name}.inputs: program {}", program.name))?;
                        let authored = &input.materials[name];
                        // Equal initial cells do not make distinct mutable grids aliases.
                        let unique_inputs=animated_inputs.contains_key(name) || authored.inputs.values().any(|value|
                            matches!(value,vork_scene::v2::Input::Grid(binding) if computed_attribute_sources.contains(&binding.grid)));
                        let mut material = materials::PendingMaterial::program(
                            id,
                            authored.double_sided,
                        );
                        material.record.extra[0] = f32::from_bits(authored.priority as u32);
                        material.hair=programs.is_hair(program);
                        material.physical_hair=programs.is_physical_hair(program);
                        let capabilities = programs.capabilities(program);
                        material.response_program = if capabilities.anisotropy {crate::forward_variant::FEATURE_ANISOTROPY} else {0}
                            | if capabilities.coat_anisotropy {crate::forward_variant::FEATURE_COAT_ANISOTROPY} else {0}
                            | if capabilities.film {crate::forward_variant::FEATURE_FILM} else {0}
                            | if capabilities.coat {materials::FEATURE_COAT} else {0}
                            | if capabilities.fuzz {materials::FEATURE_FUZZ} else {0}
                            | if capabilities.tint {materials::FEATURE_TINT} else {0}
                            // The surface stage's lobe is proven. A reflecting lit stage's
                            // closure has no such proof, so it keeps the lobe possible.
                            | if capabilities.subsurface || capabilities.lit_reflects {crate::forward_variant::FEATURE_SUBSURFACE} else {0}
                            | if capabilities.weighted_ior {materials::FEATURE_WEIGHTED_IOR} else {0};
                        if !programs.is_hair(program) && (capabilities.opacity || capabilities.transmission) {
                            material.integrate_opacity(authored.opacity, capabilities.transmission, programs.coverage_varies(program));
                        }
                        material.lobes = capabilities.lobes;
                        material.lit = capabilities.lit;
                        material.source_emission = capabilities.emission;
                        material.record.tag[3] = if unique_inputs {
                            let offset = material_words.len() as u32;
                            material_words.extend_from_slice(&words);
                            offset
                        } else {
                            *word_offsets.entry((id, words.clone())).or_insert_with(|| {
                                let offset = material_words.len() as u32;
                                material_words.extend_from_slice(&words);
                                offset
                            })
                        };
                        if animated_inputs.contains_key(name) {
                            program_word_lengths.insert(material.record.tag[3], words.len());
                        }
                        retain(material, !unique_inputs && !stages[name].medium)
                    } else {
                        let animated = animated_inputs.get(name);
                        let material = materials::install_openpbr(
                            &mut textures,
                            input,
                            Some(name),
                            base,
                            animated.unwrap_or(&Default::default()),
                        )?;
                        retain(material, animated.is_none() && !stages[name].medium)
                    };
                    native_ids.insert(name.to_owned(), index);
                    Ok(index)
                };
                let mut map = Vec::new();
                if is_splat {
                    map.push(make_material(None,Some(0))?);
                } else if let Some((_, slots)) = native_mesh {
                    for slot in 0..slots.len().max(1) {
                        map.push(make_material(None, Some(slot))?);
                    }
                } else {
                    let document=doc.as_ref().context("asset requires a glTF document")?;
                    let reachable=imported_material_slots(document)?;
                    for material in document.materials() {
                        // Keep glTF's document indices for the geometry uploader;
                        // unreachable slots are never read and need no installation.
                        map.push(if reachable.contains(&material.index()) {make_material(Some(&material),material.index())?} else {0});
                    }
                }
                let default = if native_mesh.is_some() || is_splat {
                    map.first().copied().context("native mesh requires material slots")?
                } else if imported_material_slots(doc.as_ref().unwrap())?.contains(&None) {
                    make_material(None,None)?
                } else {0};
                material_ranges.insert(material_key, (map.clone(), default));
                (map, default)
            };
            // Bind geometry is immutable even for animated instances. Only
            // the asset binding, palettes and frame outputs are instance-private.
            let animated = asset_animated;
            let geometry_key = (
                key.clone(),
                animated,
                native_binding.as_ref().map(|binding| binding.joints.len()),
            );
            if let Some(range) = geometry_ranges.get(&geometry_key) {
                for i in range.clone() {
                    let source = &draws[i];
                    draws.push(Draw {
                        splats: source.splats.clone(),
                        implicit: source.implicit.clone(),
                        sort_order: 0,
                        vertices: source.vertices.clone(),
                        attributes: source.attributes.clone(),
                        layout: source.layout,
                        uv_sets: source.uv_sets,
                        exit_geometry: source.exit_geometry.clone(),
                        exit_bvh: source.exit_bvh.clone(),
                        open_edges: source.open_edges.clone(),
                        indices: source.indices.clone(),
                        refit_indices: source.refit_indices.clone(),
                        geometry_page: source.geometry_page,
                        vertex_offset: source.vertex_offset,
                        index_offset: source.index_offset,
                        count: source.count,
                        vertex_count: source.vertex_count,
                        ray_corners: source.ray_corners.clone(),
                        material: source
                            .material_slot
                            .map_or(default_material, |slot| material_map[slot]),
                        material_slot: source.material_slot,
                        object,
                        instances: object..object + 1,
                        sort_stride: 1,
                        local: source.local,
                        gltf_node: source.gltf_node.clone(),
                        gltf_path: source.gltf_path.clone(),
                        primitive_start: source.primitive_start,
                        center: source.center,
                        bounds: source.bounds,
                        generated_bounds: source.generated_bounds,
                        computed: source.computed,
                        displacement: None,
                        displacement_leader: None,
                        deform: source.deform.as_ref().map(|d| {
                            let mut d = d.clone();
                            d.asset = asset_id;
                            if let Some(binding) = &native_binding {
                                d.native = Some(std::sync::Arc::clone(binding));
                            }
                            d
                        }),
                    });
                }
                roots.push(MeshRoot {
                    node: node_id,
                    local: Mat4::IDENTITY,
                    time_offset: 0.,
                    instance_index: None,
                    imported_visibility: constants::RAY_VISIBILITY_ALL,
                    imported_pose: None,
                });
                continue;
            }
            let draw_start = draws.len();
            let mut min = Vec3::splat(f32::INFINITY);
            let mut max = Vec3::splat(f32::NEG_INFINITY);
            let gltf_scene = doc
                .as_ref()
                .map(|doc| {
                    doc.default_scene()
                        .or_else(|| doc.scenes().next())
                        .context("glTF has no scene")
                })
                .transpose()?;
            let mut import = import::GeometryUpload {
                attribute_names: &attribute_names,
                evaluated_layout,
                stream_layout,
                asset: asset_id,
                animated: asset_animated,
                r,
                rays: geometry_rays,
                buffers: &buffers,
                draws: &mut draws,
                object,
                material_map: &material_map,
                materials: &materials,
                default: default_material,
                pages: &mut geometry_pages,
                placeholder: &placeholder,
                min: &mut min,
                max: &mut max,
                meshes: std::collections::HashMap::new(),
                path: Vec::new(),
            };
            if is_splat {
                crate::splats::upload(&mut import,&splat_clouds[&key])?;
            } else if let Some(geometry) = geometry {
                let (name, slots) = native_mesh.expect("native geometry has a mesh node");
                let source = &input.geometry[name.as_str()];
                let deform = if let Some(binding) = &native_binding {
                    let skin = match &node.node.kind {
                        NodeKind::Mesh { skin, .. } => skin.as_ref(),
                        _ => None,
                    };
                    buffers::deformation(
                        r,
                        &buffer_data,
                        source,
                        skin,
                        geometry,
                        std::sync::Arc::clone(binding),
                    )?
                } else {
                    None
                };
                let groups = buffers::groups(source, geometry.indices.len(), slots.len().max(1))?;
                import.upload(
                    geometry,
                    deform,
                    default_material,
                    (0, String::new()),
                    Mat4::IDENTITY,
                )?;
                if import.draws.len() > draw_start {
                    let upload = import.draws.pop().unwrap();
                    for (range, slot) in groups {
                        if range.is_empty() {
                            continue;
                        }
                        let mut draw = upload.clone();
                        draw.index_offset += (range.start * std::mem::size_of::<u32>()) as u64;
                        draw.count = range.len() as u32;
                        draw.primitive_start = (range.start / 3) as u32;
                        draw.bounds = geometry.range_bounds(range);
                        draw.center = (draw.bounds.0 + draw.bounds.1) * 0.5;
                        draw.material = material_map[slot];
                        draw.material_slot = Some(slot);
                        import.draws.push(draw);
                    }
                }
            } else {
                for node in gltf_scene.iter().flat_map(|scene| scene.nodes()) {
                    import.visit(node, Mat4::IDENTITY)?;
                }
            }
            // Light- or camera-only glTF assets are valid and draw nothing.
            ensure!(
                draws.len() == draw_start || (min.is_finite() && max.is_finite()),
                "glTF geometry has no finite bounds"
            );
            geometry_ranges.insert(geometry_key, draw_start..draws.len());
            roots.push(MeshRoot {
                node: node_id,
                local: Mat4::IDENTITY,
                time_offset: 0.,
                instance_index: None,
                imported_visibility: constants::RAY_VISIBILITY_ALL,
                imported_pose: None,
            });
        }
        drop(decoded);
        drop(splat_clouds);
        buffers::instances(&nodes, &mut roots, &mut objects, &mut draws, &buffer_data)?;
        let mut refused_geometry_effects = std::collections::BTreeSet::new();
        for draw in &mut draws {
            let node = &nodes[roots[draw.object as usize].node].node;
            if let NodeKind::Mesh { geometry, .. } = &node.kind {
                draw.computed = crate::computed_geometry::dynamic(input, &input.geometry[geometry]);
                if draw.computed && (materials[draw.material].record.extra[1]>0. || materials[draw.material].transmissive) {
                    refused_geometry_effects.insert(format!("node {:?}: computed geometry with opacity or transmission would be omitted; use opaque geometry or bake the geometry updates", node.name));
                }
            }
        }
        // Segmentation IDs are authored metadata, independent of transform slots.
        let mut segmentation = buffers::SegmentationIds::new(input, &buffer_data)?;
        let source_roots = std::mem::take(&mut roots);
        let source_objects = std::mem::take(&mut objects);
        for draw in &mut draws {
            let node = source_roots[draw.object as usize].node;
            let imported_visibility = match &nodes[node].node.kind {
                NodeKind::Asset {
                    nodes: overrides, ..
                } => buffers::imported_visibility_mask(overrides, &draw.gltf_path),
                _ => constants::RAY_VISIBILITY_ALL,
            };
            let object_groups = links.object_groups(node, &nodes[node].node, &draw.gltf_path)?;
            let draw_id = segmentation.prepare(node, &draw.gltf_path, &nodes, &buffer_data)?;
            let first = objects.len() as u32;
            for original in draw.instances.clone() {
                let mut root = source_roots[original as usize].clone();
                let id = draw_id.instance(root.instance_index)?;
                root.imported_visibility = imported_visibility;
                root.imported_pose = draw.deform.as_ref().and_then(|d| d.asset.map(|asset| (asset,d.node)));
                root.local *= draw.local;
                let world = nodes[node].world * root.local;
                let mut value = source_objects[original as usize];
                value.model = world;
                value.previous_model = world;
                value.next_model = world;
                value.opacity[1] = id as f32;
                value.metadata[3] = root.instance_index.unwrap_or(0);
                value.metadata[0] = object_groups;
                value.metadata[1] =
                    (if buffers::node_collapsed(&nodes, node) {0} else {buffers::node_visibility_mask(&nodes, node)}) & root.imported_visibility | buffers::node_matte_flags(&nodes,node);
                objects.push(value);
                roots.push(root);
            }
            draw.object = first;
            draw.instances = first..objects.len() as u32;
        }

        let object_ids = segmentation.metadata();
        for (page, data) in geometry_pages.into_iter().enumerate() {
            if data.vertices.is_empty() {
                continue;
            }
            let attributes = init(
                &r.device,
                "scene static attributes",
                &data.attributes,
                // Ray hits read attribute streams in place (rt::hit_record).
                wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            );
            let vertices = init(
                &r.device,
                "scene static vertices",
                &data.vertices,
                wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::STORAGE
                    | crate::rt::geometry_usage(geometry_rays),
            );
            let indices = init(
                &r.device,
                "scene static indices",
                bytemuck::cast_slice(&data.indices),
                wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::STORAGE
                    | crate::rt::geometry_usage(geometry_rays),
            );
            for draw in draws
                .iter_mut()
                .filter(|d| d.geometry_page == Some(page))
            {
                draw.vertices = vertices.clone();
                draw.attributes = attributes.clone();
                draw.indices = indices.clone();
            }
        }
        for (index, node) in embedded.iter().enumerate() {
            ensure!(
                node_names
                    .insert(node.node.name.clone(), nodes.len() + index)
                    .is_none(),
                "imported camera/light name {:?} is ambiguous",
                node.node.name
            );
        }
        // Imported asset cameras remain selectable, but do not make the authored
        // scene camera ambiguous. Authored nodes precede imports for default selection.
        let authored_camera_count = nodes
            .iter()
            .filter(|n| matches!(n.node.kind, NodeKind::Camera { .. }))
            .count();
        for node in &mut embedded {
            if let Some((asset, _, _, _)) = node.embedded { node.dynamic |= assets[asset].animated(); }
        }
        nodes.extend(embedded);
        if !imported_camera_materials.is_empty() {
            let imported: vork_scene::Scene=serde_json::from_value(serde_json::json!({"version":2,"nodes":[],"materials":imported_camera_materials,"render":{"resolution":render.resolution}}))?;
            let camera_views=crate::camera_program::views(input,&nodes,&animation,&options.initial_time);
            let mut added=crate::camera_program::install(r,&imported,&mut nodes,base,&camera_views)?;
            for camera in &mut added {camera.id+=camera_programs.len() as u32;camera.offset=material_words.len();material_words.extend_from_slice(&camera.words);}
            camera_programs.extend(added);
            slang.camera=Some(std::sync::Arc::new(crate::camera_program::native(&camera_programs)));
        }
        // Resolve only the omitted start camera, after buffers and clips are compiled.
        // Use the same ordered layer sampling as every rendered frame.
        if render.camera.is_none() {
            render.camera = animation.start_camera(&render, &options.initial_time);
        }
        let camera = nodes
            .iter()
            .position(|n| {
                matches!(n.node.kind, NodeKind::Camera { .. })
                    && render
                        .camera
                        .as_deref()
                        .is_none_or(|name| n.node.name == name)
            })
            .context("install requires a camera matching render.camera")?;
        ensure!(
            render.camera.is_some() || authored_camera_count <= 1,
            "a scene with {authored_camera_count} authored cameras requires render.camera"
        );
        // Instances with a time offset phase animated materials as they phase motion:
        // each (record, offset) pair gets its own record copy, written at time + offset.
        let material_inputs = animation.material_inputs();
        let animated: std::collections::BTreeSet<usize> = material_inputs
            .iter()
            .filter_map(|(material, _)| native_ids.get(material).copied())
            .collect();
        let mut phased = std::collections::BTreeMap::<(usize, u32), usize>::new();
        for draw in draws.iter().filter(|d| animated.contains(&d.material)) {
            for object in draw.instances.clone() {
                let offset = roots[object as usize].time_offset;
                if offset != 0. {
                    phased
                        .entry((draw.material, offset.to_bits()))
                        .or_insert_with(|| {
                            let mut material = materials[draw.material].clone();
                            if material.record.tag[0] != 0 {
                                // Resource descriptors follow the reflected prefix and use
                                // relative addresses: phase the complete packed block.
                                let width = program_word_lengths[&material.record.tag[3]];
                                let start = material.record.tag[3] as usize;
                                let words = material_words[start..start + width].to_vec();
                                material.record.tag[3] = material_words.len() as u32;
                                program_word_lengths.insert(material.record.tag[3], width);
                                material_words.extend(words);
                            }
                            materials.push(material);
                            materials.len() - 1
                        });
                }
            }
        }
        // Material tracks write the record of their material, wherever it is used.
        {
            let records: Vec<_> = materials.iter().map(|m| m.record).collect();
            let records: &[u8] = bytemuck::cast_slice(&records);
            for target in material_inputs {
                if input.materials[&target.0].program != "std/openpbr" { continue; }
                let mut slots = vec![];
                if let Some(&slot) = native_ids.get(&target.0) {
                    slots.push((slot, 0.));
                    slots.extend(
                        phased
                            .iter()
                            .filter(|((s, _), _)| *s == slot)
                            .map(|(&(_, bits), &copy)| (copy, f32::from_bits(bits))),
                    );
                }
                let offsets: std::collections::HashMap<_, _> = slots.iter().copied().collect();
                let slots: Vec<usize> = slots.iter().map(|s| s.0).collect();
                let bindings = materials::track_targets((&target.0, &target.1), &slots, records)?
                    .into_iter()
                    .map(|(slot, parameter, value)| animation::MaterialBinding {
                        slot,
                        parameter,
                        storage: None, enum_tokens: None,
                        default: animation::Sample::Numeric(value),
                        offset: offsets[&slot],
                    })
                    .collect();
                animation.bind_material(&target, bindings)?;
            }
        }
        // Non-OpenPBR consumers resolve words after the table's final placement.
        for target in animation.material_inputs() {
            if input.materials[&target.0].program != "std/openpbr" {
                animation.bind_material(&target, vec![])?;
            }
        }
        animation.mark_nodes(&mut nodes, camera)?;
        // A program's `time` input is its node's local time (rule 7), which programs cannot read yet:
        // a program that reads `time` under a node clock is refused with one line.
        for (index, node) in nodes.iter().enumerate().filter(|(index, _)| animation.under_clock(*index)) {
            let refuse = |what: String| anyhow::anyhow!("node {:?}: {what} reads time, and a program's time under a node clock is not in this release; key its inputs instead", node.node.name);
            let materials: Vec<&String> = match &node.node.kind {
                NodeKind::Mesh { materials, .. } => materials.iter().collect(),
                NodeKind::Asset { materials, .. } => materials.values().collect(),
                _ => Vec::new(),
            };
            for material in materials {
                if let (Some(programs), Some(instance)) = (programs.as_ref(), programs::instance(input, material)?) {
                    if programs.reads_time(&instance) { return Err(refuse(format!("material {material:?}"))); }
                }
            }
            if let Some(program) = camera_programs.iter().find(|p| p.node == index && p.uses_time) {
                return Err(refuse(format!("camera material {:?}", program.material)));
            }
            if environment.node == Some(index) && environment.uses_time() { return Err(refuse("the environment program".into())); }
            if environment.node == Some(index) && medium.as_ref().is_some_and(|m| m.native.time_dependent) { return Err(refuse("the medium program".into())); }
        }
        let targets = crate::targets::Plan::compile(&mut nodes)?;
        animation.finish_asset_bindings()?;
        let mut samples = animation.samples();
        let mut values = crate::values::sample_storage(&nodes);
        let mut worlds = nodes.iter().map(|node| node.world).collect::<Vec<_>>();
        // Reference sampling coordinates use the canonical pose, including
        // camera targets, rather than the still call's seek time.
        let (probe_frame,probe_eyes) = if automatic_probes && render.settings.integrator.probe_volume().is_some() {
            let mut canonical_samples=animation.samples();
            let mut canonical_values=crate::values::sample_storage(&nodes);
            let mut canonical_worlds=worlds.clone();
            crate::values::sample_pose(&nodes,&assets,&animation,&targets,&environment,0.,
                &mut canonical_samples,&mut canonical_values,&mut canonical_worlds)?;
            let canonical_objects=objects.iter().zip(&roots).map(|(object,root)| {
                let mut object=*object;object.model=root.world(&canonical_worlds);object
            }).collect::<Vec<_>>();
            let eyes=nodes.iter().enumerate().filter(|(_,n)|matches!(n.node.kind,NodeKind::Camera {..}))
                .map(|(i,_)|canonical_worlds[i].w_axis.truncate()).collect::<Vec<_>>();
            (Some(crate::probes::placement::Frame::for_camera(&nodes[camera],canonical_worlds[camera],&draws,&canonical_objects,&render)),Some(eyes))
        } else {(None,None)};
        crate::values::sample_pose(&nodes, &assets, &animation, &targets, &environment, options.initial_time.clone(),
            &mut samples, &mut values, &mut worlds)?;
        for (node, world) in nodes.iter_mut().zip(&worlds) { node.world = *world; }
        // Object tables predate animation compilation; downstream bounds and probe
        // fitting must see the same evaluated placement as node/camera initialization.
        for (root, object) in roots.iter().zip(&mut objects) {
            let world = root.world(&worlds);
            object.model = world;
            object.previous_model = world;
            object.next_model = world;
        }
        // Asset contents were unknown during reservation. Opaque imports, and
        // imports without shadowed lights, retain the original shadow
        // representation after material decoding.
        let resolved_shadow_rays=crate::rt::installed_shadow_rays(&scene,&nodes);
        let shadow_changed=resolved_shadow_rays!=light_shadow_rays;
        light_shadow_rays=resolved_shadow_rays;
        let connection_changed=transmitting_casters && (!nodes.iter().any(|node| shadowed_light(&node.node.kind)) || !draws.iter().any(|draw| {
            materials[draw.material].transmissive && draw.instances.clone().any(|object| {
                let id=roots[object as usize].node;
                buffers::node_visibility_mask(&nodes,id)&constants::RAY_SHADOW!=0
                    || crate::plan::shadow_visibility_animated(&nodes,id,&animation)
            })
        }));
        if connection_changed {
            promoted_transport=spatial_subsurface;
            if let Some(medium)=&mut medium {
                slang.medium=Some(std::sync::Arc::new(medium.native.clone()));
            }
            for view in std::iter::once(&mut render).chain(views.iter_mut()) {
                if let Some(raster)=view.settings.integrator.raster_mut() {
                    raster.shadow_method=scene.render.settings.integrator.shadow_method();
                }
            }
        }
        let limits = r.device.limits();
        // Captures share this installed light_plan. Media also consumes map transport.
        // The profile resolution is a quality floor. Large authored views and
        // captures need correspondingly sized maps, bounded by the actual GPU.
        let fibre_view_side=std::iter::once(render.resolution).chain(input.images.values().filter_map(|i|i.render.as_ref()).map(|view|view.resolution)).flatten().max().unwrap_or(1);
        let mut light_plan = LightPlan::logical(&nodes, &limits, medium.is_some())?;
        if let Some(medium) = &mut medium {
            // Keep animated/imported reservations: an empty installed plan is
            // the proof, not the current frame's zero light count or energy.
            slang.medium=Some(std::sync::Arc::new(medium.native.clone()));
        }
        light_plan.blackbody = crate::light_temperature::table(r, &nodes)?;
        ensure!(r.device.features().contains(wgpu::Features::BUFFER_BINDING_ARRAY),
            crate::gpu_memory::DeviceLimit("shadow depth-bounds tables require buffer binding arrays on this backend".into()));
        // Thin and solid interfaces use the same scene continuation. Refine
        // transport against installed materials, independently of solid occupancy.
        let transport_rays = ray_refraction && draws.iter().any(|draw| materials[draw.material].transmissive);
        let refraction_changed = transport_rays != ray_refraction;
        if refraction_changed || connection_changed || shadow_changed {
            ray_refraction = transport_rays;
            ray_shadows = light_shadow_rays || ray_refraction || effects.probes || promoted_transport || program_trace || compute_trace;
            rays = crate::rt::installed_rays(&scene, ray_refraction) || light_shadow_rays || promoted_transport || program_trace || compute_trace;
            slang.ray_refraction = ray_refraction;
        }
        // Refine imported reservations against the same decoded material table
        // used by the graph. Keep the preflight conservative only until here.
        // Mesh emission is a medium light source even when no other effect
        // requested scene rays. Reuse the installed source and hit tables.
        let medium_sources=medium.as_ref().is_some_and(|m|!m.native.scattering_free) && draws.iter().any(|draw| draw.implicit.is_none() && draw.count>0
            && materials[draw.material].physical_emitter() && draw.instances.clone().any(|object| {
                let root=&roots[object as usize];let node=root.node;
                let visible_track=std::iter::successors(Some(node),|&id|nodes[id].parent).any(|id|animation.targets().any(|target|matches!(target,
                    vork_scene::TrackTarget::Node {node,property:vork_scene::NodeProperty::Visibility(vork_scene::Ray::All)} if *node==nodes[id].node.name)));
                let imported_track=root.imported_pose.is_some() && animation.targets().any(|target|matches!(target,
                    vork_scene::TrackTarget::Imported {asset,property:vork_scene::NodeProperty::Visibility(vork_scene::Ray::All),..} if *asset==nodes[node].node.name));
                (buffers::node_visibility_mask(&nodes,node)&constants::RAY_VOLUME!=0 || visible_track)
                    && (root.imported_visibility&constants::RAY_VOLUME!=0 || imported_track)
                    && (objects[object as usize].metadata[1]&constants::RAY_VOLUME!=0
                        || visible_track || imported_track || nodes[node].dynamic)
            }));
        rays |= medium_sources;
        ray_shadows |= medium_sources;
        let transport_needed=allocation_dependencies.beauty || allocation_dependencies.aovs
            || allocation_dependencies.catcher || effects.implicit;
        if !transport_needed {
            effects.path=false; effects.subsurface=false; effects.fibre=false;
            slang.ray_refraction=false; slang.connection_transmission=false;
            slang.medium=None;
        }
        // This is the lowered/compiled proof, including imported and program alpha.
        // Resolve it before choosing the shared opaque pipeline registry.
        slang.unit_ior = materials.iter().all(|material|!material.weighted_ior());
        slang.unit_boundary_ior = materials.iter().filter(|material|material.transmissive).all(|material|!material.weighted_ior());
        slang.no_gltf_tint = materials.iter().all(|material|!material.gltf_tint());
        slang.gaussians = draws.iter().any(|draw|draw.splats.is_some());
        for draw in draws.iter().filter(|draw|draw.splats.is_some()) {
            ensure!(!materials[draw.material].solid_transmission(),"Gaussian splats have no closed dielectric boundary; use a closed triangle mesh or a thin-walled std/openpbr material");
        }
        slang.no_points = !draws.iter().any(|draw|draw.exit_geometry.as_ref().is_some_and(|g|!g.point_centers.is_empty()));
        slang.surface_coverage = draws.iter().any(|draw|materials[draw.material].surface_coverage());
        slang.attribute_coverage = attributes != 0 && draws.iter().any(|draw|materials[draw.material].attribute_coverage());
        slang.connection_transmission = rays && (draws.iter().any(|draw|materials[draw.material].transmissive) || input.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit)));
        // Field carriers are absent from the triangle ray masks. Their exact
        // crossing walk owns absorption; triangle containment cannot find them.
        slang.no_shadow_containment=!draws.iter().any(|draw| {
            if !materials[draw.material].solid_absorption() {return false;}
            let node=&nodes[roots[draw.object as usize].node].node;
            !matches!(&node.kind,NodeKind::Mesh {geometry,..} if matches!(input.geometry[geometry].topology,vork_scene::v2::Topology::Implicit))
        });
        // Reuse the proven overlap capacity before launching shader compilation.
        // Compute placement is not installed yet; reserve all domains when it
        // can change placements. Ordinary animation/deformation is already known.
        let shadow_solids=if rays {crate::passes::path::solids_for(&draws,&roots,|m|materials[m].solid_transmission())?} else {Default::default()};
        // Retain the closure proof before compiling capacity-specialized shadows.
        // Shared-index deformation preserves closure; welded attribute seams
        // without that proof remain unknown, since deformation may open them.
        let mut welds=std::collections::HashMap::new();
        let mut open_edges=std::collections::HashMap::new();
        for draw in &mut draws {
            if !draw.instances.clone().any(|object|shadow_solids.owners.get(object as usize).is_some_and(|&solid|solid!=u32::MAX)) {continue;}
            if programs.as_ref().is_some_and(|p|p.displaces(materials[draw.material].record.tag[0])) {continue;}
            let Some(geometry)=&draw.exit_geometry else {continue;};
            let start=draw.primitive_start as usize*3;let end=start+draw.count as usize;
            let address=std::sync::Arc::as_ptr(geometry);
            let keys=welds.entry(address).or_insert_with(||if draw.deform.is_some() {
                let identity=address as usize as u64;
                (0..geometry.vertices.len()).map(|i|[identity as u32,(identity>>32) as u32,i as u32]).collect()
            } else {crate::passes::path::weld_keys(geometry)});
            draw.open_edges=Some(open_edges.entry((address,start,end)).or_insert_with(||std::sync::Arc::new(crate::passes::path::open_edges(&geometry.indices[start..end],|i|keys[i as usize]))).clone());
        }
        drop(welds);drop(open_edges);
        // Prove absence of the contact/priority feature independently of the
        // conservative volume occupancy bound (open media may have finite faces).
        let contact_solids=crate::passes::path::occupancy(&shadow_solids,&draws,&roots,&nodes,&objects,None,&animation,&targets).0;
        slang.no_boundary_transactions=shadow_solids.count>0 && !nodes.iter().any(|n|crate::compute::places_instances(input,&n.node))
            && !input.geometry.values().any(|g|matches!(g.topology,vork_scene::v2::Topology::Implicit))
            && crate::passes::path::plain_boundaries(&contact_solids,draws.iter().filter(|d|materials[d.material].solid_transmission())
                .map(|d|materials[d.material].record.extra[0].to_bits() as i32));

        // Only absorbing domains enter containment. Clear transmitting solids
        // still need the closure proof above for camera occupancy, but must not
        // reserve shadow records (especially with unrelated compute placement).
        let contained=if shadow_solids.count==0 {Default::default()}
            else {crate::passes::path::solids_for(&draws,&roots,|m|materials[m].solid_absorption())?};
        slang.shadow_capacity=if nodes.iter().any(|n|crate::compute::places_instances(input,&n.node)) {contained.count}
            else {crate::passes::path::occupancy(&contained,&draws,&roots,&nodes,&objects,None,&animation,&targets).1};
        // The same installed fact selects both the candidate evaluator and
        // the hit tables it reads, including shadow-only raster views.
        let program_coverage = medium_sources || slang.connection_transmission || slang.attribute_coverage
            || programs.as_ref().and_then(|p|p.native.as_ref()).is_some_and(|p|p.coverage)
            || (effects.fibre && rays);
        if let Some(medium)=&mut medium {
            medium.native.mesh_sources=medium_sources;
            if qualified_media {
                #[cfg(not(any(test, feature="test-support")))]
                let force_general=false;
                #[cfg(any(test, feature="test-support"))]
                let force_general=options.test_force_general_media;
                medium.select_boundary_format(&draws,&native_ids,force_general);
            }
            slang.medium=Some(std::sync::Arc::new(medium.native.clone()));
        }
        // Imported assets reserve possible light connections before their contents
        // are known. Enforce transport requirements only after that reservation
        // has been refined against the actual lights and material draws.
        ensure!(!light_shadow_rays || r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY),
            crate::gpu_memory::DeviceLimit("effect unavailable: light shadows need hardware ray queries; use a ray-query capable adapter".into()));
        if promoted_transport {
            ensure!(r.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY | crate::rt::HIT_BUFFER_FEATURES),
                crate::gpu_memory::DeviceLimit("surface transport needs mesh-thickness or RGB light-connection ray queries and hit buffers; use a GPU supporting ray queries".into()));

        }
        ensure!(!rays || !program_coverage || r.device.features().contains(crate::rt::HIT_BUFFER_FEATURES), crate::gpu_memory::DeviceLimit("ray material visibility requires storage buffer binding arrays".into()));
        (slang.textures,slang.samplers)=textures.counts()?;
        let descriptor_elements=u64::from(slang.textures)+u64::from(slang.samplers)
            +u64::from(crate::passes::decoupled::evaluation_pages(&limits,slang.aov_count))
            +u64::from(constants::VISIBILITY_REFLECTION_WORDS)+u64::from(constants::RAY_HIT_BUFFERS)+crate::camera_native::TABLES.iter().map(|t|t.resources.len() as u64).sum::<u64>();
        ensure!(slang.samplers <= limits.max_binding_array_sampler_elements_per_shader_stage,
            crate::gpu_memory::device_limit("scene material samplers", format_args!("{} samplers",slang.samplers), format_args!("max_binding_array_sampler_elements_per_shader_stage={}",limits.max_binding_array_sampler_elements_per_shader_stage), "reuse material image samplers or reduce material image inputs"));
        ensure!(descriptor_elements <= u64::from(limits.max_binding_array_elements_per_shader_stage),
            crate::gpu_memory::device_limit("scene material/ray tables", format_args!("{descriptor_elements} descriptors"), format_args!("max_binding_array_elements_per_shader_stage={}",limits.max_binding_array_elements_per_shader_stage), "reduce material image inputs or geometry buffer pool sizes"));
        // The shared builtin registry is point-capable. Preserve the no-point
        // proof by using the scene-keyed registry for its lean hit variants.
        let builtin=!slang.gaussians && !slang.software_raster && !slang.no_points && propagation.is_none() && !spectral && camera_programs.is_empty() && slang.environment.is_none() && programs.is_none() && medium.is_none()
            && !ray_shadows && !slang.ray_refraction && !slang.connection_transmission && !slang.surface_coverage && !effects.denoise && !effects.probes && render.aovs.is_empty()
            && slang.attributes==0 && slang.textures==vork_slang::registry::MATERIAL_TEXTURE_CAPACITY && slang.samplers==vork_slang::registry::MATERIAL_SAMPLER_CAPACITY;
        // Material bindings can be prepared before kernel planning. The final
        // registry creates the same descriptor layout from these installed facts.
        let mat_layout=r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label:Some("material"),entries:&pipelines::material_entries_for(slang.textures.max(1),slang.samplers.max(1))});
        for d in &draws {
            if let Some(a) = &d.deform {
                if a.native.is_none() {
                    assets[a.asset.expect("glTF deformation has an asset")].active = true;
                }
            }
        }
        let material_bind_start = Instant::now();
        // One immutable storage array in the scene material table.
        // Keep a sentinel for scenes without geometry (zero-sized bindings are invalid).
        // A material's taps may read only texture coordinate sets its geometry has.
        for draw in &draws {
            let missing = materials[draw.material].uv_sets() & !draw.uv_sets;
            ensure!(
                missing == 0,
                "a material texture reads TEXCOORD_{} (uv{}), which its geometry lacks; add that UV set to the geometry or change the texture binding",
                materials[draw.material].uv_sources[missing.trailing_zeros() as usize],
                materials[draw.material].uv_sources[missing.trailing_zeros() as usize]
            );
        }
        // Texture taps follow the program words; identical tap lists share words.
        let mut tap_offsets = std::collections::HashMap::<Vec<u32>, u32>::new();
        let mut records = Vec::with_capacity(materials.len().max(1));
        for material in &materials {
            let mut record = material.record;
            // Every fact it reads is final here: programs, tracks and imports are installed.
            if material.null_connection() {record.tag[2] |= vork_slang::registry::FLAG_NULL_CONNECTION;}
            if !material.taps.is_empty() {
                let words: Vec<u32> = material.taps.iter().flatten().copied().collect();
                record.tag[1] = *tap_offsets.entry(words).or_insert_with_key(|words| {
                    let offset = material_words.len() as u32;
                    material_words.extend_from_slice(words);
                    offset
                });
            }
            records.push(record);
        }
        if records.is_empty() {
            records.push(materials::MaterialRecord::zeroed());
        }
        let record_words =
            (records.len() * std::mem::size_of::<materials::MaterialRecord>() / 4) as u32;
        for camera in &mut camera_programs {
            let start=camera.offset;
            programs::bind_inputs(&mut animation,&camera.material,&camera.program.inputs,&mut material_words[start..start+camera.words.len()],record_words as usize+start,0.)?;
            if let Some(state)=&mut nodes[camera.node].camera {state.offset=record_words as usize+start;state.words.clone_from_slice(&material_words[start..start+camera.words.len()]);}
            camera.offset+=record_words as usize;
        }
        if let Some(primary)=&environment.primary {
            ensure!(!primary.spatial || (records.len()==1 && materials.is_empty()),"spatial environments require an empty geometry table");
            programs::bind_inputs(&mut animation,&primary.material,&primary.layout,
                &mut material_words[..primary.words.len()],record_words as usize,0.)?;
        }
        for object in &mut objects {
            if object.material_index[2] != 0 { object.material_index[1] += record_words; }
        }
        for (record, material) in records.iter_mut().zip(&materials) {
            if record.tag[0] != 0 {
                record.tag[3] += record_words;
            }
            if !material.taps.is_empty() {
                record.tag[1] += record_words;
            }
        }
        if let Some(medium) = &mut medium {
            medium.mesh_words(r, input, &draws, &roots, &native_ids).with_context(|| {
                let names=stages.iter().filter(|(_,stage)|stage.medium).map(|(name,_)|name.as_str())
                    .chain(environment.settings.medium.as_deref()).collect();
                format!("medium domains: {}",medium_use_sites(&scene.nodes,&names).join(", "))
            })?;
        }
        // Only lights explicitly consuming an IES asset retain source-knot data.
        // The material-word table is shared by every radiance consumer.
        let mut emitter_profiles=std::collections::HashMap::new();
        let mut profile_assets=std::collections::HashMap::<String,u32>::new();
        for (id,node) in nodes.iter().enumerate() {
            if let NodeKind::Light { shape, image: Some(name), .. } = &node.node.kind {
                if matches!(shape,vork_scene::LightShape::Environment) {continue;}
                let address=if let Some(address)=profile_assets.get(name) {*address} else {
                    let image=input.images.get(name).context("missing emitter profile image")?;
                    let file=image.file.as_ref().context("local emission profiles require an IES file")?;
                    let path=base.join(file);
                    let profile=crate::ies::parse(&std::fs::read(&path).with_context(||format!("read emission profile {}",path.display()))?)?;
                    let address=record_words+u32::try_from(material_words.len())?;
                    material_words.extend(profile.words());
                    profile_assets.insert(name.clone(),address);address
                };
                emitter_profiles.insert(id,address);
            }
        }
        if !emitter_profiles.is_empty() {
            ensure!(effects.path, "local IES emission currently requires path beauty consumers; raster sampled-emitter integration is not installed yet");
        }
        let medium_base = record_words + material_words.len() as u32;
        if let Some(medium) = &medium {
            material_words.extend_from_slice(&medium.words);
        }
        // Generated bounds are geometry data, shared across all instances. The
        // existing reserved object word addresses them in the retained table.
        let mut generated_offsets = std::collections::HashMap::new();
        let mut instance_attribute_rows = std::collections::HashMap::new();
        let mut material_attribute_sources = Vec::new();
        for (draw_index, draw) in draws.iter().enumerate() {
            let (minimum, maximum) = draw.generated_bounds;
            let extent = maximum - minimum;
            let (lo, hi) = if draw.splats.is_some() {draw.generated_bounds} else {draw.bounds};
            // Deformation may leave the installed bounds; keep those instances eligible.
            let bounded = !draw.evaluated() && lo.is_finite() && hi.is_finite();
            let words = [minimum.x, minimum.y, minimum.z, extent.x, extent.y, extent.z,
                lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, if bounded { 1. } else { 0. }].map(f32::to_bits);
            let provenance = programs.as_ref().is_some_and(|p|p.native.is_some()).then_some(draw.exit_geometry.as_ref()).flatten();
            let identity=provenance.map(|geometry|(false,std::sync::Arc::as_ptr(geometry) as usize,draw.primitive_start,draw.count))
                // Gaussian source IDs are the identity map for the complete
                // capture. Every storage chunk of an object shares this row.
                .or_else(||draw.splats.as_ref().map(|geometry|(true,0,0,geometry.capture_count)));
            for object in draw.instances.clone() {
                let root = &roots[object as usize];
                let node = &nodes[root.node].node;
                let mut row = vec![0u32; 1 + attribute_names.len() * 4];
                let mut published = Vec::new();
                for (slot, name) in attribute_names.iter().enumerate() {
                    let source = match &node.kind {
                        NodeKind::Mesh { instances: Some(instances), .. } => instances.attributes.get(name),
                        _ => None,
                    };
                    if let Some(source) = source.filter(|source| computed_attribute_sources.contains(source)) {
                        row[0] |= 1 << slot;
                        if vork_scene::v2::is_color_attribute(name) { row[1 + slot * 4 + 3] = 1f32.to_bits(); }
                        published.push((source.clone(), root.instance_index.expect("instance row"), slot));
                    } else if let Some(value) = crate::attributes::instance_value(node, root.instance_index, name, &buffer_data) {
                        row[0] |= 1 << slot;
                        row[1+slot*4..1+(slot+1)*4].copy_from_slice(&value.map(f32::to_bits));
                    }
                }
                // Computed rows cannot alias by their initial values: each row
                // retains its source identity and publishes into frame-private words.
                let attribute_row = if !published.is_empty() {
                    let offset = record_words + material_words.len() as u32;
                    material_words.extend_from_slice(&row);
                    material_attribute_sources.extend(published.into_iter().map(|(source, index, slot)|
                        (source, index, offset + 1 + slot as u32 * 4)));
                    offset
                } else if row[0] == 0 { 0 } else {
                    *instance_attribute_rows.entry(row).or_insert_with_key(|row| {
                        let offset = record_words + material_words.len() as u32;
                        material_words.extend_from_slice(row);
                        offset
                    })
                };
                let offset = *generated_offsets.entry((words, identity, attribute_row,
                    draw.deform.as_ref().map(|_|draw_index),
                    programs.as_ref().is_some_and(|p|p.displaces(records[draw.material].tag[0]))
                        .then_some((draw.material,root.time_offset.to_bits())))).or_insert_with(|| {
                    let offset = record_words + material_words.len() as u32;
                    material_words.extend_from_slice(&words);
                    material_words.push(attribute_row);
                    if let Some(geometry)=provenance {
                        let first=draw.primitive_start as usize;
                        material_words.extend_from_slice(&geometry.primitive_ids[first..first+draw.count as usize/3]);
                    } else if let Some(geometry)=&draw.splats {
                        material_words.extend(0..geometry.capture_count);
                    }
                    offset
                });
                objects[object as usize].metadata[2] = offset;
            }
        }
        for name in animation.material_inputs().into_iter().map(|(name, _)| name).collect::<std::collections::BTreeSet<_>>() {
            if input.materials[&name].program == "std/openpbr" { continue; }
            if let Some(&slot) = native_ids.get(&name) {
                for (slot, offset) in std::iter::once((slot, 0.)).chain(phased.iter().filter(|((s, _), _)| *s == slot).map(|(&(_, bits), &copy)| (copy, f32::from_bits(bits)))) {
                    let record = records[slot];
                    if record.tag[0] != 0 {
                        let layout = programs.as_ref().unwrap().input_layout(record.tag[0]);
                        let relative = record.tag[3] as usize - record_words as usize;
                        programs::bind_inputs(&mut animation, &name, layout,
                            &mut material_words[relative..relative + layout.words], record.tag[3] as usize, offset)?;
                    }
                }
            }
        }
        if let Some(medium) = &medium {
            medium.bind_inputs(&mut animation, &mut material_words[(medium_base - record_words) as usize..], medium_base as usize)?;
        }
        let animation = std::sync::Arc::new(animation);
        for asset in &mut assets {
            asset.attach_plan(std::sync::Arc::clone(&animation));
        }
        links.finish(&mut material_words, record_words, &mut objects)?;
        let mut record_bytes = bytemuck::cast_slice::<_, u8>(&records).to_vec();
        record_bytes.extend_from_slice(bytemuck::cast_slice(&material_words));
        // Surface/compute consumers sample the exposure clock. Main final-only
        // compositor images sample the nominal output clock after accumulation.
        let mut sampled_media:std::collections::BTreeSet<_>=textures.footage.keys().cloned().collect();
        for name in &planned_image_order {
            sampled_media.extend(compositor::image_reads(&compositor,input.images[name].compositor()));
        }
        let final_media=if output_dependencies.color {compositor::image_reads(&compositor,input.compositor())} else {Default::default()};
        let mut file_images = std::collections::BTreeMap::new();
        // Compositor image handles reuse the exact scene texture installation.
        let mut post_materials=std::collections::BTreeSet::new();
        let mut add=|names:&[vork_scene::v2::CompositeEntry]| {
            let (_,live)=compositor::dependencies(&compositor,names);
            post_materials.extend(names.iter().zip(live).filter(|(_,live)|*live).map(|(entry,_)|entry.material().clone()));
        };
        if output_dependencies.color {add(input.compositor());}
        for name in &planned_image_order {add(input.images[name].compositor());}
        for name in post_materials {
            let material=&input.materials[&name];
            for (field,value) in &material.inputs {
                if !compositor[&name].reads_input(field) {continue;}
                if let vork_scene::v2::Input::Texture(binding) = value {
                    let image = &input.images[&binding.image];
                    if image.render.is_none() && !file_images.contains_key(&binding.image) {
                        let (slot, _) = textures.scene_image(&binding.image,image,base)?;
                        if let materials::TextureSlot::File(view) = slot {file_images.insert(binding.image.clone(),view);}
                    }
                }
            }
        }
        for (name,image) in &mut textures.footage {
            image.per_sample=sampled_media.contains(name);
            image.final_compositor=final_media.contains(name);
        }
        let material_table = textures.table(&mat_layout, record_bytes, shadow_solids.count>crate::passes::path::INLINE_BOUNDARIES)?;
        let mut materials: Vec<_> = materials
            .into_iter()
            .map(materials::PendingMaterial::finish)
            .collect();
        // Camera refraction reaches off-camera materials too. Use the complete
        // installed union before raster variant budgeting widens its keys.
        slang.no_film = materials::installed_features(&materials) & materials::FEATURE_FILM == 0;
        if rays {
            crate::rt::prepare_coverage(&mut draws, &materials)?;
        }
        // Decide geometry-effect admission once, from installed draws, bindings and
        // material facts. None of these failures has an approximate evaluator.
        if programs.as_ref().is_some_and(|p| p.has_displacement()) {
            let mut assignment_groups = std::collections::HashMap::<_, Vec<&Draw>>::new();
            for draw in &draws {
                let root = &roots[draw.object as usize];
                assignment_groups.entry((root.node, root.instance_index, draw.gltf_node.0))
                    .or_default().push(draw);
            }
            // Displacement precedes the compute/material publication. Resolve
            // this unsupported bound dependency per material owner, preserving
            // static uses of the same program on other nodes.
            let computed_instance_displacement: std::collections::HashSet<_> = if computed_attribute_sources.is_empty() {
                std::collections::HashSet::new()
            } else { draws.iter().filter_map(|draw| {
                let program = programs.as_ref().unwrap();
                let record = &records[draw.material];
                if !program.displaces(record.tag[0]) { return None; }
                let node = roots[draw.object as usize].node;
                let NodeKind::Mesh { instances: Some(instances), .. } = &nodes[node].node.kind else { return None; };
                program.displacement_descriptors(record.tag[0], (record.tag[3] - record_words) as usize, &material_words)
                    .iter().any(|(_, descriptor)| instances.attributes.get(&attribute_names[(descriptor[0] & 255) as usize])
                        .is_some_and(|source| computed_attribute_sources.contains(source)))
                    .then_some((draw.material, node))
            }).collect() };
            let displaces = |draw: &Draw| programs.as_ref().unwrap()
                .displaces(records[draw.material].tag[0])
                && !computed_instance_displacement.contains(&(draw.material, roots[draw.object as usize].node));
            for group in assignment_groups.values() {
                // Single-material domains and surface-only differences cannot conflict.
                // Avoid per-vertex storage for those common cases.
                if group.iter().all(|draw| draw.material == group[0].material)
                    || !group.iter().any(|draw| displaces(draw)) { continue; }
                let mut assignments = std::collections::HashMap::new();
                // Record displaced assignments first, then check rigid siblings.
                for displaced_pass in [true, false] {
                    for draw in group {
                        if displaces(draw) != displaced_pass { continue; }
                        let Some(geometry) = draw.exit_geometry.as_ref() else { continue; };
                        // Source POSITION accessor/index identities survive imported
                        // normal/attribute splits. Only referenced vertices participate.
                        let first = draw.primitive_start as usize * 3;
                        for &vertex in &geometry.indices[first..first + draw.count as usize] {
                            let identity = geometry.vertex_ids.get(vertex as usize)
                                .copied().unwrap_or(u64::from(vertex));
                            if let Some(&material) = assignments.get(&identity) {
                                if material!=draw.material {
                                    refused_geometry_effects.insert(format!("node {:?}: conflicting displacement materials on shared geometry would omit displacement; use one displacement material or split the geometry",nodes[roots[draw.object as usize].node].node.name));
                                }
                            }
                            if displaced_pass { assignments.insert(identity, draw.material); }
                        }
                    }
                }
            }
            let mut displacement_seams = std::collections::HashMap::<_, Vec<u32>>::new();
            for draw in &draws {
                let record = &records[draw.material];
                let Some(programs) = programs.as_ref().filter(|p| p.displaces(record.tag[0])) else {
                    continue;
                };
                let node=&nodes[roots[draw.object as usize].node].node;
                let implicit=matches!(&node.kind,NodeKind::Mesh {geometry,..} if matches!(input.geometry[geometry].topology,vork_scene::v2::Topology::Implicit));
                if implicit || computed_instance_displacement.contains(&(draw.material, roots[draw.object as usize].node)) { continue; }
                let root = &roots[draw.object as usize];
                let descriptors = programs.displacement_descriptors(
                    record.tag[0],
                    (record.tag[3] - record_words) as usize,
                    &material_words,
                );
                let geometry = draw
                    .exit_geometry
                    .as_ref()
                    .context("displacement requires triangle geometry")?;
                if !geometry.point_centers.is_empty() {
                    if programs.displacement_uses_normal(record.tag[0]) { continue; }
                }
                // With no vertex-varying inputs, every seam comparison is the
                // same empty tuple, independent of instance identity or time.
                if descriptors.is_empty() && !programs.displacement_uses_normal(record.tag[0]) {continue;}
                let first = draw.primitive_start as usize * 3;
                for &index in &geometry.indices[first..first + draw.count as usize] {
                    let vertex = index as usize;
                    let identity = geometry
                        .vertex_ids
                        .get(vertex)
                        .copied()
                        .unwrap_or(vertex as u64);
                    let mut values = Vec::new();
                    if programs.displacement_uses_normal(record.tag[0]) {
                        values.extend(geometry.vertices[vertex].NORMAL.map(f32::to_bits));
                    }
                    for (width, descriptor) in &descriptors {
                        let slot = (descriptor[0] & 255) as usize;
                        let channel = descriptor[0] >> 8;
                        let instance = crate::attributes::instance_value(&nodes[root.node].node,
                            root.instance_index, &attribute_names[slot], &buffer_data);
                        let attribute = instance.as_ref().or_else(|| geometry.attributes
                            .get(&attribute_names[slot]).map(|values| &values[vertex]))
                            .context("bound displacement attribute missing")?;
                        for component in 0..*width as usize {
                            let source = if channel < 4 {
                                channel as usize
                            } else {
                                component
                            };
                            values.push(
                                (attribute[source] * f32::from_bits(descriptor[component + 1]))
                                    .to_bits(),
                            );
                        }
                    }
                    let key = (root.node, root.instance_index, draw.gltf_node.0, identity);
                    if let Some(before) = displacement_seams.insert(key, values.clone()) {
                        if before!=values {
                            refused_geometry_effects.insert(format!("node {:?}: inconsistent displacement attributes at shared vertices would omit displacement; use continuous attributes or split the geometry",nodes[root.node].node.name));
                        }
                    }
                }
            }
            for draw in &draws {
                let program = programs.as_ref().unwrap();
                let tag = records[draw.material].tag[0];
                if !program.displaces(tag) { continue; }
                let node = &nodes[roots[draw.object as usize].node].node;
                let alternative = if computed_instance_displacement.contains(&(draw.material, roots[draw.object as usize].node)) {
                    Some("displacement sampling computed instance attributes; use static/authored attribute values or pre-baked displaced geometry")
                } else if draw.computed && draw.exit_geometry.as_ref().is_some_and(|g|!g.point_centers.is_empty()) {
                    Some("displacement on computed point geometry; compute offsets in the geometry producer or use triangle geometry")
                } else if matches!(&node.kind, NodeKind::Mesh {geometry,..} if matches!(input.geometry[geometry].topology,vork_scene::v2::Topology::Implicit)) {
                    Some("displacement on implicit geometry; evolve the scalar field or use pre-baked triangle geometry")
                } else if draw.exit_geometry.as_ref().is_some_and(|g| !g.point_centers.is_empty()) && program.displacement_uses_normal(tag) {
                    Some("normal-dependent point displacement; use an authored direction attribute or triangle geometry")
                } else { None };
                if let Some(alternative) = alternative {
                    refused_geometry_effects.insert(format!("node {:?}: entirely omitted {alternative}", node.name));
                }
            }
        }
        let compute_nodes: std::collections::HashSet<usize> = nodes.iter().enumerate()
            .filter_map(|(i, node)| crate::compute::places_instances(input, &node.node).then_some(i))
            .collect();
        for &i in &compute_nodes {
            if draws.iter().any(|d| roots[d.object as usize].node == i && materials[d.material].transparent) {
                refused_geometry_effects.insert(format!("node {:?}: computed instance placement on transparent material would be omitted; use an opaque material or authored placements", nodes[i].node.name));
            }
        }
        ensure!(refused_geometry_effects.is_empty(), "unsupported geometry effects:\n{}", refused_geometry_effects.into_iter().collect::<Vec<_>>().join("\n"));

        let displacement = if programs.as_ref().is_some_and(|p| p.has_displacement()) {
            let mut displacement_meshes = std::collections::HashMap::new();
            for draw in &mut draws {
                if programs.as_ref().unwrap().displaces(records[draw.material].tag[0]) {
                    ensure!(
                        draw.layout.deform,
                        "displacement geometry must carry evaluated vertex frames"
                    );
                    let geometry = draw
                        .exit_geometry
                        .as_ref()
                        .context("displacement requires retained triangle geometry")?;
                    let timed = animated.contains(&draw.material) || programs.as_ref().unwrap().displacement_timed(records[draw.material].tag[0]);
                    let key = (std::sync::Arc::as_ptr(geometry), timed);
                    let mesh = if let Some(mesh) = displacement_meshes.get(&key) {
                        std::sync::Arc::clone(mesh)
                    } else {
                        let mesh = displacement::Mesh::new(r, geometry, &attribute_names, timed)?;
                        displacement_meshes.insert(key, mesh.clone());
                        mesh
                    };
                    draw.displacement = Some(mesh);
                }
            }
            draws
                .iter()
                .any(|d| d.displacement.is_some())
                .then(|| {
                    displacement::Pipeline::new(
                        r,
                        programs.as_ref().and_then(|p| p.native.as_deref()).unwrap(),
                        &mat_layout,
                        attribute_names.len(),
                        draws.iter().any(|draw|draw.computed && draw.displacement.is_some() && draw.layout.custom!=0)
                            && crate::geometry_chain::Kind::before(&geometry_stages,crate::geometry_chain::Kind::Compute,crate::geometry_chain::Kind::Displace),
                    )
                })
                .transpose()?
        } else {
            None
        };
        // Only beauty views consume solid exit traversal. Path controls and
        // shadow passes do not. Include every reachable render-image capture.
        for view in views.iter().filter(|view|view.settings.integrator.raster().is_some_and(|r|r.refraction==vork_scene::Refraction::Screen)) {
            let included=crate::render_images::included_nodes(&nodes,view.root.as_deref())?;
            refuse_screen_contents(&draws,&roots,&included,&objects,&materials,&records)?;
        }
        let mut needs_raster_solid_exit = vec![false; nodes.len()];
        if !effects.path {
            for settings in views.iter().filter(|settings| settings.settings.integrator.raster().is_some_and(|r| r.refraction != vork_scene::Refraction::Ray)) {
                let included = crate::render_images::included_nodes(&nodes, settings.root.as_deref())?;
                for (needed, included) in needs_raster_solid_exit.iter_mut().zip(included) {
                    *needed |= included;
                }
            }
        }
        let mut exit_trees = std::collections::HashMap::new();
        // Whether a path solid is closed spans all its material groups, opaque
        // ones included: every draw of a mesh that owns a solid records its range's edges.
        // Rows are renumbered by group_draws below, so occupancy's solids are built again after it.
        let shadow_occupancy=rays && draws.iter().any(|d|materials[d.material].solid_absorption);
        let probe_field=render.settings.integrator.probe_volume().map(|(volume,_)| {
            let eyes=probe_eyes.clone().unwrap_or_else(||nodes.iter().filter(|n| matches!(n.node.kind, NodeKind::Camera { .. }))
                .map(|n| n.world.w_axis.truncate()).collect::<Vec<_>>());
            let demand=nodes.iter().any(|n| crate::compute::places_instances(input,&n.node));
            crate::probes::placement::Field::in_frame(&volume,&draws,&objects,&eyes,demand,probe_frame.unwrap_or_default(),&r.device)
        }).transpose()?;
        let mut refit_topology = std::collections::HashMap::new();
        for draw in &mut draws {
            // A program can fold its displacement away. Such points retain the
            // installed vertex streams and must not select compact vertex pulling.
            if !draw.computed && draw.displacement.is_none() {draw.layout.points=false;}
            if let Some(geometry) = draw.exit_geometry.take() {
                if rays && draw.evaluated() && !draw.layout.points {
                    draw.refit_indices = Some(refit_topology.entry(draw.indices.clone()).or_insert_with(|| {
                        init(&r.device, "immutable BLAS topology", bytemuck::cast_slice(&geometry.indices), wgpu::BufferUsages::BLAS_INPUT)
                    }).clone());
                }
                if materials[draw.material].solid_transmission
                    && !ray_refraction
                    && needs_raster_solid_exit[roots[draw.object as usize].node]
                {
                    let key = (draw.vertices.clone(), draw.vertex_offset);
                    draw.exit_bvh = Some(exit_trees
                        .entry(key)
                        .or_insert_with(|| crate::transmission::mesh(&r.device, draw, &geometry))
                        .clone());
                }
                if draw.computed || !geometry.point_centers.is_empty() { draw.exit_geometry=Some(geometry); }
            }
        }
        let mut jobs = std::collections::HashMap::new();
        for draw in &mut draws {
            if let Some(deform) = &mut draw.deform {
                let key = (
                    deform.source.clone(),
                    deform.native.as_ref().map(|binding| binding.mesh),
                    deform.asset,
                    deform.node,
                    deform.skin,
                    roots[draw.object as usize].time_offset.to_bits(),
                );
                let next = jobs.len();
                deform.job = *jobs.entry(key).or_insert(next);
            }
        }
        // CPU-fitted fibre captures cannot bound GPU-written placements.
        // Probe support covers the cameras' complete clip regions and captures
        // through the current computed TLAS after placement publication.
        if rays && !compute_nodes.is_empty() {
            ensure!(!effects.fibre,
                "computed ray placement does not support CPU-fitted fibre captures; choose --profile std/path");
            let moving_medium=compute_nodes.iter().any(|&index| matches!(&nodes[index].node.kind,
                NodeKind::Mesh { materials, .. } if materials.iter().any(|name|stages.get(name).is_some_and(|stages|stages.medium))));
            ensure!(!moving_medium,
                "computed ray placement does not support moving medium boundaries; certified medium domains must follow the same GPU placement snapshot");
        }
        let draw_groups = group_draws(
            &nodes,
            &compute_nodes,
            &materials,
            &mut draws,
            &mut objects,
            &mut roots,
            r.device
                .features()
                .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE),
        );
        let mut leaders = std::collections::HashMap::new();
        for (index, draw) in draws.iter_mut().enumerate() {
            if let Some(deform) = &mut draw.deform {
                deform.leader = *leaders.entry(deform.job).or_insert(index);
            }
        }
        let mut evaluations = std::collections::HashMap::new();
        for (index, draw) in draws.iter_mut().enumerate() {
            if let Some(mesh) = &draw.displacement {
                let object = draw.object as usize;
                let key = (std::sync::Arc::as_ptr(mesh), draw.material,
                    roots[object].time_offset.to_bits(), objects[object].metadata[2],
                    draw.deform.as_ref().map(|d| d.job),draw.computed.then_some(draw.object));
                draw.displacement_leader = Some(*evaluations.entry(key).or_insert(index));
            }
        }
        let (mut dense_destinations,dense_certificates)=medium.as_ref().map(|m|m.dense_bindings(input,medium_base,&draws,&native_ids)).transpose()?.unwrap_or_default();
        // Surface snapshots (including phased copies) have their own packed
        // descriptors. Publish both cells and range into every retained copy.
        if let Some(programs)=&programs {
            for (name,&slot) in &native_ids {
                for copy in std::iter::once(slot).chain(phased.iter().filter(|((s,_),_)|*s==slot).map(|(_, &copy)|copy)) {
                    let record=records[copy];
                    if record.tag[0]==0 {continue;}
                    let layout=programs.input_layout(record.tag[0]);
                    let base=record.tag[3] as usize;
                    for field in &layout.fields {
                        if !matches!(field.ty,vork_slang::Type::FloatGrid|vork_slang::Type::VectorGrid) {continue;}
                        let Some(vork_scene::v2::Input::Grid(binding))=input.materials[name].inputs.get(&field.name) else {continue;};
                        if input.buffers[&binding.grid].dims.is_none() || !computed_attribute_sources.contains(&binding.grid) {continue;}
                        let descriptor=material_words[base-record_words as usize+field.offset];
                        dense_destinations.push((binding.grid.clone(),record.tag[3]+descriptor+4));
                    }
                }
            }
        }
        dense_destinations.sort();
        dense_destinations.dedup();
        let dense=if scene_inputs {crate::dense::Dense::install(r,input,program_buffers.as_ref(),dense_destinations,dense_certificates)?} else {None};
        ensure!(!draws.iter().any(|d|d.computed) || render.settings.integrator.probe_volume().is_none(),
            "computed geometry cannot use CPU-planned probe support; automatic lighting selects complete path transport");

        let mut field_objects=std::collections::BTreeMap::<String,u32>::new();
        for draw in &draws {
            if let NodeKind::Mesh {geometry,..}=&nodes[roots[draw.object as usize].node].node.kind {
                let geometry=&input.geometry[geometry];
                if matches!(geometry.topology,vork_scene::v2::Topology::Implicit) {
                    let name=geometry.attributes.values().next().unwrap();
                    field_objects.entry(name.clone()).and_modify(|object|*object=(*object).min(draw.instances.start)).or_insert(draw.instances.start);
                }
            }
        }
        let mut implicit_fields=std::collections::BTreeMap::<String,ImplicitField>::new();
        for draw in &mut draws {
            let node=&nodes[roots[draw.object as usize].node].node;
            if let NodeKind::Mesh {geometry,..}=&node.kind {
                let geometry=&input.geometry[geometry];
                if matches!(geometry.topology,vork_scene::v2::Topology::Implicit) {
                    ensure!(!draw.evaluated(), "implicit geometry cannot use skinning or displacement; evolve the scalar field instead");
                    if let NodeKind::Mesh {materials:bindings,..}=&node.kind {
                        for name in bindings {
                            ensure!(!stages[name].medium, "implicit geometry does not yet support a material medium stage");
                            let subsurface=if let Some(instance)=programs::instance(input,name)? {
                                programs.as_ref().context("implicit material program")?.is_subsurface(&instance)
                            } else {materials::openpbr_values(input,Some(name),animated_inputs.get(name).unwrap_or(&Default::default()),|_|Ok(0))?.capabilities.subsurface};
                            ensure!(!subsurface, "implicit geometry does not yet support subsurface scattering");
                        }
                    }
                    let name=geometry.attributes.values().next().unwrap();
                    let pages=program_buffers.as_ref().unwrap();
                    let dims=input.buffers[name].dims.unwrap();
                    let spans:Vec<_>=pages.bindings[name].spans.iter().map(|s|(pages.storage[s.page as usize].clone(),s.clone())).collect();
                    if !implicit_fields.contains_key(name) {
                        let acceleration=std::sync::Arc::new(crate::implicit_acceleration::Acceleration::new(r,dims,&spans,field_objects[name])?);
                        implicit_fields.insert(name.clone(),ImplicitField {source:name.clone(),statuses:Vec::new(),dims,spans,acceleration});
                    }
                    if let Some(status)=&geometry.status {
                        let binding=&pages.bindings[&status.buffer];
                        ensure!(binding.count==1 && binding.width==1,"geometry.status requires one u32 element");
                        let span=&binding.spans[0];
                        implicit_fields.get_mut(name).unwrap().statuses.push((
                            format!("node {:?}: {}",node.name,status.message),
                            pages.storage[span.page as usize].clone(),u64::from(span.offset)*4));
                    }
                    for object in draw.instances.clone() {objects[object as usize].metadata[1] |= vork_slang::registry::OBJECT_IMPLICIT;}
                    draw.implicit=Some(implicit_fields[name].clone());
                }
            }
        }
        if options.geometry_stages.is_none() {
            for draw in &draws {
                let name=&nodes[roots[draw.object as usize].node].node.name;
                for (stage,active) in [(vork_scene::v2::GeometryStage::Deform,draw.deform.is_some()),(vork_scene::v2::GeometryStage::Compute,draw.computed),(vork_scene::v2::GeometryStage::Displace,draw.displacement.is_some())] {
                    ensure!(!active || input.geometry_stages.contains(&stage),"geometry_stages omits {} required by node {:?}; add \"{}\" to geometry_stages",stage.name(),name,stage.name());
                }
            }
        }
        let sources: Vec<_> = nodes.iter().enumerate().map(|(i,n)|
            (!crate::compute::places_instances(input,&n.node) || compute_nodes.contains(&i)).then_some(&n.node)).collect();
        // A source row cannot represent several displaced flat corners when
        // deformation can change their normals independently.
        let reference_names=compute::reference_geometries(input, &prepared_compute)?;
        for draw in draws.iter().filter(|draw|draw.deform.is_some() && draw.displacement.is_some()) {
            let root=&roots[draw.object as usize];
            let node=&nodes[root.node].node;
            let NodeKind::Mesh {geometry,..}=&node.kind else {continue;};
            let shape=&input.geometry[geometry];
            let canonical=reference_names.contains(&node.name) || draw.computed && crate::geometry_chain::Kind::before(&geometry_stages,crate::geometry_chain::Kind::Displace,crate::geometry_chain::Kind::Compute);
            if !canonical || shape.attributes.contains_key("NORMAL") || !programs.as_ref().unwrap().displacement_uses_normal(records[draw.material].tag[0]) {continue;}
            if let Some(indices)=&shape.indices {
                let mut used=std::collections::HashSet::new();
                ensure!(buffer_data.data[indices].words.iter().all(|&vertex|used.insert(vertex)),
                    "node {:?}: normal-dependent canonical displacement has shared deforming flat-normal rows; author NORMAL or split triangle vertices before compute",node.name);
            }
        }
        // Canonical geometry dispatches run during compute initialization, so
        // their objects need the same material routing as presentation draws.
        for draw in &draws {
            for object in draw.instances.clone() {
                objects[object as usize].material_index[0] = phased
                    .get(&(draw.material, roots[object as usize].time_offset.to_bits()))
                    .copied()
                    .unwrap_or(draw.material)
                    as u32;
            }
        }
        let geometry_inputs = if !reference_names.is_empty() || draws.iter().any(|draw| draw.computed
            && matches!(&nodes[roots[draw.object as usize].node].node.kind,
                NodeKind::Mesh {geometry,..} if compute::geometry_has_deformation(input,&input.geometry[geometry]) || draw.displacement.is_some())) {
            let required = compute::predecessor_geometries(&prepared_compute)?;
            crate::geometry_sampling::Inputs::new(r, input, &buffer_data,
                program_buffers.as_ref().context("geometry input buffers")?, &required, &reference_names,
                &nodes, &assets, &animation, &targets, &roots, &draws,
                &geometry_stages, displacement.as_ref(), &material_table, &objects, &attribute_names)?
        } else { None };
        let mut compute = if scene_inputs {compute::Compute::install(
            r,
            input,
            base,
            program_buffers.as_ref(),
            &prepared_compute,
            Some(crate::compute_trace::Scene {nodes:&nodes,objects:&objects,materials:&materials,integrator:&render.settings.integrator,words:&material_words,word_base:record_words as usize,identity:query_identity.map(|h|h.finalize())}),
            &sources,
            &roots,
            &draws,
            render.fps,
            render.duration.clone(),
            geometry_inputs,
            None,
            opened,
        )?} else {None};
        if let Some(compute) = &mut compute {
            compute.bind_material_attributes(program_buffers.as_ref().unwrap(), material_attribute_sources)?;
        }
        if let Some(compute) = &compute {
            for (i, draw) in draws.iter().enumerate() {
                if compute.computed_draw(i) {
                    ensure!(
                        !materials[draw.material].transparent,
                        "compute instances require opaque or alpha-mask material"
                    );
                    for object in draw.instances.clone() {
                        let root = &mut roots[object as usize];
                        nodes[root.node].dynamic = true;
                        root.local = Mat4::IDENTITY;
                        let previous = objects[object as usize];
                        objects[object as usize] = Object {
                            model: nodes[root.node].world,
                            previous_model: nodes[root.node].world,
                            next_model: nodes[root.node].world,
                            ..previous
                        };
                    }
                }
            }
        }
        let computed_geometry = draws.iter().map(|draw| {
            if !draw.computed { return Ok(None); }
            ensure!(!materials[draw.material].surface_coverage && !materials[draw.material].transmissive, "computed geometry requires opaque coverage; dynamic ray alpha and moving medium boundaries are unsupported");
            let node = &nodes[roots[draw.object as usize].node].node;
            let NodeKind::Mesh { geometry, .. } = &node.kind else { unreachable!() };
            let buffers = compute.as_ref().and_then(|compute| compute.geometry_buffers(draw.object))
                .context("computed geometry requires object-private solver storage")?;
            let historical_attributes=draw.displacement.is_some() && draw.layout.custom!=0 && crate::geometry_chain::Kind::before(&geometry_stages,
                crate::geometry_chain::Kind::Compute,crate::geometry_chain::Kind::Displace);
            crate::computed_geometry::Mesh::new(r, &input.geometry[geometry], draw, &attribute_names, buffers,historical_attributes).map(Some)
        }).collect::<Result<Vec<_>>>()?;
        let path_media=medium.as_ref().map(|m|m.words[1]);
        let plan = crate::plan::ScenePlan::new(
            effects, &render, &draws, &materials, &nodes, &roots, &objects, &light_plan, &animation,
            rays, compute.as_ref(),
        );
        let mut plan = plan;
        if !allocation_dependencies.beauty && !allocation_dependencies.aovs && !allocation_dependencies.catcher && !effects.implicit {
            plan.decoupled=false; plan.screen_refraction=false; plan.cluster=false;
            plan.effects.path=false;
        }
        let mut solids = if plan.effects.path || ray_refraction || shadow_occupancy {
            crate::passes::path::solids(&draws, &roots, &materials)?
        } else { Default::default() };
        if (ray_refraction || plan.effects.path) && solids.boundaries.is_empty() {
            // Thin-only transport still queries the common per-object/instance
            // identity tables. Every entry explicitly denotes no solid owner.
            solids.boundaries.resize(objects.len(),u32::MAX);
            solids.owners.resize(objects.len(),u32::MAX);
        }
        let (occupancy_solids, installed_capacity)=crate::passes::path::occupancy(&solids,&draws,&roots,&nodes,&objects,compute.as_ref(),&animation,&targets);
        let boundary_capacity=crate::passes::path::kernel_boundaries(installed_capacity);
        // Camera containment can only find an interior if some sample starts
        // within a solid's bounds. Prove that none does, for every reachable
        // camera at every view size and for the whole install, or keep the query.
        // A scene without solids holds the proof whatever its cameras do, so its
        // kernel key does not depend on them.
        let camera_exterior = occupancy_solids.is_empty() || {
            let resolutions:Vec<[u32;2]>=std::iter::once(render.resolution)
                .chain(input.images.values().filter_map(|i|i.render.as_ref()).map(|view|view.resolution))
                .chain(crate::camera_program::views(input,&nodes,&animation,&options.initial_time).into_values().flatten()).collect();
            let origins=crate::camera_program::origin_balls(&camera_programs,&nodes,&materials::animated_inputs(input)?,&resolutions,
                |node|crate::passes::path::position_at_rest(node,&nodes,&animation,&targets))?;
            crate::passes::path::camera_exterior(&occupancy_solids,origins.as_deref())
        };
        // A thin-only walk retains an empty occupancy state. One unused slot
        // keeps the existing transport specialization distinct from no transport;
        // real solid capacity remains the geometry-derived bound.
        let refraction_capacity=if ray_refraction {boundary_capacity.max(1)} else {0};
        if plan.decoupled {
            crate::passes::decoupled::supported(&plan, !ray_refraction && medium.as_ref().is_some_and(|m|m.words[1]>0) && draws.iter().any(|d|materials[d.material].solid_transmission))?;
            let main_graph=crate::graph::Graph::for_outputs(&plan,readback.any(),Some(output_dependencies))?;
            if main_graph.runs(crate::graph::Pass::Visibility) || (effects.implicit && main_graph.runs(crate::graph::Pass::Forward)) {
            let shape=crate::passes::decoupled::ViewShape {transparent: plan.transparent,
                diffusion: plan.effects.subsurface, reflections: plan.effects.reflections, screen_refraction: plan.screen_refraction, table_only: records.iter().all(|r| r.tag[0] == 0),
                boundaries: refraction_capacity, reserved: reservations.visibility_pool};
            if crate::passes::decoupled::packet_eligible(shape,plan.visibility_samples) && medium.is_none() {
                crate::passes::decoupled::check_packet_minimum(&r.device.limits(),render.resolution,plan.visibility_samples,shape)?;
            } else {
                crate::passes::decoupled::check_view(&r.device.limits(),render.resolution,plan.visibility_samples,shape)?;
            }
            }
        }
        plan.effects.image_feedback = (rays || effects.probes || effects.fibre)
            && input.image_reads(None).iter().any(|name|
                input.images.get(name).is_some_and(|image| image.render.is_some()));
        let graph = crate::graph::Graph::new(&plan, readback.any())?;
        let transients = crate::graph::Transients::new(&graph, &r.residency);
        let random_walk=plan.effects.path && material_spatial_subsurface
            && render.settings.integrator.subsurface_method()==Some(vork_scene::v2::SubsurfaceMethod::RandomWalk);
        let subsurface_ids=if random_walk {crate::passes::path::mesh_identities(&draws,&roots)?} else {Vec::new()};
        // Node topology is fixed at install, including imported and extracted suns.
        // Dynamic lights may cross zero size; retain both emitter implementations.
        // The light_plan covers installed/imported sources and render-image map
        // consumers. Shadow permission survives zero energy and animation.
        let mut light_features = 0;

        if light_plan.nodes.iter().any(|&i| crate::rt::traces_shadows(&nodes[i].node.kind)) {
            light_features |= materials::FEATURE_SHADOW_RAYS;
        }
        for node in &nodes {
            if let NodeKind::Light {
                shape,
                radius,
                angle,
                ..
            } = &node.node.kind
            {
                use vork_scene::LightShape;
                let (kind, finite) = match shape {
                    LightShape::Environment => continue,
                    LightShape::Sun => (0, *angle * 0.5 != 0.),
                    LightShape::Point => (1, *radius != 0.),
                    LightShape::Spot => (2, *radius != 0.),
                    LightShape::Rectangle | LightShape::Ellipse => (3, true),
                };
                light_features |= 1 << (materials::FEATURE_LIGHT_SHIFT + kind);
                light_features |= if finite {
                    materials::FEATURE_FINITE
                } else {
                    materials::FEATURE_PUNCTUAL
                };
                if node.dynamic {
                    light_features |= materials::FEATURE_FINITE | materials::FEATURE_PUNCTUAL;
                }
                if finite || node.dynamic {
                    light_features |= if matches!(shape, LightShape::Rectangle) {
                        materials::FEATURE_RECTANGLE
                    } else {
                        materials::FEATURE_ELLIPSE
                    };
                }
            }
        }
        // Ray hits can reach every installed material, including off-camera draws.
        // Capture the union before raster's variant budget can replace individual
        // material keys with its generic fallback.
        let hit_material_features = materials.iter().fold(materials::FEATURE_SPECIALIZED, |union, m| union | m.features);
        let path_kernel = if plan.effects.path {
            // One rule, every scene: the path kernel's key is the union of what is
            // installed. Materials give their response and transport capabilities
            // (table values, textures, tracks and compiled program proofs); lights
            // give their kind when a scene installs exactly one, the one light fact
            // transport folds on (no light or several kinds compile every kind, and
            // the raster shadow and shape bits select nothing here: either would
            // only split the cache); geometry gives its implicit fields, and the
            // cameras and solids give the proof that no camera origin starts inside one.
            // Each is fixed at install: animation and compute change values, never
            // which capability exists. The same scene always derives the same key, and an
            // edit that adds or removes a used capability derives a different one
            // at its next install.
            let features = materials::installed_features(&materials)
                | crate::forward_variant::folded_light_kinds(light_features)
                | if draws.iter().any(|draw| draw.implicit.is_some()) {
                    crate::forward_variant::FEATURE_IMPLICIT_GEOMETRY
                } else { 0 }
                // The one key that may carry this proof: absence is the safe value, so
                // it is never part of a union.
                | if camera_exterior { crate::forward_variant::FEATURE_CAMERA_EXTERIOR } else { 0 };
            // Beauty convergence does not establish independently authored AOV convergence.
            Some(pipelines::ProgramKernel::Path {adaptive:crate::constants::PATH_USE_ADAPTIVE_SAMPLING && render.aovs.is_empty() && !crate::catcher::has_catchers(input),catchers:crate::catcher::has_catchers(input),random_walk,features,boundaries:boundary_capacity,emitter_profiles:!emitter_profiles.is_empty(),media:path_media})
        } else { None };
        if let Some(compute) = &mut compute {
            compute.prepare_bridge(r, &draws, &objects)?;
        }
        let material_bind_ms = material_bind_start.elapsed().as_secs_f64() * 1000.;
        // Ray shadow candidates read other materials' opacity, not only the
        // receiver's textures. Never prune that lookup using the receiver key.
        let ray_alpha = if ray_shadows {
            materials::texture_feature("geometry_opacity")
        } else {
            0
        };
        for material in &mut materials {
            material.features |= light_features | ray_alpha;
        }
        // Independent hit kernels share immutable installed material facts. Compile
        // Slang and create driver pipelines concurrently, joining before drawing.
        // The hit union already contains every candidate’s texture facts;
        // raster’s receiver-only alpha expansion would erase that global proof.
        let features = hit_material_features | light_features;
        let mut visibility = vec![plan.visibility_samples];
        if animation.has_shutter_animation() { visibility.push(1); }
        if plan.decoupled {
            for settings in &views[1..] {
                visibility.push(crate::render_images::camera_sampling(&nodes, settings, camera)?.2);
            }
        }
        let mut camera_program_features=vec![materials::FEATURE_SPECIALIZED; records.iter().map(|r|r.tag[0] as usize).max().unwrap_or(0)+1];
        for material in &materials {camera_program_features[material.program as usize]|=material.features;}
        materials::bound_features(&mut materials, draws.iter().map(|d| (d.layout, d.material)));
        // Controls and camera evaluations compile the same authored source for
        // independent GPU stages. Install both concurrently before any draw.
        let geometry_keys:std::collections::HashSet<pipelines::GeometryKey> =
            draws
                .iter()
                .zip(&plan.draws)
                .filter(|(draw,_)|draw.rasterized())
                .flat_map(|(draw, flags)| {
                    pipelines::Role::kinds(flags, plan.decoupled).iter().copied().chain(crate::fibre_transport::Stage::ALL.into_iter().filter(|_|plan.effects.fibre && flags.hair).map(|stage|crate::bundles::Kind::Fibre(stage,false))).filter(|&kind|graph.draws(kind)).map(|kind| {
                        (
                            draw.layout,
                            pipelines::Role::of(kind, flags, &materials[draw.material]),
                        )
                    })
                })
                .collect();
        let camera_kernels=plan.decoupled.then(||crate::passes::decoupled::KernelPlan::new(&draws,features,&plan,&render,&visibility,&records,&camera_program_features,ray_refraction,refraction_capacity,!emitter_profiles.is_empty(),variants,slang.programs.is_none()));
        let mut kernels:Vec<_>=path_kernel.into_iter().chain(camera_kernels.iter().flat_map(|plan|plan.kernels.iter().copied())).collect();
        if let Some(pipelines::ProgramKernel::Path {adaptive,..})=path_kernel {
            if slang.path_environment_prepass() {kernels.push(pipelines::ProgramKernel::PathEnvironment {adaptive});}
        }
        if plan.effects.reflections {kernels.push(pipelines::ProgramKernel::Reflections {features});}
        if plan.effects.ray_indirect {kernels.push(pipelines::ProgramKernel::Gi {features});}
        if plan.effects.probes {kernels.extend([false,true].map(|directional|pipelines::ProgramKernel::Probes {features,directional}));}
        let pipelines=if builtin {r.cached_pipelines(raster_samples)} else {r.program_pipelines(RESOLVE,raster_samples,ray_shadows,slang.clone(),&geometry_keys,&kernels)?};
        let cached=&*pipelines;
        let effects_compiler=Some({let pipelines=pipelines.clone();scope.spawn(move ||pipelines.ensure_effects(r,effects))});
        let path=if let Some(pipelines::ProgramKernel::Path {adaptive,catchers,random_walk,features,boundaries,emitter_profiles,media})=path_kernel {
            Some(crate::passes::path::Pipeline::new(r,adaptive,catchers,random_walk,features,boundaries,emitter_profiles,media,slang.path_environment_prepass(),options,&pipelines)?)
        } else {None};
        let asset_pipeline_start = Instant::now();
        let (decoupled, reflections, gi, geometry_pipelines) = std::thread::scope(|scope| {
            let assets = scope.spawn(|| cached.ensure_assets(r, &plan));
            let geometry = scope.spawn(|| pipelines.install_geometry(r, geometry_keys, variants));
            let reflections = plan.effects.reflections.then(|| scope.spawn(||
                crate::passes::reflections::pipeline(r, &pipelines, features, plan.effects.probes, variants)));
            let gi = plan.effects.ray_indirect.then(|| scope.spawn(||
                crate::passes::ao::ray_pipeline(r, &pipelines, features, plan.effects.probes, variants)));
            let decoupled = plan.decoupled.then(||
                crate::passes::decoupled::Install::new(r, &pipelines, &draws, &plan, &render, ray_refraction, refraction_capacity, variants, camera_kernels.as_ref().expect("planned camera kernels"))).transpose();
            use pipelines::join;
            join(assets)?;
            anyhow::Ok((decoupled?,
             reflections.map(join).transpose()?,
             gi.map(join).transpose()?,
             join(geometry)?))
        })?;
        let software_geometry = (pipelines.software_raster && !plan.decoupled).then(|| crate::passes::decoupled::RasterGeometry::new(r,&draws,&plan)).transpose()?;
        // Stop at a driver failure creating these pipelines instead of building
        // the rest of the scene on a lost device.
        r.errors.check()?;
        let vertex_defaults = init(
            &r.device,
            "default vertex channels",
            bytemuck::cast_slice(&[0f32, 0., 0., 0., 1., 1., 1., 1.]),
            wgpu::BufferUsages::VERTEX,
        );
        pipeline_ms +=
            asset_pipeline_start.elapsed().as_secs_f64() * 1000. + textures.pipeline_ms.get();
        // Texture GPU work runs while geometry and its pipelines are installed.
        // Complete readback/alpha correction before any shadow or colour draw.
        // Decoder ownership outlives source metadata; unused images retain the
        // same error behavior and every worker finishes before scene readiness.
        image_decoders.finish()?;
        let texture_finish = Instant::now();
        textures.preparation.finish(r)?;
        textures.upload_ms.set(textures.upload_ms.get() + texture_finish.elapsed().as_secs_f64() * 1000.);
        let instances: Vec<_> = draws
            .iter()
            .enumerate()
            .flat_map(|(i, d)| d.instances.clone().map(move |object| (i, object)))
            .collect();
        let object_bytes = (objects.len().max(1) * std::mem::size_of::<GpuObject>()) as u64;
        let object_limit = limits.max_buffer_size.min(limits.max_storage_buffer_binding_size);
        ensure!(
            object_bytes <= object_limit,
            crate::gpu_memory::device_limit(format_args!("object storage (backend {:?}, adapter {:?}, driver {:?})",r.adapter_info.backend,r.adapter_info.name,r.adapter_info.driver), format_args!("{object_bytes} bytes"), format_args!("{object_limit} bytes"), "reduce geometry instance count")
        );
        let geometry_vertex_bytes = draws
            .iter()
            .flat_map(|d| [d.vertices.clone(), d.attributes.clone()])
            .collect::<std::collections::HashSet<_>>()
            .iter()
            .map(|buffer| buffer.size())
            .sum();
        // The compute bridge broadcasts one evaluated parent per draw. GPU-written
        // transforms must never enter the per-instance CPU dirty/upload path.
        let dynamic_objects = roots
            .iter()
            .enumerate()
            .filter_map(|(i, root)| {
                let computed = compute.as_ref().and_then(|c| {
                    draws.iter().enumerate().find(|(id, d)| {
                        c.computed_draw(*id) && d.instances.contains(&(i as u32))
                    }).map(|(_, d)| d.instances.start as usize)
                });
                match computed {
                    Some(parent) => (i == parent).then_some(i),
                    None => (nodes[root.node].dynamic || root.imported_pose.is_some()).then_some(i),
                }
            })
            .collect();
        let rays = rays
            .then(|| crate::rt::Rays::new(r, &draws, &instances, &objects, &plan.casters.instances[0], ray_refraction || crate::rt::other_hit_surfaces(&scene) || program_coverage || program_trace, &materials, &render.settings.integrator, &solids.boundaries, &solids.owners, &subsurface_ids, &occupancy_solids, medium_sources))
            .transpose()?;
        // Point shape construction and immutable compute-query snapshots need
        // the retained source topology. Only computed publications keep it after
        // their installation lowerings have acquired the shared shape.
        for draw in &mut draws {
            if !draw.computed && draw.exit_geometry.as_ref().is_some_and(|g|!g.point_centers.is_empty()) {draw.exit_geometry=None;}
        }
        let capture_controls = input.materials.values().any(|m| {
            m.program != "std/openpbr"
                && m.inputs.values().any(|v| {
                    let name = match v {
                        vork_scene::v2::Input::Texture(t) => Some(&t.image),
                        _ => None,
                    };
                    name.is_some_and(|n| input.images.get(n).is_some_and(|i| i.render.is_some()))
                })
        });
        let fibre = if effects.fibre {
            let suns:Vec<_>=light_plan.nodes.iter().map(|&i|matches!(&nodes[i].node.kind,NodeKind::Light {shape:vork_scene::LightShape::Sun,..})).collect();
            let resolution=light_plan.nodes.iter().map(|&i| match &nodes[i].node.kind {
                NodeKind::Light {..}=>crate::constants::FIBRE_CAPTURE_RESOLUTION.max(fibre_view_side),_=>1
            }).max().unwrap_or(1);
            let capture_plan=crate::fibre_transport::Plan::new(&suns,resolution,r.max_texture_allocation,&limits)?;
            Some(crate::fibre_transport::Persistent::new(r,capture_plan,suns.len(),&pipelines.fibre_capture_layout,
                programs.as_ref().map_or(Default::default(), |p|p.dependencies()),plan.deform))
        } else {None};
        let probes = render.settings.integrator.probe_volume().map(|(_volume, bounces)| {
            let dependencies = programs.as_ref().map_or(Default::default(), |p| p.dependencies());
            let timed = dependencies.time || medium.as_ref().is_some_and(|m|m.native.time_dependent) || compute.is_some() || plan.deform || animation.has_material_bindings();
            {
                let mut field=probe_field.context("retained indirect lighting: missing receiver field")?;
                if let Some(rays)=rays.as_ref().filter(|r|r.probe_emitters) {field.header.visibility_policy[3]=rays.emitters;}
                crate::probes::Volume::new(r, &pipelines, field, bounces, timed, plan.exposure_samples, animation.has_shutter_animation(), dependencies, dependencies.time || compute.is_some() || plan.deform || animation.has_material_bindings(), !programs.as_ref().is_some_and(|p|p.has_lit()), features, variants)
            }
        }).transpose()?;
        for draw in &mut draws {
            draw.ray_corners = None;
            if draw.computed {draw.exit_geometry=None;}
        }

        let effect_wait = Instant::now();
        if let Some(compiler) = effects_compiler {
            compiler.join().map_err(|panic| crate::failure::panicked("scene effect pipeline compiler", panic))??;
        }
        pipeline_ms += effect_wait.elapsed().as_secs_f64() * 1000.;
        let sampling = crate::sampling::Parameters::new(&r.device, &scene, &render);
        Ok(Self {
            has_catchers: crate::catcher::has_catchers(&scene) && (allocation_dependencies.beauty || allocation_dependencies.catcher),
            geometry_stages,
            footage: std::mem::take(&mut textures.footage), file_images,
            reservations,
            sampling,
            displacement,
            compositor,
            image_order: planned_image_order,
            main_outputs: output_dependencies,
            shadows: crate::residency::Cache::new(&r.residency),
            compute,
            dense,
            computed_geometry,
            rays,
            animation,
            dynamic_objects,
            pipelines,
            plan,
            graph,
            transients,
            path,
            reflections,
            decoupled,
            software_geometry,
            gi,
            probes,
            fibre,
            geometry_vertex_bytes,
            geometry_pipelines,
            vertex_defaults,
            environment,
            propagation_base: record_words + propagation_offset as u32,
            camera_programs,
            spectral,
            links,
            medium_base,
            environment_base: record_words,
            has_media: medium.is_some(),
            medium_certified_times: medium.as_ref().filter(|m|m.native.tracking).map(|_|medium::certification_times(input)),
            emitter_profiles,
            assets,
            pipeline_ms,
            environment_install_ms,
            checked_shader_capacity: true,
            material_table,
            material_bind_ms,
            asset_decode_ms,
            texture_upload_ms: textures.upload_ms.get(),
            texture_cache_hits: textures.cache_hits.get(),
            render,
            scene,
            nodes,
            targets,
            roots,
            object_ids: std::sync::Arc::new(serde_json::Value::Array(object_ids)),
            camera,
            light_plan,
            draws,
            implicit_fields:implicit_fields.into_values().collect(),
            draw_groups,
            materials,
            objects,
            display_program,
            surface_dependencies: programs.as_ref().map_or(Default::default(), |p|p.program_dependencies()),
            main_subset: None,
            capture_controls,
            readback,
            install_time: Duration::ZERO,
        })
    }
    pub fn render(&self) -> &vork_scene::Render {
        &self.render
    }
    /// The name of the camera that renders at `time` (camera cuts included).
    pub fn camera_name_at_request(&self, time:&vork_time::Request) -> Result<&str> {
        Ok(&self.nodes[self.animation.camera_at_request(time,self.camera)?].node.name)
    }
    pub fn camera_name_at(&self, time: f32) -> &str {
        &self.nodes[self.animation.camera_at(time, self.camera)].node.name
    }
    pub fn scene(&self) -> &Scene {
        &self.scene
    }
    /// Transient bytes all frame slots and views requested, and bytes the
    /// frame graph allocated for them.
    pub fn transient_bytes(&self) -> (u64, u64) {
        self.transients.bytes()
    }
    pub fn depth_unit_m(&self) -> f32 {
        self.nodes[self.camera].camera.as_ref().and_then(|c|c.calibration(self.render.resolution[0] as f32/self.render.resolution[1] as f32).ok().flatten()).map_or(1.,|c|c[1][1]/65535.)
    }
    fn refresh_frame_slot(&self, r: &Renderer, slot: &mut FrameSlot) -> Result<()> {
        ensure!(!slot.pending, "cannot replace a submitted frame slot");
        let precision = if slot.view.float_output { crate::ReadbackPrecision::Float } else { crate::ReadbackPrecision::Packed };
        let mut replacement = self.create_frame_slot(r, precision)?;
        std::mem::swap(&mut replacement.ring, &mut slot.ring);
        std::mem::swap(&mut replacement.spare_ring, &mut slot.spare_ring);
        replacement.ring.rebind_pixels(&replacement.view.post.pixels);
        for ring in &mut replacement.spare_ring { ring.rebind_pixels(&replacement.view.post.pixels); }
        if let Some((output, bind)) = &replacement.ring.output {
            replacement.view.output = output.clone();
            replacement.view.output_bind = bind.clone();
        } else {
            // Library attachments set through set_output are not ring-owned.
            replacement.view.output = slot.view.output.clone();
            replacement.view.output_bind = slot.view.output_bind.clone();
        }
        replacement.profile = slot.profile;
        replacement.write_color = slot.write_color;
        replacement.completed = slot.completed.take();
        *slot = replacement;
        Ok(())
    }

    /// Render one frame, growing frame storage on the same installation if needed.
    pub fn render_frame(&mut self, r: &Renderer, slot: &mut FrameSlot, time: f32) -> Result<crate::FrameOutput> {
        if slot.probe_generation != self.reservations.probe_pages || slot.dielectric_spill_reservation != self.reservations.dielectric_spill {
            self.refresh_frame_slot(r, slot)?;
        }
        loop {
            match slot.submit(self, r, time).and_then(|()| slot.collect(r)) {
                Ok(output) => return Ok(output),
                Err(error) => {
                    let Some(redo) = error.downcast_ref::<crate::FrameRedo>() else { return Err(error); };
                    self.redo_frame(r, std::slice::from_mut(slot), redo.0, (time * self.render.fps.as_f32()).round() as usize)?;
                }
            }
        }
    }

    /// Drain rejected work and rebuild only frame storage, preserving encoder surfaces
    /// and outstanding output leases. The scene and its shared caches stay installed.
    /// Callers must include every frame slot of this install: all are replaced
    /// (Transients::begin_generation); on error they are unusable.
    pub fn redo_frame(
        &mut self, r: &Renderer, slots: &mut [FrameSlot],
        mut demand: crate::Reservations, frame: usize,
    ) -> Result<u32> {
        ensure!(!slots.is_empty(), "redo needs a frame slot");
        let mut growths = 0;

        loop {
            for slot in slots.iter_mut().filter(|slot| slot.pending) {
                // Captures can reuse an earlier exposure's provisional image.
                // Until its producers are accepted, a successor is not independent.
                match slot.collect_for_redo(r, self.graph.views.images.is_empty()) {
                    Ok(()) => {},
                    // A refused successor keeps no completed frame: it is resubmitted
                    // after the redo, and the frame loop handles its refusal then.
                    Err(error) if crate::failure::cause::<crate::FrameRefused>(&error).is_some() => {},
                    Err(error) => match error.downcast_ref::<crate::FrameRedo>() {
                        Some(redo) => {
                            demand=demand.max(redo.0);

                        },
                        None => return Err(error),
                    },
                }
            }
            let grown = self.reservations.max(demand);
            ensure!(grown != self.reservations,
                crate::gpu_memory::DeviceLimit(format!("frame storage cannot grow beyond {:?}: a shading band of camera evaluations, or a merge batch of transparent fragments, needs more records than the device can address",
                    self.reservations)));

            if grown.probe_pages>self.reservations.probe_pages {
                self.probes.as_mut().context("probe demand without installed probes")?.rebuild(r)?;
                // Every shadow reader pins the old probe bindings. Rebuild the
                // shared dependency and invalidate all provisional captures.
                let replacement=self.shadows().rebind_probes(self.probes.as_ref().unwrap().storage());
                let bytes=replacement.bytes();
                self.shadows.insert(replacement,bytes);
            }
            self.reservations = grown;
            growths += 1;
            self.transients.begin_generation();
            for slot in slots.iter_mut() {
                self.refresh_frame_slot(r, slot)?;
            }
            self.transients.release_previous_generations();
            self.transients.submit.lock().unwrap().invalidate_captures();
            let previous_images = self.probes.as_ref().is_some_and(|p| p.dependencies.images)
                || self.fibre.as_ref().is_some_and(|p| p.dependencies.images);
            if !previous_images || self.graph.views.images.is_empty() { return Ok(growths); }
            if frame == 0 {
                let mut encoder = r.device.create_command_encoder(&Default::default());
                for image in self.graph.views.image_renders() { image.clear(&mut encoder); }
                crate::command_finish::Segments::submit(&r.queue, encoder);
                self.transients.submit.lock().unwrap().image_revision += 1;
                return Ok(growths);
            }
            let mut warm_slot = self.create_frame_slot(r, Default::default())?;
            let slot = &mut warm_slot;
            slot.write_color = true;
            // Restore the previous published generation before pre-capture readers.
            // Seek replay also restores temporal history and frame simulations.
            let warm = slot.submit_indexed_outputs(self, r, u32::try_from(frame - 1)?, vork_time::Request::Frame(u32::try_from(frame-1)?,self.render.fps), crate::ReadbackOutputs::default())
                .and_then(|()| slot.collect_readback(r).map(drop));
            match warm {
                Ok(()) => return Ok(growths),
                // The previous frame was refused and skipped; its replay is too.
                Err(error) if crate::failure::cause::<crate::FrameRefused>(&error).is_some() => return Ok(growths),
                Err(error) => match error.downcast_ref::<crate::FrameRedo>() {
                    Some(redo) => {demand=redo.0;},
                    None => return Err(error),
                },
            }
        }
    }

    pub fn create_frame_slot(
        &self,
        r: &Renderer,
        precision: crate::ReadbackPrecision,
    ) -> Result<FrameSlot> {
        let _foreground = r.compiler.foreground_preparation();
        r.errors.check()?;
        let slot = FrameSlot::new(self, r, precision);
        r.errors.check()?;
        slot
    }
}

/// Primitive-major slots let the persistent recorder issue one instanced draw per
/// geometry/material group, even when an instance has several primitives.
fn group_draws(
    nodes: &[EvalNode],
    compute_nodes: &std::collections::HashSet<usize>,
    materials: &[Material],
    draws: &mut Vec<Draw>,
    objects: &mut Vec<Object>,
    roots: &mut Vec<MeshRoot>,
    transparent_instancing: bool,
) -> Vec<std::ops::Range<usize>> {
    let mut group_ids = std::collections::HashMap::new();
    let mut groups: Vec<Vec<Draw>> = Vec::new();
    for draw in draws.drain(..) {
        let node = &nodes[roots[draw.object as usize].node];
        let group = if transparent_instancing || (!materials[draw.material].transparent) {
            let key = (
                (
                    draw.vertices.clone(),
                    draw.attributes.clone(),
                    draw.indices.clone(),
                    draw.vertex_offset,
                    draw.index_offset,
                    draw.count,
                    draw.material,
                    draw.primitive_start,
                ),
                (
                    node.dynamic,
                    compute_nodes
                        .contains(&roots[draw.object as usize].node)
                        .then_some(roots[draw.object as usize].node),
                    materials[draw.material].transparent,
                    // Sorted transparency remaps instances into any draw slot in the
                    // group. Every slot must bind the same exit geometry.
                    draw.exit_bvh.clone(),
                    draw.deform.as_ref().map(|deform| deform.job),
                    draw.displacement.as_ref().map(|_|(roots[draw.object as usize].time_offset.to_bits(), objects[draw.object as usize].metadata[2])),
                ),
            );
            *group_ids.entry(key).or_insert_with(|| {
                groups.push(Vec::new());
                groups.len() - 1
            })
        } else {
            // Preserve transparent tie order and keep per-instance deform outputs private.
            groups.push(Vec::new());
            groups.len() - 1
        };
        groups[group].push(draw);
    }
    let old_objects = std::mem::take(objects);
    let old_roots = std::mem::take(roots);
    let mut draw_groups = Vec::new();
    for group in groups {
        let first_draw = draws.len();
        let first_object = objects.len() as u32;
        let transparent = materials[group[0].material].transparent;
        let mut shared = group[0].clone();
        // Computed streams own object-private solver state. A draw cannot bind
        // one object's evaluated output across another object's instance slot.
        let private = transparent || shared.computed;
        // Preserve authored primitive/instance order even when two primitives
        // share geometry. This also preserves opaque depth ties.
        let mut instances: Vec<_> = group
            .iter()
            .flat_map(|draw| {
                draw.instances
                    .clone()
                    .enumerate()
                    .map(move |(i, object)| (draw.sort_order + i * draw.sort_stride, draw, object))
            })
            .collect();
        instances.sort_by_key(|&(order, _, _)| order);
        for (order, draw, original) in instances {
            let object = objects.len() as u32;
            objects.push(old_objects[original as usize]);
            roots.push(old_roots[original as usize].clone());
            if private {
                let mut instance = draw.clone();
                instance.object = object;
                instance.instances = object..object + 1;
                instance.sort_order = order;
                draws.push(instance);
            }
        }
        if !private {
            shared.object = first_object;
            shared.instances = first_object..objects.len() as u32;
            draws.push(shared);
        }
        let range = first_draw..draws.len();
        draw_groups.extend(std::iter::repeat_n(range.clone(), range.len()));
    }
    draw_groups
}

#[cfg(all(test, vork_private_tests))]
include!("install_material_binding_tests_private.rs");
