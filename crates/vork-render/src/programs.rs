//! Slang material programs (docs/PROGRAMS.md). Each distinct program compiles
//! once through the shared disk cache, retaining native modules for raster stages.
use crate::{Renderer, disk_cache};
use anyhow::{Context, Result, ensure};
use std::{
    collections::HashMap,
    fmt::Write as _,
    path::{Path, PathBuf},
};
use vork_scene::v2;
use vork_slang::{Abi, Compiler, Program};

pub(crate) fn program_source(
    base: &Path,
    name: &str,
    program: &v2::Program,
) -> Result<(String, PathBuf)> {
    // Authored origins must mean the same thing after the compiler enters its
    // workspace, and under relative/absolute spellings of the scene path.
    let base: PathBuf = std::env::current_dir()?.join(base).components().collect();
    match (&program.file, &program.source) {
        (Some(file), None) => {
            let path = base.join(file);
            Ok((
                std::fs::read_to_string(&path)
                    .with_context(|| format!("program {name}: read {}", path.display()))?,
                path,
            ))
        }
        (None, Some(source)) => Ok((source.clone(), base.join(program.source_origin.as_deref().map(PathBuf::from).unwrap_or_else(||PathBuf::from(format!("programs.{name}")))))),
        _ => anyhow::bail!("program {name}: exactly one of file/source required"),
    }
}
// Bundled and custom materials keep the same closure semantics across integrators.
pub(crate) fn material_source(scene: &v2::Scene, base: &Path, name: &str) -> Result<(String, PathBuf)> {
    if let Some(source) = vork_slang::module_source(name) {
        // Bundled modules own their imports. A scene's directory must not
        // multiply their reflection/native tags or invalidate setup's artifacts.
        Ok((source.to_owned(), PathBuf::from(format!("{name}.slang"))))
    } else {
        let program=&scene.programs[name];
        let (source,origin)=program_source(base,name,program)?;
        // Slang checks unused entry bodies too. A matching compute declaration of
        // this authored module supplies its API, but no implementation that a
        // surface could accidentally execute.
        let compute=if program.is_compute() {Some(program)} else {
            scene.programs.values().find(|candidate|candidate.is_compute()
                && ((program.file.is_some() && candidate.file==program.file)
                    || (program.source.is_some() && candidate.source==program.source)))
        };
        if let Some(compute)=compute {
            let declarations=crate::compute::stage_declarations(scene,compute)?;
            Ok((format!("{declarations}\n#line 1\n{source}"),origin))
        } else {Ok((source,origin))}
    }
}

/// ShadingContext as far as shading provides it, and every OpenPBR Closure
/// field (the table holds only fields this renderer shades). Programs also set
/// the coat's shading normal: every surface stage shades `geometry_coat_normal`
/// (unset, it stays the unperturbed normal); std/openpbr has no input for it.
pub(crate) fn abi() -> Abi {
    Abi { aov_count: 0, context: Abi::spec_context(), closure: vork_slang::openpbr_fields() }
}

/// A material's program instance: module source (inline or a file relative to
/// the scene) and its supplied inputs, checked against reflection at compile.
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Instance {
    pub name: String,
    pub file: Option<PathBuf>,
    pub source: Option<String>,
    pub inputs: std::collections::BTreeMap<String, serde_json::Value>,
    /// Image inputs, resolved from the scene's image table.
    pub images: Vec<Image>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Image {
    /// The scene image's name, for diagnostics only: two names for one file are one resource.
    #[serde(skip)]
    pub image: String,
    pub input: String,
    pub source: crate::texture_source::TextureSource,
    pub srgb: bool,
    pub color_space: Option<v2::ColorSpace>,
    pub sampler: v2::Sampler,
}
/// The program instance of a non-OpenPBR material; None for std/openpbr, whose
/// records are installed natively (materials::install_openpbr).
pub(crate) fn instance(scene: &v2::Scene, name: &str) -> Result<Option<Instance>> {
    let material = scene
        .materials
        .get(name)
        .with_context(|| format!("unknown material {name}"))?;
    if material.program == "std/openpbr" {
        return Ok(None);
    }
    let program = scene.programs.get(&material.program);
    ensure!(
        program.is_some() || vork_slang::module_source(&material.program).is_some(),
        "material {name}: unknown program {}",
        material.program
    );
    let mut images = Vec::new();
    for (input_name, value) in &material.inputs {
        let image_name = match value {
            v2::Input::Texture(binding) => &binding.image,
            _ => continue,
        };
        let image = scene
            .images
            .get(image_name)
            .with_context(|| {
                format!(
                    "material {name}: input {input_name}: {}",
                    v2::diagnostics::unknown("image", image_name, scene.images.keys().map(String::as_str))
                )
            })?;
        images.push(Image {
            image: image_name.to_string(),
            input: input_name.clone(),
            source: crate::texture_source::TextureSource::scene(image_name, image),
            srgb: image.is_srgb(),
            color_space: image.color_space,
            sampler: image.sampler(),
        });
    }
    Ok(Some(Instance {
        name: material.program.clone(),
        file: program.and_then(|p| p.file.as_ref()).map(Into::into),
        source: program.and_then(|p| p.source.clone()),
        inputs: material
            .inputs
            .iter()
            .map(|(name, value)| Ok((name.clone(), serde_json::to_value(value)?)))
            .collect::<Result<_>>()?,
        images,
    }))
}

pub(crate) struct Programs {
    /// The same programs as Slang modules for the native colour passes; None
    /// when the scene has only named attributes.
    pub(crate) native: Option<std::sync::Arc<Native>>,
    compiled: Vec<Program>,
    /// Capabilities proved from each instance’s material words.
    bound: HashMap<String, vork_slang::ProgramCapabilities>,
    /// Per instance of a program that may produce coverage: whether that
    /// coverage can change between frames (EffectProbe::coverage_varies).
    coverage_varies: HashMap<String, bool>,
    physical_hair: bool,
    aovs: std::collections::BTreeMap<String,u32>,
    /// Compiled program index per instance.
    instances: HashMap<String, usize>,
}
impl Programs {
    pub(crate) fn input_layout(&self, id: u32) -> &vork_slang::InputLayout {
        &self.compiled[id as usize - 1].inputs
    }
    pub(crate) fn dependencies(&self) -> vork_slang::ProgramDependencies {
        self.compiled.iter().fold(vork_slang::ProgramDependencies::default(), |mut d, p| {
            d.include(p.capabilities.dependencies);
            d
        })
    }

    pub(crate) fn program_dependencies(&self) -> Vec<vork_slang::ProgramDependencies> {
        self.compiled.iter().map(|p| p.capabilities.dependencies).collect()
    }

    pub(crate) fn has_lit(&self)->bool {self.compiled.iter().any(|p|p.capabilities.lit)}
    pub(crate) fn has_displacement(&self)->bool {self.compiled.iter().any(|p|p.capabilities.displacement)}
    pub(crate) fn displacement_timed(&self,id:u32)->bool {self.compiled[id as usize-1].displacement_uses_time}
    pub(crate) fn displacement_uses_normal(&self,id:u32)->bool {self.compiled[id as usize-1].displacement_uses_normal}
    pub(crate) fn displacement_descriptors(&self, id:u32, base:usize, words:&[u32])->Vec<(u32,Vec<u32>)> {
        let program=&self.compiled[id as usize-1];
        program.displacement_attributes.iter().map(|name| {
            let field=program.inputs.fields.iter().find(|f|&f.name==name).unwrap();
            let vork_slang::Type::Attribute(width)=field.ty else {unreachable!()};
            let descriptor=base+words[base+field.offset] as usize;
            (width,words[descriptor..descriptor+5].to_vec())
        }).collect()
    }
    pub(crate) fn displaces(&self, id: u32) -> bool { id.checked_sub(1).and_then(|i| self.compiled.get(i as usize)).is_some_and(|p| p.capabilities.displacement) }
    pub(crate) fn attribute_width(&self,p:&Instance,input:&str)->Option<u32> {
        self.compiled[self.instances[&instance_key(p)]].inputs.fields.iter().find_map(|f|if f.name==input {if let vork_slang::Type::Attribute(n)=f.ty {Some(n)} else {None}} else {None})
    }
    /// Whether the instance's program reads `time` (surface, lit or displacement).
    pub(crate) fn reads_time(&self, p: &Instance) -> bool {
        self.instances.get(&instance_key(p)).is_some_and(|&index| self.compiled[index].capabilities.dependencies.time)
    }
    pub(crate) fn is_hair(&self, p: &Instance) -> bool { self.compiled[self.instances[&instance_key(p)]].capabilities.hair }
    pub(crate) fn has_physical_hair(&self) -> bool { self.physical_hair }
    pub(crate) fn is_physical_hair(&self, p: &Instance) -> bool { self.bound.get(&instance_key(p)).is_some_and(|c| c.physical_hair) }
    /// Admission uses the same bound proof as launch normalization. Keep the
    /// program-wide capability unchanged for shader generation.
    pub(crate) fn is_subsurface(&self, p: &Instance) -> bool {
        let key = instance_key(p);
        self.bound.get(&key).map_or(self.compiled[self.instances[&key]].capabilities.subsurface, |c| c.subsurface)
    }
    /// The instance's coverage can change between frames on an object that
    /// does not move: never without program opacity; unproven is true.
    pub(crate) fn coverage_varies(&self, p: &Instance) -> bool {
        let key = instance_key(p);
        self.compiled[self.instances[&key]].capabilities.opacity && self.coverage_varies.get(&key).copied().unwrap_or(true)
    }
    /// The program's capabilities, with opacity and transmission narrowed to
    /// what this instance's material words can produce. Solid scattering also
    /// narrows to the instance's thin-walled state; coverage picks the
    /// material's route. Other effects keep the program's value.
    pub(crate) fn capabilities(&self, p: &Instance) -> vork_slang::ProgramCapabilities {
        let key = instance_key(p);
        let mut capabilities = self.compiled[self.instances[&key]].capabilities;
        if let Some(bound) = self.bound.get(&key) {
            capabilities.opacity &= bound.opacity;
            capabilities.transmission &= bound.transmission;
            capabilities.spatial_subsurface &= bound.spatial_subsurface;
            capabilities.emission &= bound.emission;
            capabilities.anisotropy &= bound.anisotropy;
            capabilities.coat_anisotropy &= bound.coat_anisotropy;
            capabilities.film &= bound.film;
            capabilities.coat &= bound.coat;
            capabilities.fuzz &= bound.fuzz;
            capabilities.tint &= bound.tint;
        }
        capabilities
    }
    /// Program id (1-based dispatch index) and an instance's input words; image
    /// inputs pack the scene material table word of their image.
    pub(crate) fn pack(
        &self,
        p: &Instance,
        image_words: &std::collections::BTreeMap<String, u64>,
        scene: &v2::Scene,
        base: &Path,
    ) -> Result<(u32, Vec<u32>)> {
        let index = self.instances[&instance_key(p)];
        let mut words = pack_inputs(&self.compiled[index].inputs, &p.inputs, scene, base, |name, value| {
                if self.compiled[index].inputs.fields.iter().any(|f| f.name == name && f.ty == vork_slang::Type::Aov) {
                    let name=value.get("aov").and_then(serde_json::Value::as_str).with_context(|| format!("input {name}: Aov inputs take {{\"aov\": name}} naming a render.aovs entry, not {value}"))?;
                    return self.aovs.get(name).copied().with_context(|| format!("input {name}: {}", v2::diagnostics::unknown("render.aovs entry", name, self.aovs.keys().map(String::as_str))));
                }
                image_words
                    .get(name)
                    .map(|&word|word as u32)
                    .with_context(|| format!("input {name}: Image inputs take {{\"image\": name}}, not {value}"))
            })
            .with_context(|| format!("program {}", p.name))?;
        for field in &self.compiled[index].inputs.fields {
            if field.ty == vork_slang::Type::Image {
                let Some(&descriptor)=image_words.get(&field.name) else {continue;};
                words[field.offset]=u32::try_from(words.len())?;
                words.extend([descriptor as u32,(descriptor>>32) as u32]);
            }
        }
        Ok((index as u32 + 1, words))
    }
}
/// All stages of a material use the same numeric/grid input layout and relocation.
/// Each stage supplies the resource kinds it supports beyond named grids.
pub(crate) fn pack_inputs(
    layout: &vork_slang::InputLayout,
    values: &std::collections::BTreeMap<String, serde_json::Value>,
    scene: &v2::Scene,
    base: &Path,
    mut resource: impl FnMut(&str, &serde_json::Value) -> Result<u32>,
) -> Result<Vec<u32>> {
    let mut grids = Vec::new();
    let mut words = layout.pack(values, |name, value| {
        let field = layout.fields.iter().find(|f| f.name == name).context("program resource field")?;
        let expected = match field.ty {
            vork_slang::Type::FloatGrid => v2::BufferType::F32,
            vork_slang::Type::VectorGrid => v2::BufferType::Vec3,
            vork_slang::Type::Geometry => {
                let binding:v2::GeometryBinding=serde_json::from_value(value.clone())?;
                return crate::geometry_reference::handle(scene,&binding.geometry);
            }
            vork_slang::Type::Attribute(width) => {
                let binding: v2::AttributeBinding=serde_json::from_value(value.clone()).with_context(||format!("input {name}: Attribute inputs require {{attribute, channel?, factor?}}"))?;
                let names=crate::attributes::names(scene)?;
                let slot=names.iter().position(|n|n==&binding.attribute).context("attribute binding not installed")?;
                let channel=crate::attributes::channel(&binding,width)?;
                let factors=crate::attributes::factor(&binding,width)?;
                let offset=u32::try_from(layout.words+grids.len())?;
                grids.push(slot as u32 | channel<<8);
                grids.extend(factors.map(f32::to_bits));
                return Ok(offset);
            }
            _ => return resource(name, value),
        };
        let buffer_name = value.get("grid").and_then(serde_json::Value::as_str)
            .with_context(|| format!("input {name}: grid inputs take {{\"grid\": buffer-name}}, not {value}"))?;
        let buffer = scene.buffers.get(buffer_name).with_context(|| format!("unknown grid buffer {buffer_name}"))?;
        ensure!(buffer.element_type == expected, "grid input {name}: voxel type mismatch");
        if let Some(dims) = buffer.dims {
            let data = crate::buffers::AttributeData::decode(buffer, base)?;
            let offset = u32::try_from(layout.words + grids.len())?;
            // Dense descriptor: sentinel, dimensions, then cell-centred words.
            let (lo,hi) = data.words.iter().map(|&w| f32::from_bits(w)).fold((0f32,0f32), |(lo,hi),v| (lo.min(v),hi.max(v)));
            grids.extend([u32::MAX, dims[0], dims[1], dims[2], lo.to_bits(), hi.to_bits()]);
            grids.extend(data.words);
            return Ok(offset);
        }
        let grid = crate::vdb::Grid::load(buffer, base)?;
        let offset = u32::try_from(layout.words + grids.len())?;
        grids.push(u32::try_from(grid.words.len())?);
        grids.extend(grid.words);
        Ok(offset)
    })?;
    for field in &layout.fields {
        if matches!(field.ty, vork_slang::Type::FloatGrid | vork_slang::Type::VectorGrid) {
            ensure!(values.contains_key(&field.name), "grid input {} must name a buffer", field.name);
        }
    }
    words.extend(grids);
    Ok(words)
}

fn instance_key(p: &Instance) -> String {
    serde_json::to_string(p).expect("program instance serializes")
}

/// Resolve surface/geometry combinations before compiling their GPU consumers.
/// Copies are per node, so a shared material keeps its supported uses unchanged.
pub(crate) fn normalize_launch_bindings(r:&Renderer, scene:&mut v2::Scene, base:&Path, stages:&std::collections::BTreeMap<String,Stages>)->Result<()> {
    let mut replacements=std::collections::BTreeMap::<String,(v2::Material,String)>::new();
    let mut nodes=scene.nodes.clone();
    fn computed_placements(scene:&v2::Scene,nodes:&[v2::Node])->bool {
        nodes.iter().any(|node|crate::compute::places_instances(scene,node) || computed_placements(scene,&node.children))
    }
    // Fibre captures fit the whole scene, including ordinary computed blockers
    // on other nodes. A same-node test misses those cross-node dependencies.
    let computed=computed_placements(scene,&scene.nodes);
    let compiler=Compiler::discover();
    let mut abi=abi(); abi.aov_count=scene.render.aovs.len();
    // Keep complete proofs, including failures, until their original traversal
    // position. Compilation is parallel; policy, warnings and error precedence
    // remain serial. Four workers cap concurrent frontend memory on large hosts.
    let names = launch_analysis_programs(scene, stages, computed);
    let analyzed = parallel_map_with_limit(&names, 4, |name| {
        Ok(material_source(scene, base, name).and_then(|(source, origin)| {
            compile_quiet(r, &compiler, &abi, &source, &origin).map(|program| (program, origin))
        }))
    })?;
    let mut analyses = names.into_iter().zip(analyzed).collect::<std::collections::BTreeMap<_, _>>();
    let mut proof=std::collections::BTreeMap::new();
    fn visit(scene:&v2::Scene,nodes:&mut [v2::Node],
        analyses:&mut std::collections::BTreeMap<String,Result<(Program,PathBuf)>>,
        stages:&std::collections::BTreeMap<String,Stages>,proof:&mut std::collections::BTreeMap<String,vork_slang::ProgramCapabilities>,replacements:&mut std::collections::BTreeMap<String,(v2::Material,String)>,computed:bool)->Result<()> {
        for node in nodes {
            if let v2::NodeKind::Light {shape:v2::LightShape::Environment,material:Some(name),..}=&node.kind {
                if scene.materials[name].inputs.values().any(|v|matches!(v,v2::Input::Texture(binding) if scene.images[&binding.image].render.is_some())) {
                    crate::warnings::dropped("environment_render_image", serde_json::json!({"node":node,"material":scene.materials[name]}), format!("warning: node {:?}: ignored environment material {name} sampling a render image; using the light's colour",node.name));
                    if let v2::NodeKind::Light {material,..}=&mut node.kind {*material=None;}
                }
            }
            let (bindings,curves,implicit):(Vec<&mut String>,bool,bool)=match &mut node.kind {
                v2::NodeKind::Mesh {materials,geometry,..}=>(materials.iter_mut().collect(),matches!(scene.geometry[geometry].topology,v2::Topology::Curves),matches!(scene.geometry[geometry].topology,v2::Topology::Implicit)),
                v2::NodeKind::Asset {materials,..}=>(materials.values_mut().collect(),false,false),
                _=>(Vec::new(),false,false),
            };
            for binding in bindings {
                // Bound media were refused by admission. Unused medium remaps
                // have no surface to approximate and keep their authored data.
                if stages.get(binding.as_str()).is_some_and(|stage|stage.medium) {continue;}
                let material=&scene.materials[binding.as_str()];
                let mut replacement=None;
                if material.program=="std/openpbr" {
                    if implicit && (material.inputs.get("subsurface_weight").is_some_and(|v|!matches!(v,v2::Input::Number(x) if *x==0.)) || scene.animation.iter().flat_map(|l|l.tracks.iter().flatten()).any(|t|t.path==["materials",binding.as_str(),"inputs","subsurface_weight"])) {
                        let mut m=material.clone();m.inputs.insert("subsurface_weight".into(),v2::Input::Number(0.));
                        crate::warnings::dropped("implicit_subsurface", serde_json::json!({"node":node.name,"material":binding,"inputs":material.inputs,"tracks":scene.animation.iter().flat_map(|layer|layer.tracks.iter().flatten()).filter(|track|matches!(track.path.as_slice(),[root,target,..] if root=="materials" && target.as_str()==binding.as_str())).collect::<Vec<_>>()}), format!("warning: node {:?}, material {}: ignored subsurface scattering on implicit geometry",node.name,binding));
                        replacement=Some(m);
                    }
                } else if !curves || implicit || computed {
                    let capability=if let Some(c)=proof.get(binding.as_str()) {*c} else {
                        let (program, origin) = analyses.remove(&material.program)
                            .expect("launch analysis collected every requested surface")?;
                        emit_dropped_stages(&program, &origin);
                        let dynamic=scene.animation.iter().flat_map(|l|l.tracks.iter().flatten()).filter_map(|t|match t.path.as_slice(){[root,name,inputs,field] if root=="materials" && name.as_str()==binding.as_str() && inputs=="inputs"=>Some(field.clone()),_=>None}).collect();
                        let values=material.inputs.iter().map(|(k,v)|Ok((k.clone(),serde_json::to_value(v)?))).collect::<Result<std::collections::BTreeMap<_,_>>>()?;
                        let words=program.inputs.effect_words(&values,&dynamic)?;
                        let c=program.effect_probe()?.capabilities(&words);
                        analyses.insert(material.program.clone(), Ok((program, origin)));
                        proof.insert(binding.clone(),c);c
                    };
                    if (capability.hair && !curves) || (implicit && capability.subsurface) {
                        crate::warnings::dropped("surface_geometry", serde_json::json!({"node":node.name,"material":binding,"inputs":material.inputs,"program":material.program,"tracks":scene.animation.iter().flat_map(|layer|layer.tracks.iter().flatten()).filter(|track|matches!(track.path.as_slice(),[root,target,..] if root=="materials" && target.as_str()==binding.as_str())).collect::<Vec<_>>()}), format!("warning: node {:?}, material {}: ignored unsupported {} surface on this geometry; using default OpenPBR surface",node.name,binding,if capability.hair {"Hair"} else {"custom subsurface"}));
                        replacement=Some(serde_json::from_value(serde_json::json!({}))?);
                    }
                }
                if let Some(m)=replacement {
                    let name=format!("__vork_launch_{}",replacements.len());replacements.insert(name.clone(),(m,binding.clone()));*binding=name;
                }
            }
            visit(scene,&mut node.children,analyses,stages,proof,replacements,computed)?;
        }
        Ok(())
    }
    visit(scene,&mut nodes,&mut analyses,stages,&mut proof,&mut replacements,computed)?;
    scene.nodes=nodes;
    for layer in &mut scene.animation {
        if let Some(tracks)=&mut layer.tracks {
            let mut copies=Vec::new();
            for (name,(material,original)) in &replacements {
                if material.program=="std/openpbr" && scene.materials[original].program=="std/openpbr" {
                    for track in tracks.iter().filter(|t|matches!(t.path.as_slice(),[root,target,inputs,field] if root=="materials" && target==original && inputs=="inputs" && field!="subsurface_weight")) {
                        let mut copy=track.clone();copy.path[1]=name.clone();copies.push(copy);
                    }
                }
            }
            tracks.extend(copies);
        }
    }
    scene.materials.extend(replacements.into_iter().map(|(name,(material,_))|(name,material)));
    Ok(())
}

/// Only request proofs that launch normalization will consume. In particular,
/// unused programs, media and supported static curve surfaces stay untouched.
fn launch_analysis_programs(scene: &v2::Scene, stages: &std::collections::BTreeMap<String, Stages>, computed: bool) -> Vec<String> {
    fn visit(scene: &v2::Scene, nodes: &[v2::Node], stages: &std::collections::BTreeMap<String, Stages>, computed: bool,
        names: &mut std::collections::BTreeSet<String>) {
        for node in nodes {
            let bindings: Vec<&String> = match &node.kind {
                v2::NodeKind::Mesh { materials, geometry, .. }
                    if !matches!(scene.geometry[geometry].topology, v2::Topology::Curves) || computed => materials.iter().collect(),
                v2::NodeKind::Asset { materials, .. } => materials.values().collect(),
                _ => Vec::new(),
            };
            for binding in bindings {
                let material = &scene.materials[binding];
                if material.program != "std/openpbr" && !stages.get(binding).is_some_and(|s| s.medium) {
                    names.insert(material.program.clone());
                }
            }
            visit(scene, &node.children, stages, computed, names);
        }
    }
    let mut names = std::collections::BTreeSet::new();
    visit(scene, &scene.nodes, stages, computed, &mut names);
    names.into_iter().collect()
}

/// Independently routed stages of materials actually bound to meshes/assets.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Stages {
    pub surface: bool,
    pub medium: bool,
    pub displacement: bool,
}
pub(crate) fn mesh_stages(r: &Renderer, scene: &v2::Scene, base: &Path) -> Result<std::collections::BTreeMap<String, Stages>> {
    source_mesh_stages(&r.disk_cache, scene, base)
}

fn mesh_material_names(scene: &v2::Scene) -> std::collections::BTreeSet<String> {
    fn visit(nodes: &[v2::Node], names: &mut std::collections::BTreeSet<String>) {
        for node in nodes {
            match &node.kind {
                v2::NodeKind::Mesh { materials, .. } => names.extend(materials.iter().cloned()),
                v2::NodeKind::Asset { materials, .. } => names.extend(materials.values().cloned()),
                _ => {}
            }
            visit(&node.children, names);
        }
    }
    let mut names = std::collections::BTreeSet::new();
    visit(&scene.nodes, &mut names);
    names
}

/// A negative source test, not stage discovery: only the compiler supplies stage
/// facts. Without the name, preprocessing or external modules, a medium entry
/// cannot exist. Keep comments/strings intact: false positives cost a probe but
/// can never admit fog. The generated ABI and registered library-only modules
/// are written into every compiler workspace and export no material stages.
fn may_declare_medium(source: &str) -> bool {
    if ["medium", "#", "\\", "include", "implementing", "__"].iter().any(|text| source.contains(text)) {
        return true;
    }
    source.match_indices("import").any(|(offset, word)| {
        let Some((module, _)) = source[offset + word.len()..].split_once(';') else { return true; };
        let module = module.trim();
        module != "vork" && !vork_slang::library_modules()
            .any(|known| !known.material && module == known.name.replace('/', "."))
    })
}

#[cfg(test)]
mod fog_admission_tests {
    use super::*;

    #[test]
    fn negative_source_test_keeps_module_loading_and_preprocessing_uncertain() {
        for source in ["", "import vork;", "import \n vork ; import std.texture;",
            "Closure surface() {} // colour edit"] {
            assert!(!may_declare_medium(source), "{source}");
        }
        for source in ["MediumCoefficients medium() {}", "// medium", "#define entry surface",
            "import fog;", "import std.hair;", "import vork_extra;", "import vork.extra;",
            "import /* comment */ vork;", "import \"fog.slang\";", "__include \"fog.slang\";",
            "implementing fog;", "im\\\nport fog;", "#include \"fog.slang\""] {
            assert!(may_declare_medium(source), "{source}");
        }
    }

    #[test]
    fn surface_library_imports_cannot_supply_medium() -> Result<()> {
        // Exercise surface helpers and their transitive exports with Slang.
        // Compute-library include fragments require their generator's constants
        // and are not standalone imports for this mesh-program ABI.
        let compiler = Compiler::discover();
        let root = std::env::temp_dir().join(format!("vork-library-admission-{}", std::process::id()));
        std::fs::create_dir_all(root.join("std"))?;
        // Authored files must not shadow the compiler-owned imports trusted by
        // the negative test, even when they declare a medium entry.
        let shadow = "import vork; struct Inputs {}; MediumCoefficients medium(MediumContext c, Inputs i) { return MediumCoefficients(); }";
        std::fs::write(root.join("vork.slang"), shadow)?;
        std::fs::write(root.join("std/texture.slang"), shadow)?;
        for module in ["vork", "std.texture", "blender.nodes", "blender.volume"] {
            let (stages, _) = compiler.stages(&abi(), &format!("import {module};"), &root.join("surface.slang"))?;
            assert!(!stages.contains("medium"), "library-only module {module} exports medium");
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}

/// Fog's pre-device refusal needs AST facts only when a bound source can supply
/// a medium, or an environment explicitly binds one. Ordinary surface editing
/// must not prepare compiler inputs or open/trim the disk store during parsing.
pub(crate) fn needs_fog_admission(scene: &v2::Scene, base: &Path) -> Result<bool> {
    let names = mesh_material_names(scene);
    let programs: std::collections::BTreeSet<_> = names.iter()
        .map(|name| scene.materials[name].program.as_str()).filter(|name| *name != "std/openpbr").collect();
    if programs.is_empty() { return Ok(false); }
    fn world_medium(nodes: &[v2::Node]) -> bool {
        nodes.iter().any(|node| matches!(&node.kind,
            v2::NodeKind::Light { shape: v2::LightShape::Environment, medium: Some(_), .. })
            || world_medium(&node.children))
    }
    if world_medium(&scene.nodes) { return Ok(true); }
    for program in programs {
        let (source, _) = material_source(scene, base, program)?;
        if may_declare_medium(&source) { return Ok(true); }
    }
    Ok(false)
}

/// AST stage facts are host work and share installation's portable cache.
pub(crate) fn source_mesh_stages(store: &disk_cache::Store, scene: &v2::Scene, base: &Path) -> Result<std::collections::BTreeMap<String, Stages>> {
    let names = mesh_material_names(scene);
    let builtin=Stages {surface:true,medium:false,displacement:false};
    let mut programs=std::collections::BTreeMap::new();
    for name in &names {
        let program=scene.materials[name].program.as_str();
        if program!="std/openpbr" {programs.entry(program).or_insert(name);}
    }
    let stages=if programs.is_empty() {std::collections::BTreeMap::new()} else {
        let compiler=Compiler::discover();
        let mut abi=abi();
        abi.aov_count=scene.render.aovs.len();
        // Stage discovery shares the same source/import identity as reflection.
        // Deduplicate before launching workers so aliases cannot race cache misses.
        let mut groups=std::collections::BTreeMap::<(String,PathBuf),Vec<String>>::new();
        for (program,_) in programs {
            let (source,origin)=material_source(scene,base,program)?;
            groups.entry((source,origin.to_path_buf())).or_default().push(program.to_owned());
        }
        let programs:Vec<_>=groups.into_iter().collect();
        let probe=|((source,origin),names):&((String,PathBuf),Vec<String>)| -> Result<Vec<(String,Stages)>> {
            let identity=program_identity(store,&compiler,&abi,source,origin)?;
            let stages=cached_proof(store, input_key(b"mesh-stages-v5", &abi, &identity.identity)?.finalize(), &compiler.with_identity(&identity), &abi, source, origin, |compiler| {
                let (declarations, _) = compiler.stages(&abi, source, origin)?;
                let stages = Stages {
                    surface: declarations.contains("surface"),
                    medium: declarations.contains("medium"),
                    displacement: declarations.contains("displace"),
                };
                ensure!(stages.surface || stages.medium, "program {}: mesh program must export surface or medium; lit() and displace() supplement surface(), they do not replace it; see vork help programs --all",names[0]);
                Ok(stages)
            })?;
            Ok(names.iter().map(|name|(name.clone(),stages)).collect())
        };
        parallel_map(&programs, probe)?.into_iter().flatten().collect::<std::collections::BTreeMap<_,_>>()
    };
    Ok(names.into_iter().map(|name| {
        let program=&scene.materials[&name].program;
        let stage=if program=="std/openpbr" {builtin} else {stages[program]};
        (name,stage)
    }).collect())
}

/// Whether any authored material is a program instance. Used only to avoid
/// speculatively compiling the built-in registry before install filters roots.
pub(crate) fn has_programs(scene: &crate::Scene) -> bool {
    scene.materials.values().any(|m| {
        m.program != "std/openpbr"
            || m.inputs
                .values()
                .any(|v| matches!(v, v2::Input::Attribute(_)))
    })
}

/// Compile every distinct program among `materials` (material name and its
/// program instance); None when there are none.
pub(crate) fn install<'a>(
    r: &Renderer,
    scene: &v2::Scene,
    materials: impl Iterator<Item = (&'a str, &'a Instance)>,
    base: &Path,
) -> Result<Option<Programs>> {
    let mut abi = abi();
    abi.aov_count = scene.render.aovs.len();
    let named: Vec<_> = materials.collect();
    let materials: Vec<_> = named.iter().map(|(_, p)| *p).collect();
    let mut sources = Vec::<(String, PathBuf)>::new();
    let mut ids = HashMap::new();
    for p in &materials {
        let key = instance_key(p);
        if ids.contains_key(&key) {
            continue;
        }
        let (source, origin) = if !scene.programs.contains_key(&p.name) && vork_slang::module_source(&p.name).is_none() { (p.source.clone().context("imported program source")?,base.join("imported.slang")) } else { material_source(scene, base, &p.name)? };
        let index = match sources
            .iter()
            .position(|(s, o)| *s == source && o.parent() == origin.parent())
        {
            Some(i) => i,
            None => {
                sources.push((source, origin));
                sources.len() - 1
            }
        };
        ids.insert(key, index);
    }
    if sources.is_empty() && crate::attributes::names(scene)?.is_empty() {
        return Ok(None);
    }
    let compiler = Compiler::discover();
    let compiled: Vec<Program> = parallel_map(&sources, |(source, origin)| {
        compile(r, &compiler, &abi, source, origin).map_err(|error| {
            if error.downcast_ref::<vork_slang::CompilerUnavailable>().is_some() { return error; }
            let users = named.iter().filter(|(_, p)| p.source.as_ref() == Some(source) || p.file.as_ref().is_some_and(|file| base.join(file) == *origin))
                .map(|(name, p)| format!("material {name}, program {}", p.name)).collect::<Vec<_>>();
            scene.authored_error(error.context(users.join("; ")))
        })
    })?;
    // Raw grouping above only avoids duplicate work. Program routing must use
    // compiler identity too: commenting one of two identical aliases must not
    // create another program bin or grow the scene dispatcher.
    let identities=compiled.iter().map(|p|p.install_identity.as_ref().map(|i|i.identity.as_slice()).context("material input identity")).collect::<Result<Vec<_>>>()?;
    let remap=semantic_indices(identities.into_iter());
    let mut unique=Vec::new();
    let mut unique_sources=Vec::new();
    for (index,(program,source)) in compiled.into_iter().zip(sources).enumerate() {
        if remap[index]==unique.len() {unique.push(program);unique_sources.push(source);}
    }
    for index in ids.values_mut() {*index=remap[*index];}
    let mut compiled=unique;
    let sources=unique_sources;
    for (program,(_,origin)) in compiled.iter().zip(&sources) {
        if !program.warnings.trim().is_empty() {eprintln!("{}: {}",origin.display(),program.warnings.trim());}
    }
    let mut dynamic_inputs = HashMap::<String, std::collections::BTreeSet<String>>::new();
    for track in scene.animation.iter().flat_map(|layer| layer.tracks.iter().flatten()) {
        if let [root, name, inputs, field] = track.path.as_slice() {
            if root == "materials" && inputs == "inputs" {
                if let Some(p) = instance(scene, name)? {
                    dynamic_inputs.entry(instance_key(&p)).or_default().insert(field.clone());
                }
            }
        }
    }
    // Prove effects and temporal coverage from each instance, independently of
    // opacity semantics. Fully opaque results need no coverage integration.
    let mut probed = std::collections::BTreeMap::<String, (&str, &Instance)>::new();
    for &(name, p) in &named {
        let c = compiled[ids[&instance_key(p)]].capabilities;
        let response=c.anisotropy || c.film || c.coat || c.fuzz || c.tint;
        if response || c.emission || c.hair || c.transmission || c.subsurface || c.spatial_subsurface || c.opacity {
            probed.entry(instance_key(p)).or_insert((name,p));
        }
    }
    let mut groups = std::collections::BTreeMap::<usize, Vec<(String, &str, &Instance)>>::new();
    for (instance, (name, p)) in probed { groups.entry(ids[&instance]).or_default().push((instance, name, p)); }
    let groups: Vec<_> = groups.into_iter().collect();
    let probed = parallel_map(&groups, |(index, instances)| {
        let program = &compiled[*index];
        let probe = program.effect_probe()?;
        instances.iter().map(|(instance, name, p)| {
            let dynamic = dynamic_inputs.get(instance).cloned().unwrap_or_default();
            let words = program.inputs.effect_words(&p.inputs, &dynamic)
                .with_context(|| format!("materials.{name}.inputs: program {}: closure effects", p.name))?;
            let live_texels = p.images.iter().any(|i| i.source.changes_between_frames());
            let varies = !dynamic.is_empty() || probe.coverage_varies(&words, live_texels);
            Ok((instance.clone(), probe.capabilities(&words), varies))
        }).collect::<Result<Vec<_>>>()
    })?;
    let mut bound = HashMap::new();
    let mut coverage_varies = HashMap::new();
    for (instance, capabilities, varies) in probed.into_iter().flatten() {
        bound.insert(instance.clone(), capabilities);
        coverage_varies.insert(instance, varies);
    }
    let physical_hair = bound.values().any(|c| c.physical_hair);
    validate_bound_geometry(scene, &ids, &mut compiled)?;
    Ok(Some(Programs {
        native: if compiled.is_empty() {
            None
        } else {
            Some(std::sync::Arc::new({
                let mut all=native(&abi, &compiled, physical_hair, false)?;
                all.receivers=Some(Box::new(native(&abi,&compiled,physical_hair,true)?));
                all.camera=compiled.iter().zip(&sources).map(|(p,(_,origin))| {
                    let mut one=native(&abi,std::slice::from_ref(p),physical_hair,false)?;
                    one.name=origin.display().to_string();
                    // Primary evaluation is dispatched by bin, so program ids never
                    // enter its source or cache key. Secondary hits use the full table.
                    one.code=one.code.replace("record.tag.x", "1u")
                        .replace("material_records[material_record(v.object)].tag.x", "1u");
                    for name in ["program_dispatch","hair_dispatch","lobe_dispatch","lit_dispatch","displacement_dispatch","aov_dispatch"] {
                        one.code=one.code.replace(name,&format!("primary_{name}"));
                        for (module,text) in &mut one.modules {
                            if module==name {*module=format!("primary_{name}");}
                            *text=text.replace(name,&format!("primary_{name}"));
                        }
                    }
                    Ok(one)
                }).collect::<Result<_>>()?;
                all
            }))
        },
        compiled,
        bound,
        coverage_varies,
        physical_hair,
        aovs: scene.render.aovs.iter().enumerate().map(|(i,a)|(a.name.clone(),i as u32)).collect(),
        instances: ids,
    }))
}

/// One cached compile. The key includes compiler tokens and resolved imports;
/// proof subprocesses verify their captured imports before publication.
pub(crate) fn compile(
    r: &Renderer,
    compiler: &Compiler,
    abi: &Abi,
    source: &str,
    origin: &Path,
) -> Result<Program> {
    let program = compile_quiet(r, compiler, abi, source, origin)?;
    emit_dropped_stages(&program, origin);
    Ok(program)
}

/// The key `compile` stores a program under; its tests probe the store with it.
fn compile_key(abi: &Abi, identity: &vork_slang::InputIdentity) -> Result<blake3::Hash> {
    Ok(input_key(b"program-v8", abi, &identity.identity)?.finalize())
}

// Launch preanalysis defers visible-loss diagnostics to the consuming traversal.
fn compile_quiet(r: &Renderer, compiler: &Compiler, abi: &Abi, source: &str, origin: &Path) -> Result<Program> {
    let identity=program_identity(&r.disk_cache,compiler,abi,source,&origin)?;
    let key = compile_key(abi, &identity)?;
    let mut program = cached(
        r,
        key,
        &compiler.with_identity(&identity),
        abi, source, &origin,
        |compiler| {
            let tag = &key.to_hex()[..12];
            let program = compiler
                .compile_surface(abi, source, &origin, tag)
                .with_context(|| format!("compile {}", origin.display()))?;
            Ok(program)
        },
    )?;
    program.set_native_source(source, origin)?;
    program.install_identity=Some(identity);
    Ok(program)
}

fn emit_dropped_stages(program: &Program, origin: &Path) {
    for stage in &program.dropped_stages {
        crate::warnings::dropped("hair_stage", serde_json::json!({"program":origin,"stage":stage}),
            format!("{stage} stage on Hair; retaining the physical hair surface"));
    }
}

fn validate_bound_geometry(
    scene: &v2::Scene,
    ids: &HashMap<String, usize>,
    compiled: &mut [Program],
) -> Result<()> {
    fn visit(
        scene: &v2::Scene,
        nodes: &[v2::Node],
        ids: &HashMap<String, usize>,
        compiled: &mut [Program],
    ) -> Result<()> {
        for node in nodes {
            let (materials, curves): (Vec<&String>, bool) = match &node.kind {
                v2::NodeKind::Mesh {
                    geometry,
                    materials,
                    ..
                } => (
                    materials.iter().collect(),
                    matches!(scene.geometry[geometry].topology, v2::Topology::Curves),
                ),
                v2::NodeKind::Asset { materials, .. } => (materials.values().collect(), false),
                _ => (Vec::new(), false),
            };
            for name in materials {
                if scene.materials[name].program=="std/openpbr" {
                    for (input,value) in &scene.materials[name].inputs {
                        if let v2::Input::Attribute(binding)=value {
                            if let v2::NodeKind::Mesh {geometry,..}=&node.kind {
                                let buffer=crate::attributes::resolve(scene,node,&binding.attribute).with_context(||format!("material {name}"))?;
                                let width=match vork_scene::OpenPbrParam::find(input).context("OpenPBR input")?.kind {vork_scene::OpenPbrKind::Float=>1,vork_scene::OpenPbrKind::Color3=>3,_=>anyhow::bail!("attribute requires float or colour destination")};
                                crate::attributes::validate(binding,width,crate::attributes::arity(scene.buffers[buffer].element_type)?).with_context(||format!("material {name}: input {input}: geometry {geometry}"))?;
                            }
                        }
                    }
                }
                if let Some(instance) = instance(scene, name)? {
                    if let Some(&i) = ids.get(&instance_key(&instance)) {
                        for image in &instance.images {
                            if compiled[i].capabilities.displacement && compiled[i].displacement_images.contains(&image.input) && image.source.changes_between_frames() {
                                crate::warnings::dropped("displacement_image", serde_json::json!({"material":name,"program":scene.materials[name].program,"image":image.input,"inputs":scene.materials[name].inputs}), format!("warning: material {name}: ignored displacement sampling a changing/render image before geometry evaluation"));
                                compiled[i].capabilities.displacement=false;
                            }
                        }

                        for field in &compiled[i].inputs.fields {
                            if let vork_slang::Type::Attribute(width)=field.ty {
                                let material=&scene.materials[name];
                                let Some(v2::Input::Attribute(binding))=material.inputs.get(&field.name) else {anyhow::bail!("material {name}: Attribute input {} requires an attribute binding",field.name)};
                                if let v2::NodeKind::Mesh {geometry,..}=&node.kind {
                                    let buffer=crate::attributes::resolve(scene,node,&binding.attribute).with_context(||format!("material {name}"))?;
                                    let arity=crate::attributes::arity(scene.buffers[buffer].element_type)?;
                                    crate::attributes::validate(binding,width,arity).with_context(||format!("material {name}: input {}: geometry {geometry}",field.name))?;
                                }
                            }
                        }
                        ensure!(
                            !compiled[i].capabilities.hair || curves,
                            "material {name}: Hair surface requires curves topology"
                        );
                    }
                }
            }
            visit(scene, &node.children, ids, compiled)?;
        }
        Ok(())
    }
    visit(scene, &scene.nodes, ids, compiled)
}

/// Map `f` over `items` on up to one thread per core, keeping their order.
pub(crate) fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> Result<R> + Sync) -> Result<Vec<R>> {
    parallel_map_with_limit(items, usize::MAX, f)
}

pub(crate) fn parallel_map_with_limit<T: Sync, R: Send>(items: &[T], limit: usize, f: impl Fn(&T) -> Result<R> + Sync) -> Result<Vec<R>> {
    let warning_collector = crate::warnings::Collector::current();
    let cursor = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let workers = items.len().min(limit).min(std::thread::available_parallelism().map_or(1, usize::from));
        let jobs: Vec<_> = (0..workers).map(|_| scope.spawn(|| {
            let _warning_scope = warning_collector.as_ref().map(|c| c.enter());
            let mut results = Vec::new();
            loop {
                let index = cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(item) = items.get(index) else { break };
                results.push((index, f(item)));
            }
            Ok::<_, anyhow::Error>(results)
        })).collect();
        let mut results = Vec::new();
        for job in jobs { results.extend(job.join().expect("program worker panicked")?); }
        results.sort_by_key(|(index, _)| *index);
        let mut values = Vec::new();
        let mut errors = Vec::new();
        for (_, result) in results {
            match result {
                Ok(value) => values.push(value),
                Err(error) => errors.push(error),
            }
        }
        if !errors.is_empty() {
            let messages = errors.iter().map(|error| format!("{error:#}")).collect::<Vec<_>>().join("\n");
            return Err(errors.remove(0).context(messages));
        }
        Ok(values)
    })
}

fn semantic_indices<'a>(identities:impl Iterator<Item=&'a [u8]>)->Vec<usize> {
    let mut unique=std::collections::BTreeMap::new();
    identities.map(|identity| {
        let next=unique.len();
        *unique.entry(identity).or_insert(next)
    }).collect()
}

/// The cache key prefix of a proof about one compiler-input snapshot.
fn input_key(tag: &[u8], abi: &Abi, identity: &[u8]) -> Result<blake3::Hasher> {
    let mut key = blake3::Hasher::new();
    let abi_identity=abi.cache_identity()?;
    for part in [tag,abi_identity.as_bytes(),identity] {
        key.update(&(part.len() as u64).to_le_bytes());key.update(part);
    }
    Ok(key)
}

/// The store's record of the toolchains that passed the version check
/// (`vork_slang::Compiler::toolchain`). One entry per toolchain in the store that `cache`
/// names entries in, named by the toolchain itself, so every kind of lookup in that store
/// reads and writes the same record. It names this host's files: like the exact receipts it
/// sits beside them and is not among the kinds read from shipped shaders.
struct Admissions<'a>(&'a dyn Fn(blake3::Hash)->PathBuf);
impl vork_slang::AdmissionLedger for Admissions<'_> {
    fn entry(&self, toolchain: blake3::Hash) -> Option<PathBuf> {
        Some((self.0)(toolchain).parent()?.join(format!("{}.slang-admission-v1.entry",toolchain.to_hex())))
    }
    fn admitted(&self, entry: &Path, toolchain: blake3::Hash) -> bool {
        disk_cache::read(entry).is_some_and(|held| held==toolchain.as_bytes())
    }
    fn record(&self, entry: &Path, toolchain: blake3::Hash) {
        let _=disk_cache::write(entry,toolchain.as_bytes());
    }
}

/// Admit the selected compiler against the store that `cache` names entries in. Every
/// receipt, proof and compiler-named key starts here.
pub(crate) fn admit<'c>(cache: &dyn Fn(blake3::Hash)->PathBuf, compiler: &'c Compiler) -> Result<vork_slang::Admitted<'c>> {
    compiler.admit(&Admissions(cache))
}

fn program_identity(store: &disk_cache::Store, compiler: &Compiler, abi: &Abi, source: &str, origin: &Path) -> Result<vork_slang::InputIdentity> {
    bound_program_identity(store,compiler,abi,source,origin,"")
}

pub(crate) fn bound_program_identity(
    store: &disk_cache::Store, compiler: &Compiler, abi: &Abi,
    source: &str, origin: &Path, bindings: &str,
) -> Result<vork_slang::InputIdentity> {
    let admitted=admit(&|key|store.path("slang-input",1,key),compiler)?;
    let mut raw=input_key(b"program-input-receipt-v2",abi,source.as_bytes())?;
    raw.update(&(bindings.len() as u64).to_le_bytes());raw.update(bindings.as_bytes());
    for part in [admitted.input_policy(),std::env::current_dir()?.join(origin).to_string_lossy().into_owned()] {
        raw.update(&(part.len() as u64).to_le_bytes());raw.update(part.as_bytes());
    }
    let path=store.path("slang-input",1,raw.finalize());
    if let Some(identity) = read_native_input_receipt(&path) { return Ok(identity); }
    let roots=compiler.program_input_roots(origin)?;
    let mut portable=input_key(b"closed-program-input-v1",abi,source.as_bytes())?;
    let policy=admitted.portable_input_policy();
    // Cache lookup must not add a UTF-8 restriction to compiler origins.
    for part in [bindings.as_bytes(),origin.as_os_str().as_encoded_bytes(),policy.as_bytes()] {
        portable.update(&(part.len() as u64).to_le_bytes());portable.update(part);
    }
    let closed=closed_receipt_path(&store.path("slang-program",2,portable.finalize()))?;
    if let Some(identity)=read_closed_input_receipt(&closed,&roots) {
        write_input_receipt(&path,&identity)?;
        return Ok(identity);
    }
    let prepared=compiler.prepare_bound_program(abi,source,origin,bindings)
        .with_context(|| format!("program source {}", origin.display()))?;
    let identity=prepared.receipt();
    write_input_receipt(&path, &identity)?;
    write_closed_input_receipt(&closed,&identity,&roots)?;
    Ok(identity)
}

/// Exact-input receipts and normalized identities use the same native artifact
/// store. A miss retains the captured filesystem for subsequent code generation.
pub(crate) fn native_input_identity(
    cache: &dyn Fn(blake3::Hash)->PathBuf, compiler: &Compiler,
    modules: &[(&str,&str)], main: &str, entries: &[(&str,&str)], args: &[&str],
    identities: &std::collections::BTreeMap<String,Vec<u8>>,
    inputs: &[vork_slang::InputIdentity],
) -> Result<(vork_slang::InputIdentity,Option<vork_slang::PreparedModules>)> {
    let path=native_input_receipt_path(cache,compiler,modules,main,entries,args,identities,inputs)?;
    if let Some(identity)=read_native_input_receipt(&path)
        && inputs.iter().all(|input| identity.verify(input).is_ok()) {return Ok((identity,None));}
    let (closed,roots)=native_closed_input_receipt(cache,compiler,modules,main,entries,args,identities,inputs)?;
    if let Some(identity)=read_closed_input_receipt(&closed,&roots)
        && inputs.iter().all(|input| identity.verify(input).is_ok()) {
        write_input_receipt(&path,&identity)?;
        return Ok((identity,None));
    }
    let prepared=compiler.prepare_identified_modules(modules,main,entries,args,identities)?;
    let identity=prepared.receipt();
    for input in inputs { identity.verify(input)?; }
    write_input_receipt(&path, &identity)?;
    write_closed_input_receipt(&closed,&identity,&roots)?;
    Ok((identity,Some(prepared)))
}

/// A receipt is a lookup hint, never a reason to reject freshly captured inputs.
/// Return code only under the identity of the snapshot that produced it.
pub(crate) fn native_code(
    identity: &vork_slang::InputIdentity,
    prepared: Option<vork_slang::PreparedModules>,
    inputs: &[vork_slang::InputIdentity],
    receipt_path: &Path,
    artifact: impl Fn(&[u8]) -> PathBuf,
    prepare: impl FnOnce() -> Result<vork_slang::PreparedModules>,
    compile: impl FnOnce(&vork_slang::PreparedModules) -> Result<Vec<u8>>,
) -> Result<Vec<u8>> {
    if let Some(code) = disk_cache::read(&artifact(&identity.identity)) { return Ok(code); }
    let prepared = match prepared { Some(prepared) => prepared, None => prepare()? };
    let fresh = prepared.receipt();
    for input in inputs { fresh.verify(input)?; }
    if fresh.identity != identity.identity { write_input_receipt(receipt_path, &fresh)?; }
    let path = artifact(prepared.identity());
    if let Some(code) = disk_cache::read(&path) { return Ok(code); }
    let code = compile(&prepared)?;
    let _ = disk_cache::write(&path, &code);
    Ok(code)
}

/// Scheduling probes never compile: an unknown receipt is a miss. Like every receipt
/// path they admit the compiler first, which runs it only when the store has not seen
/// this toolchain pass.
/// Only the build owns preparation, retaining its snapshot through codegen.
pub(crate) fn cached_native_input_identity(
    cache: &dyn Fn(blake3::Hash)->PathBuf, compiler: &Compiler,
    modules: &[(&str,&str)], main: &str, entries: &[(&str,&str)], args: &[&str],
    identities: &std::collections::BTreeMap<String,Vec<u8>>,
    inputs: &[vork_slang::InputIdentity],
) -> Result<Option<vork_slang::InputIdentity>> {
    let local=read_native_input_receipt(&native_input_receipt_path(cache,compiler,modules,main,entries,args,identities,inputs)?);
    let identity=if local.is_some() {local} else {
        let (closed,roots)=native_closed_input_receipt(cache,compiler,modules,main,entries,args,identities,inputs)?;
        read_closed_input_receipt(&closed,&roots)
    };
    Ok(identity.filter(|identity| inputs.iter().all(|input| identity.verify(input).is_ok())))
}

fn closed_receipt_path(artifact: &Path) -> Result<PathBuf> {
    let name=artifact.file_name().and_then(|n|n.to_str()).and_then(|n|n.split_once('.')).context("native cache file name")?.0;
    Ok(artifact.with_file_name(format!("{name}.slang-closed-input-v1.entry")))
}

fn read_closed_input_receipt(path: &Path, roots: &[PathBuf]) -> Option<vork_slang::InputIdentity> {
    let identity:vork_slang::ClosedInputIdentity=serde_json::from_slice(&disk_cache::read(path)?).ok()?;
    identity.relocate(roots)
}

fn write_closed_input_receipt(path: &Path, identity: &vork_slang::InputIdentity, roots: &[PathBuf]) -> Result<()> {
    if let Some(closed)=identity.closed(roots) {
        let _=disk_cache::write(path,&serde_json::to_vec(&closed)?);
    }
    Ok(())
}

fn native_closed_input_receipt(
    cache: &dyn Fn(blake3::Hash)->PathBuf, compiler: &Compiler,
    modules: &[(&str,&str)], main: &str, entries: &[(&str,&str)], args: &[&str],
    identities: &std::collections::BTreeMap<String,Vec<u8>>, inputs: &[vork_slang::InputIdentity],
) -> Result<(PathBuf,Vec<PathBuf>)> {
    let mut include=false;
    let mut includes=Vec::new();
    let portable_args:Vec<_>=args.iter().map(|arg| {
        if include {include=false;includes.push(PathBuf::from(arg)); "<search-root>"}
        else {include=*arg=="-I";*arg}
    }).collect();
    let mut raw=blake3::Hasher::new();
    // This key binds the entire request, including unused supplied modules.
    // Only path strings are relocated; each observed absence is checked again.
    fn part(hash:&mut blake3::Hasher,bytes:&[u8]) {hash.update(&(bytes.len() as u64).to_le_bytes());hash.update(bytes);}
    part(&mut raw,b"closed-native-input-v1");part(&mut raw,main.as_bytes());
    raw.update(&(entries.len() as u64).to_le_bytes());
    for (entry,stage) in entries {part(&mut raw,entry.as_bytes());part(&mut raw,stage.as_bytes());}
    raw.update(&(portable_args.len() as u64).to_le_bytes());
    for arg in portable_args {part(&mut raw,arg.as_bytes());}
    raw.update(&(modules.len() as u64).to_le_bytes());
    for (name,source) in modules {part(&mut raw,name.as_bytes());part(&mut raw,source.as_bytes());}
    part(&mut raw,admit(cache,compiler)?.portable_input_policy().as_bytes());
    raw.update(&(inputs.len() as u64).to_le_bytes());
    for input in inputs {part(&mut raw,&input.identity);}
    raw.update(&(identities.len() as u64).to_le_bytes());
    for (name,identity) in identities {part(&mut raw,name.as_bytes());part(&mut raw,identity);}
    Ok((closed_receipt_path(&cache(raw.finalize()))?,compiler.input_roots(&includes)?))
}

fn read_native_input_receipt(path: &Path) -> Option<vork_slang::InputIdentity> {
    let identity: vork_slang::InputIdentity = serde_json::from_slice(&disk_cache::read(path)?).ok()?;
    identity.valid().then_some(identity)
}

pub(crate) fn write_input_receipt(path: &Path, identity: &vork_slang::InputIdentity) -> Result<()> {
    let bytes = serde_json::to_vec(identity)?;
    let _ = disk_cache::write(path, &bytes);
    Ok(())
}

pub(crate) fn native_input_receipt_path(
    cache: &dyn Fn(blake3::Hash)->PathBuf, compiler: &Compiler,
    modules: &[(&str,&str)], main: &str, entries: &[(&str,&str)], args: &[&str],
    identities: &std::collections::BTreeMap<String,Vec<u8>>,
    inputs: &[vork_slang::InputIdentity],
) -> Result<PathBuf> {
    let cwd = std::env::current_dir()?;
    let mut include = false;
    let absolute_args: Vec<String> = args.iter().map(|arg| {
        let value = if include {
            let path = cwd.join(arg);
            path.canonicalize().unwrap_or(path).to_string_lossy().into_owned()
        } else { (*arg).to_owned() };
        include = *arg == "-I";
        value
    }).collect();
    let args: Vec<&str> = absolute_args.iter().map(String::as_str).collect();
    let mut raw=blake3::Hasher::new();
    for part in std::iter::once("native-input-receipt-v2").chain(std::iter::once(main))
        .chain(entries.iter().flat_map(|(entry,stage)|[*entry,*stage])).chain(args.iter().copied())
        .chain(modules.iter().flat_map(|(name,source)|[*name,*source])) {
        raw.update(&(part.len() as u64).to_le_bytes());raw.update(part.as_bytes());
    }
    raw.update(admit(cache,compiler)?.input_policy().as_bytes());
    // Every installed program is a dependency of this native composition, even
    // if its import paths are disjoint from a previously cached composition.
    for input in inputs {
        raw.update(&(input.identity.len() as u64).to_le_bytes());
        raw.update(&input.identity);
    }
    for (name,identity) in identities {raw.update(&(name.len() as u64).to_le_bytes());raw.update(name.as_bytes());raw.update(&(identity.len() as u64).to_le_bytes());raw.update(identity);}
    // Exact resolver receipts contain host paths. Keep them in the same store
    // but outside the four portable shader entry kinds shipped with releases.
    let artifact_path=cache(raw.finalize());
    let name=artifact_path.file_name().and_then(|n|n.to_str()).and_then(|n|n.split_once('.')).context("native cache file name")?.0;
    let path=artifact_path.with_file_name(format!("{name}.slang-input-v1.entry"));
    Ok(path)
}

#[cfg(test)]
fn program_key(tag: &[u8], abi: &Abi, source: &str, origin: &Path) -> Result<blake3::Hasher> {
    let prepared=Compiler::discover().prepare_program(abi,source,&origin.to_path_buf())?;
    input_key(tag,abi,prepared.identity())
}

pub(crate) fn cached<T: serde::Serialize + serde::de::DeserializeOwned>(
    r: &Renderer,
    key: blake3::Hash,
    compiler: &Compiler,
    abi: &Abi, source: &str, origin: &Path,
    build: impl FnOnce(&Compiler) -> Result<T>,
) -> Result<T> {
    cached_proof(&r.disk_cache,key,compiler,abi,source,origin,|compiler| {
        let start=std::time::Instant::now();
        let result=build(compiler);
        crate::timing::creation(&r.device,&format!("program {key}"),"Slang",start.elapsed().as_secs_f64()*1000.,0.);
        result
    })
}

/// All proof stages share exact compiler receipts. Stage ABI, generated bindings
/// and options belong in the caller key; the compiler policy belongs here.
fn cached_proof<T: serde::Serialize + serde::de::DeserializeOwned>(
    store: &disk_cache::Store, key: blake3::Hash, compiler: &Compiler,
    abi: &Abi, source: &str, origin: &Path,
    build: impl FnOnce(&Compiler) -> Result<T>,
) -> Result<T> {
    // A stored proof is this compiler's answer: admit it before reading one, also when
    // the caller supplies the receipt.
    admit(&|key|store.path("slang-input",1,key),compiler)?;
    let identity=match compiler.input_identity() {
        Some(identity)=>identity.clone(),
        None=>program_identity(store,compiler,abi,source,origin)?,
    };
    let key=input_key(key.as_bytes(),abi,&identity.identity)?.finalize();
    let path=store.path("slang-program",2,key);
    if identity.valid()
        && let Some(bytes)=disk_cache::read(&path)
        && let Ok(program)=serde_json::from_slice::<T>(&bytes) {return Ok(program);}
    let (program,receipt)=compiler.with_identity(&identity).capture_inputs(build)
        .with_context(|| format!("program source {}", origin.display()))?;
    receipt.verify_proof(&identity)?;
    if identity.valid() && receipt.valid() {
        let _=disk_cache::write(&path,&serde_json::to_vec(&program)?);
    }
    Ok(program)
}

/// Complete generated shaders have no authored resolver inputs. Their caller
/// keys include the whole validated source, binding map and compiler options.
pub(crate) fn cached_generated<T: serde::Serialize + serde::de::DeserializeOwned>(
    r: &Renderer, key: blake3::Hash, build: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let path=r.disk_cache.path("slang-program",2,key);
    if let Some(bytes)=disk_cache::read(&path)
        && let Ok(program)=serde_json::from_slice::<T>(&bytes) {return Ok(program);}
    let start=std::time::Instant::now();
    let program=build();
    crate::timing::creation(&r.device,&format!("program {key}"),"Slang",start.elapsed().as_secs_f64()*1000.,0.);
    let program=program?;
    let _=disk_cache::write(&path,&serde_json::to_vec(&program)?);
    Ok(program)
}

/// A scene's programs linked into the Slang colour passes (docs/design/shaders.md,
/// Programs). Each program is its own module, so its identifiers need no tags;
/// `program_dispatch` is the only module that imports them, and the variant module
/// applies its result through `struct Programs : IProgram`. Context and closure
/// cross as packed vec4s: one module cannot see both `vork` and the renderer's
/// Closure, and the packing inlines away.
#[derive(Clone)]
pub(crate) struct Native {
    /// Independent primary evaluators, indexed by installed program id minus one.
    pub(crate) camera: Vec<Native>,
    /// Volume receiver depth needs opacity, including fibre extinction, not
    /// unrelated opaque material bodies. Tags and the shared Blend ABI stay global.
    receivers: Option<Box<Native>>,
    pub(crate) coverage: bool,
    /// Modules a variant compile adds: native `vork` and its std modules, each
    /// program, and the dispatch.
    pub(crate) modules: Vec<(String, String)>,
    /// Directories program imports resolve from.
    pub(crate) include: Vec<PathBuf>,
    pub(crate) inputs: Vec<vork_slang::InputIdentity>,
    pub(crate) identities: std::collections::BTreeMap<String,Vec<u8>>,
    /// The variant module's imports and `struct Programs`.
    pub(crate) code: String,
    /// A primary evaluator's program source, for diagnostics.
    pub(crate) name: String,
}
impl Native {
    pub(crate) fn for_coverage(&self)->&Self {self.receivers.as_deref().unwrap_or(self)}
}

/// Path kernels keep each material dispatch out of line in Slang (docs/design/implementation.md).
const PATH_NOINLINE: &str = "#ifdef VORK_PATH_SLANG_OUTLINE\n[noinline]\n#endif\n";
const CLAMPS: &str = "o.closure.geometry_opacity=clamp(o.closure.geometry_opacity,0.,1.);o.closure.transmission_weight=clamp(o.closure.transmission_weight,0.,1.);\n";
pub(crate) fn native(abi: &Abi, programs: &[Program], physical_hair: bool, coverage_only: bool) -> Result<Native> {
    let retained=|p:&Program|!coverage_only || p.capabilities.opacity || p.capabilities.hair;
    let scattering = if physical_hair { "h[2].w!=0." } else { "false" };
    let (c, k) = (abi.context_vec4s(), abi.closure_vec4s());
    let aovs = abi.aov_count.max(1);
    let aov_count = abi.aov_count;
    let hair_k = vork_slang::HAIR_VEC4S;
    let mut modules = abi.native_modules();
    // Standalone displacement and medium coverage compiles share this exact
    // Surface layout too, including compiles outside SlangForward.
    let constants = crate::native::SOURCES.iter().find(|(name,_)| *name=="vork_constants").unwrap().1;
    modules.push(("vork_constants".into(), format!("#ifndef VORK_AOV_COUNT\n#define VORK_AOV_COUNT {aov_count}\n#endif\n{constants}")));
    if programs.iter().any(|p| p.capabilities.hair) { modules.push(("hair_lighting".into(), include_str!("../shaders/programs/hair_lighting.slang").into())); }
    let mut include = Vec::new();
    let mut inputs:Vec<vork_slang::InputIdentity>=Vec::new();
    let mut identities=std::collections::BTreeMap::new();
    let mut imports = String::new();
    let mut cases = String::new();
    let mut aov_cases=String::new();
    let mut hair_cases = String::new();
    let mut lobe_cases = String::new();
    let mut lit_cases = String::new();
    let mut displacement_cases = String::new();
    for (i, p) in programs.iter().enumerate() {
        if !retained(p) {continue;}
        if let Some(identity)=&p.install_identity {
            for previous in &inputs { identity.verify(previous)?; }
            inputs.push(identity.clone());
        }
        let tag = p
            .surface
            .strip_prefix("vork_surface_")
            .and_then(|s| s.split("_0_").next())
            .context("program surface name")?;
        // The native entry's compiler snapshot owns resolved import identity.
        modules.push((format!("program_{tag}"), p.slang.clone()));
        if let Some(input)=&p.install_identity {
            if let Some(previous)=identities.insert(format!("program_{tag}"),input.identity.clone()) {
                ensure!(previous==input.identity,"material module identifier collision");
            }
        }
        include.extend(p.include.iter().cloned());
        writeln!(imports, "import program_{tag};")?;
        if p.capabilities.hair { writeln!(hair_cases, "    case {}u: vork_surface_{tag}(a,base,r); return true;", i+1)?; } else if p.capabilities.lobes {
        writeln!(lobe_cases, "    case {}u: vork_surface_{tag}(a,base,lobe,u,r,lobes); return true;", i + 1)?;
        } else {
        writeln!(cases, "    case {}u: vork_surface_{tag}(a,base,r); return true;", i + 1)?;
        }
        if p.aov.is_some() {writeln!(aov_cases,"case {}u: vork_aov_{tag}(a,base,r); return;",i+1)?;}
        if p.capabilities.displacement {writeln!(displacement_cases, "case {}u: return vork_displace_{tag}(position,normal,time,base);",i+1)?;}
        if p.capabilities.lit {writeln!(lit_cases, "    case {}u: vork_lit_{tag}(a,base,light,r); return true;", i + 1)?;}
    }
    modules.push((
        "program_dispatch".into(),
        format!(
            "// Generated by programs.rs: the scene's material programs by record tag.\n\
             {imports}\
             {PATH_NOINLINE}public bool program_dispatch(uint program, float4 a[{c}], uint base, out float4 r[{k}]) {{\n    \
             for (uint i=0u;i<{k}u;i++) {{r[i]=float4(0.);}}\n    \
             switch (program) {{\n{cases}    default: return false;\n    }}\n}}\n\
             {PATH_NOINLINE}public void aov_dispatch(uint program,float4 a[{c}],uint base,out float4 r[{aovs}]) {{r={{}};switch(program){{{aov_cases} default:return;}}}}\n"
        ),
    ));
    if !displacement_cases.is_empty() {
        modules.push(("displacement_dispatch".into(), format!("{imports}\npublic float3 displacement_dispatch(uint program,float3 position,float3 normal,float time,uint base) {{switch(program) {{{displacement_cases} default:return float3(0.);}}}}")));
    }
    let lobes = !lobe_cases.is_empty();
    if lobes {
        modules.push(("lobe_dispatch".into(), format!("{imports}\n// Blend programs: one lobe's closure, (weight, lobe count).\n{PATH_NOINLINE}public bool lobe_dispatch(uint program,float4 a[{c}],uint base,uint lobe,float u,out float4 r[{k}],out float2 lobes) {{for(uint i=0u;i<{k}u;i++) {{r[i]=float4(0.);}}lobes=float2(1.,1.);switch(program) {{\n{lobe_cases}default:return false;}}}}")));
    }
    let lit = !lit_cases.is_empty();
    if lit {
        modules.push(("lit_dispatch".into(), format!("{imports}\n// Lit stages (Shader to RGB): the closure shaded in place of the surface's.\n{PATH_NOINLINE}public bool lit_dispatch(uint program,float4 a[{c}],uint base,float4 light[2],out float4 r[{k}]) {{for(uint i=0u;i<{k}u;i++) {{r[i]=float4(0.);}}switch(program) {{\n{lit_cases}default:return false;}}}}")));
    }
    let lit_reflects = programs.iter().any(|p| retained(p) && p.capabilities.lit && p.capabilities.lit_reflects);
    modules.push(("hair_dispatch".into(), format!("{imports}\n{PATH_NOINLINE}public bool hair_dispatch(uint program,float4 a[{c}],uint base,out float4 r[{hair_k}]) {{for(uint i=0u;i<{hair_k}u;i++) r[i]=float4(0.);switch(program) {{{hair_cases}default:return false;}}}}")));
    for (name, source) in &mut modules {
        if matches!(name.as_str(), "program_dispatch" | "lobe_dispatch" | "hair_dispatch" | "lit_dispatch") {
            *source = vork_slang::outline_native_material_functions(source);
        }
    }
    include.sort();
    include.dedup();
    let pack = abi.pack_context_slang(|name, element| {
        let value = match (name, element) {
            ("POSITION", _) => "v.world",
            ("NORMAL", _) => "s.closure.geometry_normal",
            ("true_normal", _) => "true_normal",
            ("incoming", _) => "k.incoming",
            ("TANGENT", _) => "v.TANGENT.xyz*select(s.front_facing,1.,-1.)",
            ("position_dx", _) => "k.position_dx",
            ("position_dy", _) => "k.position_dy",
            ("COLOR_0", _) => "v.COLOR_0",
            ("TEXCOORD_0", _) => "v.TEXCOORD_0",
            ("TEXCOORD_1", _) => "varying_uv1(v)",
            ("duv_dx", 0) => "k.uv_dx",
            ("duv_dy", 0) => "k.uv_dy",
            ("duv_dx", 1) => "k.uv1_dx",
            ("duv_dy", 1) => "k.uv1_dy",
            ("time", _) => "frame.scene.x",
            ("wavelength", _) => "vork_path_wavelength",
            // The ID channel's object id, not the internal object index.
            ("object_id", _) => "uint(objects[v.object].opacity.y)",
            ("object_index", _) => "v.object",
            ("instance_index", _) => "objects[v.object].padding.y",
            ("primitive_id", _) => "material_words[objects[v.object].reserved+14u+v.primitive]",
            ("front_facing", _) => "s.front_facing",
            ("tangent_sign", _) => "select(v.TANGENT.w<0.,-1.,1.)",
            ("sample_spread", _) => "k.material_filter",
            ("has_tangent", _) => "abs(v.TANGENT.w)<2. && dot(v.TANGENT.xyz,v.TANGENT.xyz)>=1e-12",
            ("rest_position", _) => "v.rest",
            ("rest_position_dx", _) => "k.rest_dx",
            ("rest_position_dy", _) => "k.rest_dy",
            ("unperturbed_normal", _) => "s.geometric",
            ("unperturbed_normal_dx", _) => "k.normal_dx",
            ("unperturbed_normal_dy", _) => "k.normal_dy",
            ("incoming_dx", _) => "k.incoming_dx",
            ("incoming_dy", _) => "k.incoming_dy",
            _ => return None,
        };
        Some(value.to_owned())
    });
    let coverage = programs.iter().any(|p| p.capabilities.opacity);
    // Blend programs unpack the requested lobe over the same defaults.
    let lobe_branch = if lobes {
        format!("    float2 lobes;\n    \
                 if (lobe_dispatch(record.tag.x,a,record.tag.w,lobe.index,lobe.u,r,lobes)) {{\n    \
                 o.closure=closure_defaults(s.closure.geometry_normal,v.TANGENT.xyz);\n{}    \
                 o.weight=lobes.x;o.lobes=uint(lobes.y);\n    \
                 return o;\n    }}\n", abi.unpack_closure_slang("o.closure") + CLAMPS)
    } else {
        String::new()
    };
    // The lit stage sees the surface stage's context: its shading normal, not the program's.
    let lit_stage = if lit {
        format!("    surface_material_context(v,k);\n    \
                 let true_normal=coordinates_true_normal(k,s.geometric)*select(s.front_facing,1.,-1.);\n    \
                 let record=material_records[material_record(v.object)];\n    \
                 var base=s;base.closure.geometry_normal=s.normal;\n    \
                 float4 a[{c}]={{}};\n{}    \
                 float4 light[2]={{float4(lighting.color,lighting.alpha),float4(lighting.direct,0.)}};\n    \
                 float4 r[{k}];\n    \
                 if (!lit_dispatch(record.tag.x,a,record.tag.w,light,r)) {{return s;}}\n    \
                 var o=s;o.weight=1.;o.lobes=1u;\n    \
                 o.closure=closure_defaults(s.normal,v.TANGENT.xyz);\n{}    \
                 // Coverage stays the surface stage's; the result does not refract.\n    \
                 o.closure.geometry_opacity=s.closure.geometry_opacity;o.closure.transmission_weight=0.;\n    \
                 return o;\n",
            pack.replace("s.closure.geometry_normal", "base.closure.geometry_normal"),
            abi.unpack_closure_slang("o.closure"))
    } else {
        "    return s;\n".into()
    };
    let lit_programs = programs
        .iter()
        .enumerate()
        .filter(|(_, p)| retained(p) && p.capabilities.lit)
        .map(|(i, _)| format!("case {}u: return true;", i + 1))
        .collect::<String>();
    let code = format!(
        "import material;\nimport frame;\nimport stream;\nimport closure;\nimport surface;\nimport program_dispatch;\nimport hair_dispatch;\n{}{}\
         struct Programs : IProgram {{\n\
         static const bool surface = true;\n\
         static const bool coverage = {coverage};\n\
         static const bool lobes = {lobes};\n\
         static const bool lit = {lit};\n\
         static const bool lit_reflects = {lit_reflects};\n\
         static bool lit_material(Varying v) {{switch (material_records[material_record(v.object)].tag.x) {{{lit_programs} default: return false;}}}}\n\
         static Surface apply_lit(Varying v, Surface s, Coordinates k, SurfaceLighting lighting) {{\n{lit_stage}}}\n\
         static Surface apply(Varying v, Surface s, Coordinates k, Lobe lobe) {{\n    \
         let true_normal=coordinates_true_normal(k,s.geometric)*select(s.front_facing,1.,-1.);\n    \
         let record=material_records[material_record(v.object)];\n    \
         if (record.tag.x==0u) {{return s;}}\n    \
         float4 a[{c}]={{}};\n{pack}    \
         var o=s;\n    \
         float4 aovs[{aovs}];if ({aov_count}u!=0u) aov_dispatch(record.tag.x,a,record.tag.w,aovs);\n    \
         o.aovs=aovs;\n    \
         float4 h[{hair_k}];\n    \
         if (hair_dispatch(record.tag.x,a,record.tag.w,h)) {{o.hair0=h[0];o.hair1=h[1];o.is_hair={scattering};o.strand=v.TANGENT.xyz;o.closure.specular_weight=0.;o.closure.coat_weight=0.;o.closure.fuzz_weight=0.;o.closure.base_color=h[2].xyz;o.closure.base_weight=select(o.is_hair,0.,1.);o.closure.base_metalness=0.;o.closure.transmission_weight=0.;return o;}}\n    \
         float4 r[{k}];\n{lobe_branch}    \
         if (!program_dispatch(record.tag.x,a,record.tag.w,r)) {{return o;}}\n    \
         // OpenPBR defaults in the unperturbed frame, then the program's fields.\n    \
         o.closure=closure_defaults(s.closure.geometry_normal,v.TANGENT.xyz);\n{}    \
         return o;\n}}\n}};\n",
        if lobes { "import lobe_dispatch;\n" } else { "" },
        if lit { "import lit_dispatch;\n" } else { "" },
        abi.unpack_closure_slang("o.closure") + CLAMPS
    );
    Ok(Native { camera:Vec::new(), receivers:None, coverage, modules, include, inputs, identities, code, name:String::new() })
}

#[cfg(test)]
mod help_tests {
    use super::*;
    #[test]
    fn parallel_errors_follow_input_order() {
        if std::thread::available_parallelism().map_or(1, usize::from) < 2 { return; }
        let second_started = std::sync::atomic::AtomicBool::new(false);
        let error = parallel_map_with_limit(&[0, 1, 2], 2, |i| -> Result<()> {
            match i {
                0 => {
                    while !second_started.load(std::sync::atomic::Ordering::Acquire) { std::thread::yield_now(); }
                    Ok(())
                }
                1 => {
                    second_started.store(true, std::sync::atomic::Ordering::Release);
                    anyhow::bail!("first error")
                }
                _ => anyhow::bail!("later error"),
            }
        }).unwrap_err();
        assert_eq!(error.root_cause().to_string(), "first error");
        assert!(error.to_string().starts_with("first error"), "{error:#}");
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn volume_receivers_retain_coverage_ids_hair_and_shader_capacity() -> Result<()> {
        let compiler=Compiler::discover();let abi=abi();
        let bodies=[
            "Blend surface(ShadingContext ctx,Inputs i) {Closure c={};Blend s=add(c,c);s=add(s,s);return add(s,s);}",
            "Closure surface(ShadingContext ctx,Inputs i) {Closure c={};c.geometry_opacity=i.value;return c;}",
            "Hair surface(ShadingContext ctx,Inputs i) {Hair h={};return h;}",
            "Closure surface(ShadingContext ctx,Inputs i) {Closure c={};c.base_color=float3(.314159);return c;}",
        ];
        let programs=bodies.iter().enumerate().map(|(i,body)|compiler.compile_surface(&abi,
            &format!("import vork;struct Inputs {{float value;}};{body}"),Path::new("receiver-program.slang"),&format!("receiver{i}")))
            .collect::<Result<Vec<_>>>()?;
        assert!(!programs[0].capabilities.opacity && !programs[3].capabilities.opacity);
        assert!(programs[1].capabilities.opacity && programs[2].capabilities.hair);
        assert_eq!(programs[0].capabilities.blend_lobes,Some(8));
        let receivers=native(&abi,&programs,true,true)?;
        for (i,p) in programs.iter().enumerate() {
            assert_eq!(receivers.modules.iter().any(|(_,s)|s.starts_with(&p.slang)),i==1 || i==2);
        }
        let dispatch=&receivers.modules.iter().find(|(name,_)|name=="program_dispatch").unwrap().1;
        assert!(dispatch.contains("case 2u:"));
        assert!(!dispatch.contains("case 1u:") && !dispatch.contains("case 4u:"));
        let hair=&receivers.modules.iter().find(|(name,_)|name=="hair_dispatch").unwrap().1;
        assert!(hair.contains("case 3u:"));
        for (name,text) in abi.native_modules() {
            assert_eq!(receivers.modules.iter().find(|(n,_)|*n==name).unwrap().1,text);
        }
        Ok(())
    }
    /// Programs may set the coat's shading normal (the Blender bridge writes
    /// Principled "Coat Normal" there). Unset or set to the unperturbed normal,
    /// the coat shades the same; tilted away from the half vector, its highlight
    /// falls. Before the ABI exposed it, install failed with
    /// "'geometry_coat_normal' is not a member of 'Closure'".
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    #[cfg(vork_private_tests)]
    fn programs_shade_their_coat_normal() -> Result<()> {
        let r = crate::test_renderer()?;
        let render=|integrator:&str,coat_normal:&str|->Result<[f32;3]> {
            let mut scene:serde_json::Value=vork_scene::v2::with_test_settings(include_str!("../tests/fixtures/analytic.json"), include_str!("../tests/presets/analytic-7fefd52a.json"))?;
            scene["animation"]=serde_json::json!([]);
            scene["render"]=serde_json::json!({"resolution":[8,8],"settings":{"integrator":integrator,"denoiser":null,"samples":4}});
            // The orthographic camera sees only the unit box's front face (+Z);
            // the sun shines along the view, so the half vector is the face normal.
            let mut camera=scene["nodes"][0].clone();camera["material"]["inputs"]["ortho_scale"]=serde_json::json!(0.5);
            scene["nodes"]=serde_json::json!([camera,scene["nodes"][1].clone(),
                {"name":"sun","type":"light","shape":"sun","energy":3}]);
            scene["nodes"][1]["materials"]=serde_json::json!(["coat"]);
            scene["programs"]=serde_json::json!({"coat":{"source":format!("import vork; struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs i) {{Closure c={{}};c.base_weight=0.;c.specular_weight=0.;c.coat_weight=1.;c.coat_roughness=.3;c.coat_ior=1.5;{coat_normal}return c;}}")}});
            scene["materials"]=serde_json::json!({"coat":{"program":"coat"}});
            crate::path_medium_tests::mean_color(&r,scene)
        };
        for integrator in ["raster","path"] {
            let unset=render(integrator,"")?;
            let unperturbed=render(integrator,"c.geometry_coat_normal=ctx.NORMAL;")?;
            // 30 degrees off the half vector: far outside a roughness .3 GGX lobe.
            let tilted=render(integrator,"c.geometry_coat_normal=ctx.NORMAL*cos(radians(30.))+float3(1.,0.,0.)*sin(radians(30.));")?;
            eprintln!("{integrator}: unset {unset:?} unperturbed {unperturbed:?} tilted {tilted:?}");
            for channel in 0..3 {
                assert!(unset[channel]>0.05,"{integrator}: no coat highlight {unset:?}");
                assert!((unset[channel]-unperturbed[channel]).abs()<=1e-3*unset[channel],"{integrator}: unset {unset:?} vs unperturbed {unperturbed:?}");
                assert!(tilted[channel]<0.5*unset[channel],"{integrator}: tilted {tilted:?} vs unset {unset:?}");
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn camera_programs_deduplicate_and_keep_independent_content() -> Result<()> {
        let r = crate::test_renderer()?;
        // Keep the compile-count assertion independent of an earlier test process's disk cache.
        let source=format!("{}\n// independence test {}", "import vork; struct Inputs {float value=0.25;}; Closure surface(ShadingContext ctx,Inputs i) {Closure c;c.base_color=float3(i.value);return c;}",std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos());
        let mut scene:v2::Scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[],
            "programs":{"a":{"source":source},"duplicate":{"source":source},"z":{"source":source.replace("i.value)","i.value*0.5)")}},
            "materials":{"a":{"program":"a"},"duplicate":{"program":"duplicate","inputs":{"value":0.75}},"z":{"program":"z"}}
        }))?;
        let install=|scene:&v2::Scene|->Result<Programs> {
            let instances=scene.materials.keys().map(|name|Ok(instance(scene,name)?.map(|p|(name.as_str(),p)))).collect::<Result<Vec<_>>>()?;
            Ok(super::install(&r,scene,instances.iter().flatten().map(|(name,p)|(*name,p)),Path::new("."))?.unwrap())
        };
        // Mesh stage discovery must also deduplicate aliases before compiling.
        scene.nodes=vork_scene::v2::from_test_value(serde_json::json!([
            {"type":"mesh","geometry":"unused","materials":["a","duplicate","z"]}
        ]))?;
        let before=crate::timing::creations(&r.device).len();
        let stages=mesh_stages(&r,&scene,Path::new("."))?;
        assert!(stages.values().all(|stage|stage.surface && !stage.medium));
        let creations=crate::timing::creations(&r.device);
        assert_eq!(creations[before..].iter().filter(|c|c.kind=="Slang").count(),2);
        scene.nodes.clear();
        let first=install(&scene)?;
        assert_eq!(first.compiled.len(),2,"identical source and input layouts share a program");
        assert_eq!(first.instances[&instance_key(&instance(&scene,"a")?.unwrap())],first.instances[&instance_key(&instance(&scene,"duplicate")?.unwrap())]);
        let unchanged=first.native.as_ref().unwrap().camera[1].clone();
        scene.programs.get_mut("a").unwrap().source=Some(source.replace("i.value)","i.value*0.8)"));
        let edited=install(&scene)?;
        let index=edited.instances[&instance_key(&instance(&scene,"z")?.unwrap())];
        let after=&edited.native.as_ref().unwrap().camera[index];
        assert_eq!(unchanged.code,after.code);
        assert_eq!(unchanged.modules,after.modules,"editing another program, even renumbering ids, keeps native artifact content");
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn source_support_comes_only_from_surface_emission() -> Result<()> {
        let compiler=Compiler::discover();let abi=abi();
        for (body,lit,expected) in [
            ("return unlit(float3(0.));", "", false),
            ("return unlit(float3(1.));", "", true),
            ("return unlit(float3(10.),0.);", "", false),
            ("return unlit(float3(0.));", "Closure lit(ShadingContext c,Inputs i,Lighting l) {return unlit(float3(10.));}", false),
            ("return unlit(float3(i.strength));", "", true),
        ] {
            let source=format!("struct Inputs {{float strength=0.;}}; Closure surface(ShadingContext c,Inputs i) {{{body}}} {lit}");
            let program=compiler.compile_surface(&abi,&source,Path::new("source-support.slang"),"support")?;
            assert_eq!(program.capabilities.emission,expected,"{source}");
            if body.contains("i.strength") {
                let words=program.inputs.effect_words(&vork_scene::v2::from_test_value(serde_json::json!({"strength":0.}))?,&Default::default())?;
                assert!(!program.effect_probe()?.capabilities(&words).emission);
            }
        }
        for body in ["return blend(unlit(float3(0.)),unlit(float3(1.)),.5);", "return add(unlit(float3(0.)),unlit(float3(c.time)));"] {
            let source=format!("struct Inputs {{}}; Blend surface(ShadingContext c,Inputs i) {{{body}}}");
            let program=compiler.compile_surface(&abi,&source,Path::new("blend-source.slang"),"blendsource")?;
            assert!(program.capabilities.emission);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn source_emission_retains_only_real_time_dependencies() -> Result<()> {
        let compiler=Compiler::discover();let abi=abi();
        for (expression,timed) in [("float3(1.)",false),("float3(1.+ctx.time)",true)] {
            let source=format!("import vork; struct Inputs {{}}; Closure surface(ShadingContext ctx,Inputs inputs) {{return unlit({expression});}}");
            let program=compiler.compile_surface(&abi,&source,Path::new("emission-domain.slang"),"sourceemission")?;
            assert!(program.capabilities.emission);
            assert_eq!(program.capabilities.dependencies.time,timed);
            assert!(!program.capabilities.dependencies.camera);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn documented_stages_compile_against_the_renderer_abi() -> Result<()> {
        // One minimal module per stage the programs help page lists; the page must list each stage.
        const STAGES: &[(&str, &str)] = &[
            ("surface", r#"import vork;
struct Inputs {};
Closure surface(ShadingContext ctx, Inputs inputs) { Closure c = {}; return c; }
"#),
            ("displace", r#"import vork;
struct Inputs { Attribute<float> height; float scale = 0.02; };
float3 displace(DisplacementContext ctx, Inputs inputs) {
    return ctx.NORMAL * read_attribute(inputs.height, ctx) * inputs.scale;
}
Closure surface(ShadingContext ctx, Inputs inputs) { return {}; }
"#),
            ("surface (Hair)", r#"import vork;
struct Inputs {};
Hair surface(ShadingContext ctx, Inputs inputs) { Hair h = {}; return h; }
"#),
            ("surface (Blend)", r#"import vork;
struct Inputs { float weight = .5; };
Blend surface(ShadingContext ctx, Inputs inputs) {
    Closure paint = {}; paint.base_color = float3(.6, .05, .02);
    Closure metal = {}; metal.base_metalness = 1.; metal.specular_roughness = .1;
    return blend(paint, metal, inputs.weight);
}
"#),
            ("aov", r#"import vork;
struct Inputs { Aov mask; };
Closure surface(ShadingContext ctx, Inputs inputs) { Closure c = {}; return c; }
void aov(ShadingContext ctx, Inputs inputs, inout AovOutputs output) {
    write_aov(output, inputs.mask, 1.);
}
"#),
            ("lit", r#"import vork;
struct Inputs {};
Closure surface(ShadingContext ctx, Inputs inputs) { Closure c = {}; return c; }
Closure lit(ShadingContext ctx, Inputs inputs, Lighting light) {
    return unlit(dot(light.color, float3(.2126, .7152, .0722)) < .2 ? float3(.1) : float3(.8));
}
"#),
            ("medium", r#"import vork;
struct Inputs {};
MediumCoefficients medium(MediumContext ctx, Inputs inputs) { MediumCoefficients m = {}; return m; }
"#),
            ("environment", r#"import vork;
struct Inputs {};
float3 environment(EnvironmentContext ctx, Inputs inputs) { return float3(.1); }
"#),
            ("composite", r#"import vork;
struct Inputs {};
float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_sample(ctx.uv); }
"#),
            ("view", r#"import vork;
struct Inputs {};
import std.view; ViewResult view(ViewContext ctx, Inputs inputs) { return finish_view(ctx.color / (1. + ctx.color),ctx,1.); }
"#),
            ("compute", r#"import vork;
struct Inputs {};
void compute(ComputeContext ctx, Inputs inputs) {}
"#),
        ];
        let compiler=Compiler::discover();
        let abi=abi();
        let page=include_str!("../../../docs/help/programs.md");
        for (count,(stage,source)) in STAGES.iter().enumerate() {
            let name=stage.split(' ').next().unwrap();
            ensure!(name=="medium" || page.contains(&format!(" {name}(")),"programs help does not list stage {name}");
            let tag=format!("help{count}");
            let origin=PathBuf::from(format!("help-{stage}.slang"));
            match *stage {
                "surface" | "surface (Blend)" | "surface (Hair)" | "aov" | "lit" | "displace" => compiler.compile_surface(&abi,source,&origin,&tag).map(|_|())?,
                "medium" => compiler.compile_medium(&abi,source,&origin,&tag).map(|_|())?,
                "environment" => compiler.compile_environment(&abi,source,&origin,&tag).map(|_|())?,
                "composite" => compiler.compile_composite(&abi,source,&origin,&tag).map(|_|())?,
                "view" => compiler.compile_display(&abi,source,&origin,&tag).map(|_|())?,
                "compute" => compiler.compile_compute(source,&origin,crate::compute::CONTEXT_BINDINGS,1,[1,1,1]).map(|_|())?,
                _ => anyhow::bail!("unknown documented stage {stage}"),
            }
        }
        Ok(())
    }
}

/// Bind reflected numeric words to the same ordered track evaluator as nodes.
/// Resource handles and enums select structure and cannot be interpolated.
pub(crate) fn bind_inputs(
    plan: &mut crate::animation::Plan,
    material: &str,
    layout: &vork_slang::InputLayout,
    words: &mut [u32],
    base: usize,
    offset: f32,
) -> Result<()> {
    bind_input_words(plan, material, layout, words, base, offset, false)
}
fn bind_input_words(
    plan: &mut crate::animation::Plan, material: &str, layout: &vork_slang::InputLayout,
    words: &mut [u32], base: usize, offset: f32, discrete: bool,
) -> Result<()> {
    for (name, input) in plan.material_inputs().into_iter().filter(|(name, _)| name == material) {
        let (index, field) = layout.fields.iter().enumerate().find(|(_, f)| f.name == input)
            .with_context(|| format!("track materials.{name}.inputs.{input}: unknown program input"))?;
        ensure!((discrete || field.enumeration.is_none()) && (matches!(field.ty, vork_slang::Type::Float | vork_slang::Type::Vector(_) | vork_slang::Type::Int | vork_slang::Type::Uint) || (discrete && field.ty == vork_slang::Type::Bool)),
            "track materials.{name}.inputs.{input}: resource handles, booleans and enums are structural program inputs");
        let value: Vec<f32> = words[field.offset..field.offset + field.ty.words()].iter().map(|&w| match field.ty {
            vork_slang::Type::Int => w as i32 as f32,
            vork_slang::Type::Uint => w as f32,
            _ => f32::from_bits(w),
        }).collect();
        let enum_tokens = field.enumeration.as_ref().map(|e| e.cases.iter().map(|c| c.to_ascii_lowercase()).collect::<Vec<_>>());
        let default = if let Some(tokens) = &enum_tokens {
            crate::animation::Sample::String(tokens[words[field.offset] as usize].clone().into())
        } else if field.ty == vork_slang::Type::Bool {
            crate::animation::Sample::Bool(words[field.offset] != 0)
        } else { crate::animation::Sample::Numeric(value) };
        words[index / 32] |= 1 << (index % 32);
        plan.bind_material(&(name, input), vec![crate::animation::MaterialBinding {
            slot: base + field.offset, parameter: usize::MAX, storage: Some(field.ty),
            default, enum_tokens, offset,
        }])?;
    }
    Ok(())
}

pub(crate) fn sample_words(value: &crate::animation::Sample, binding: &crate::animation::MaterialBinding) -> Vec<u32> {
    let ty = binding.storage.unwrap();
    let value = match value {
        crate::animation::Sample::Numeric(value) => value,
        crate::animation::Sample::Bool(value) => return vec![u32::from(*value)],
        crate::animation::Sample::Style(_) => unreachable!("CSS/SVG channels do not bind material inputs"),
                crate::animation::Sample::String(value) => return vec![binding.enum_tokens.as_ref().unwrap().iter().position(|v| v == value.as_ref()).unwrap() as u32],
    };
    value.iter().map(|&v| match ty {
        vork_slang::Type::Int => (v as i32) as u32,
        vork_slang::Type::Uint => v as u32,
        _ => v.to_bits(),
    }).collect()
}

/// Standalone word consumers (environment and compute) retain the common plan,
/// binding only their own instance. There is no alternate interpolation path.
pub(crate) struct InputAnimation {
    plan: crate::animation::Plan,
}
impl InputAnimation {
    /// On its own, reading any media the plan needs itself.
    #[cfg(test)]
    pub(crate) fn new(scene: &v2::Scene, base: &Path, material: &str, layout: &vork_slang::InputLayout, words: &mut [u32]) -> Result<Option<Self>> {
        Self::with_discrete(scene, base, material, layout, words, false, &Default::default())
    }
    /// Within a document open: the plan reads media through the sources that open read.
    pub(crate) fn new_in(scene: &v2::Scene, base: &Path, material: &str, layout: &vork_slang::InputLayout, words: &mut [u32], opened: &crate::footage::Opened) -> Result<Option<Self>> {
        Self::with_discrete(scene, base, material, layout, words, false, opened)
    }
    pub(crate) fn with_discrete(scene: &v2::Scene, base: &Path, material: &str, layout: &vork_slang::InputLayout, words: &mut [u32], discrete: bool, opened: &crate::footage::Opened) -> Result<Option<Self>> {
        if !crate::materials::animated_inputs(scene)?.contains_key(material) { return Ok(None); }
        let mut plan = crate::animation::Plan::compile_in(scene, base, None, opened)?;
        bind_input_words(&mut plan, material, layout, words, 0, 0., discrete)?;
        Ok(Some(Self { plan }))
    }
    pub(crate) fn identity(&self) -> blake3::Hash { self.plan.input_identity() }
    pub(crate) fn words_request(&self, request:&vork_time::Request, time:f32, words:&mut [u32])->Result<()> {
        let mut samples=self.plan.samples();
        self.plan.sample_material_request(request,time,0.,&mut samples)?;
        for (binding,value) in self.plan.material_values(&samples,0.) {
            let value=sample_words(value,binding);
            words[binding.slot..binding.slot+value.len()].copy_from_slice(&value);
        }
        Ok(())
    }
    pub(crate) fn words(&self, time: f32, words: &mut [u32]) {
        let mut samples = self.plan.samples();
        self.plan.sample_material_offset(time, 0., &mut samples);
        for (binding, value) in self.plan.material_values(&samples, 0.) {
            let value = sample_words(value, binding);
            words[binding.slot..binding.slot + value.len()].copy_from_slice(&value);
        }
    }
}

#[cfg(test)]
mod bundled_origin_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang"]
    fn mixed_surface_compute_installs_bugbash_file() -> Result<()> {
        let base=Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mixed-programs");
        for name in ["scene.json","geometry-scene.json"] {
            Renderer::check(&base.join(name),&vork_scene::Profile::load("std/final")?,false)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn mixed_surface_compute_discovers_and_compiles_surface() -> Result<()> {
        // BB039/BB055: pins.slang combines surface(), compute(), typed buffer
        // accessors and a Geometry input in one authored module.
        let mut scene = crate::test_scene(serde_json::json!({
            "version":2,"nodes":[{"type":"camera","name":"camera"}],"render":{},
            "buffers":{"p":{"type":"vec3","data":[[0.,0.,0.]]}},
            "programs":{"mixed":{"source":"struct Inputs { Geometry paddle; }; Closure surface(ShadingContext c, Inputs i){return unlit(float3(.2,.4,.6));} void compute(ComputeContext c, Inputs i){positions_store(c.id.x,positions_load(c.id.x)+geometry_POSITION_load(i.paddle,0u));}","bindings":{"positions":"p"},"dispatch":1,"schedule":"install"}}
        }),Path::new("."))?;
        let (source,origin)=material_source(&scene,Path::new("."),"mixed")?;
        let compiler=Compiler::discover();
        compiler.prepare_program(&abi(),&source,&origin)?;
        let (stages,_)=compiler.stages(&abi(),&source,&origin)?;
        assert!(stages.contains("surface") && stages.contains("compute"));
        compiler.compile_surface(&abi(),&source,&origin,"mixedsurface")?;
        let mut alias=scene.programs["mixed"].clone();
        alias.dispatch=None;
        alias.schedule=None;
        alias.bindings.clear();
        scene.programs.insert("surface_alias".into(),alias);
        let (source,origin)=material_source(&scene,Path::new("."),"surface_alias")?;
        compiler.compile_surface(&abi(),&source,&origin,"mixedalias")?;
        let invalid=source.replace("float3(.2,.4,.6)","positions_load(0u)");
        assert!(compiler.compile_surface(&abi(),&invalid,&origin,"invalidsurface").is_err(),
            "compute-only accessors must not acquire surface implementations");
        Ok(())
    }


    #[test]
    fn launch_analysis_keeps_failures_in_traversal_order() -> Result<()> {
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let mut scene = crate::test_scene(serde_json::json!({
            "version":2,
            "programs":{"a":{"file":"missing-second.slang"},"z":{"file":"missing-first.slang"}},
            "materials":{"first":{"program":"z"},"second":{"program":"a"}},
            "nodes":[{"type":"mesh","geometry":{"std":"rounded_box"},"materials":["first"]},
                {"type":"node","children":[{"type":"mesh","geometry":{"std":"rounded_box"},"materials":["second"]}]},{"type":"camera"}]
        }), Path::new("."))?;
        let stages = std::collections::BTreeMap::new();
        assert_eq!(launch_analysis_programs(&scene, &stages, false), ["a", "z"]);
        let error = normalize_launch_bindings(&renderer, &mut scene, Path::new("."), &stages).unwrap_err();
        assert!(format!("{error:#}").contains("missing-first.slang"), "{error:#}");
        Ok(())
    }

    #[test]
    fn launch_analysis_shares_programs_and_preserves_warning_order() -> Result<()> {
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let mut scene = crate::test_scene(serde_json::json!({
            "version":2,
            "programs":{
                "hair":{"source":"struct Inputs {}; Hair surface(ShadingContext c, Inputs i) {Hair h={};return h;}"},
                "unused":{"file":"must-not-read.slang"}},
            "materials":{"one":{"program":"hair"},"two":{"program":"hair"},"unused":{"program":"unused"}},
            "nodes":[{"type":"mesh","name":"one","geometry":{"std":"rounded_box"},"materials":["one"]},
                {"type":"mesh","name":"two","geometry":{"std":"rounded_box"},"materials":["two"]},{"type":"camera"}]
        }), Path::new("."))?;
        let stages = std::collections::BTreeMap::new();
        assert_eq!(launch_analysis_programs(&scene, &stages, false), ["hair"]);
        let mut curves = scene.clone();
        for geometry in curves.geometry.values_mut() { geometry.topology = v2::Topology::Curves; }
        assert!(launch_analysis_programs(&curves, &stages, false).is_empty());
        assert_eq!(launch_analysis_programs(&curves, &stages, true), ["hair"]);
        let media = ["one", "two"].into_iter().map(|name| (name.to_owned(), Stages {medium:true, ..Default::default()})).collect();
        assert!(launch_analysis_programs(&scene, &media, false).is_empty());
        let collector = crate::warnings::Collector::default();
        let _scope = collector.enter();
        normalize_launch_bindings(&renderer, &mut scene, Path::new("."), &stages)?;
        let warnings = collector.snapshot();
        let dropped = warnings.iter().filter(|w| w.code == "surface_geometry").collect::<Vec<_>>();
        assert_eq!(dropped.len(), 2);
        assert_eq!(dropped[0].requested["node"], "one");
        assert_eq!(dropped[1].requested["node"], "two");
        for node in &scene.nodes {
            let v2::NodeKind::Mesh { materials, .. } = &node.kind else { continue };
            assert_eq!(scene.materials[&materials[0]].program, "std/openpbr");
        }
        Ok(())
    }

    #[test]
    fn launch_analysis_preserves_static_and_animated_implicit_inputs() -> Result<()> {
        let renderer = pollster::block_on(Renderer::create(crate::DeviceNeeds::ALL, true))?;
        let mut scene = crate::test_scene(serde_json::json!({
            "version":2,
            "buffers":{"field":{"type":"f32","dims":[2,2,2]}},
            "geometry":{"field":{"topology":"implicit","attributes":{"phi":"field"}}},
            "programs":{"skin":{"source":"struct Inputs {float weight=0.;}; Closure surface(ShadingContext c, Inputs i) {Closure o={};o.subsurface_weight=i.weight;return o;}"}},
            "materials":{"zero":{"program":"skin","inputs":{"weight":0}},
                "scattering":{"program":"skin","inputs":{"weight":1}},
                "animated":{"program":"skin","inputs":{"weight":0}}},
            "nodes":[{"type":"mesh","geometry":"field","materials":["zero"]},
                {"type":"mesh","geometry":"field","materials":["scattering"]},
                {"type":"mesh","geometry":"field","materials":["animated"]},{"type":"camera"}],
            "animation":[{"tracks":[{"path":["materials","animated","inputs","weight"],"keys":[[0,0],[1,1]],"interpolation":"LINEAR"}]}]
        }), Path::new("."))?;
        let stages = std::collections::BTreeMap::new();
        assert_eq!(launch_analysis_programs(&scene, &stages, false), ["skin"]);
        normalize_launch_bindings(&renderer, &mut scene, Path::new("."), &stages)?;
        for (index, node) in scene.nodes.iter().enumerate() {
            let v2::NodeKind::Mesh { materials, .. } = &node.kind else { continue };
            assert_eq!(scene.materials[&materials[0]].program, if index == 0 { "skin" } else { "std/openpbr" });
        }
        Ok(())
    }

    #[test]
    fn admission_parallel_map_preserves_results_and_propagates_errors() -> Result<()> {
        assert_eq!(parallel_map(&[3, 1, 2], |n| Ok(n * 2))?, [6, 2, 4]);
        assert!(parallel_map::<u32, u32>(&[], |_| unreachable!())?.is_empty());
        let error = parallel_map(&[0, 1, 2], |n| {
            anyhow::ensure!(*n != 1, "authored surface compilation failed");
            Ok(*n)
        }).unwrap_err();
        assert!(error.to_string().contains("authored surface compilation failed"));
        Ok(())
    }

    #[test]
    fn launch_analysis_worker_limit_and_result_order() -> Result<()> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let active = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let workers = 4.min(std::thread::available_parallelism().map_or(1, usize::from));
        let barrier = std::sync::Barrier::new(workers);
        let items = (0..12).collect::<Vec<_>>();
        let results = parallel_map_with_limit(&items, 4, |i| {
            let count = active.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(count, Ordering::SeqCst);
            barrier.wait();
            active.fetch_sub(1, Ordering::SeqCst);
            barrier.wait();
            Ok(if *i == 3 { Err("third") } else { Ok(*i) })
        })?;
        assert_eq!(peak.load(Ordering::SeqCst), workers);
        assert_eq!(results[3], Err("third"));
        assert_eq!(results[11], Ok(11));
        Ok(())
    }

    #[test]
    fn workers_report_every_failure_and_finish_all_inputs() {
        let visited = std::sync::Mutex::new(Vec::new());
        let items = (0..9).collect::<Vec<_>>();
        let error = parallel_map_with_limit(&items, 2, |index| -> Result<()> {
            visited.lock().unwrap().push(*index);
            anyhow::bail!("broken program {index}")
        }).unwrap_err();
        let message = format!("{error:#}");
        assert_eq!(visited.lock().unwrap().len(), items.len());
        for index in items { assert!(message.contains(&format!("broken program {index}")), "{message}"); }
    }

    #[test]
    fn program_diagnostics_keep_authored_filename_and_line() {
        let error = program_key(b"diagnostic", &abi(),
            "struct Inputs {};\nClosure surface(ShadingContext c, Inputs i) {return missing_author_symbol;}\n",
            Path::new("author-palette.slang")).err().expect("invalid program");
        let message = format!("{error:#}");
        assert!(message.contains("author-palette.slang:2:"), "{message}");
        assert!(!message.contains("material.slang"), "{message}");
    }

    #[test]
    fn display_proof_cache_tracks_shadowing_and_real_edits() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-display-proof-{}",std::process::id()));
        let _=std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("sub"))?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let compiler=Compiler::discover(); let abi=abi();
        let origin=root.join("main.slang");
        let source="import sub.wrapper; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs inputs){return finish_view(float3(wrapped()),ctx,1.);}";
        let helper=root.join("helper.slang"); let sibling=root.join("sub/helper.slang");
        std::fs::write(root.join("sub/wrapper.slang"),"import helper; public float wrapped(){return value();}")?;
        std::fs::write(&helper,"public float value(){return 0.25;}")?;
        // Keep the raw source/options key identical through all dependency edits.
        let key=blake3::hash(b"display-proof-test");
        let build=|compiler:&Compiler|compiler.compile_display(&abi,source,&origin,"cachetest");
        let first=cached_proof(&store,key,&compiler,&abi,source,&origin,build)?;
        let hit:vork_slang::DisplayProgram=cached_proof(&store,key,&compiler,&abi,source,&origin,|_|anyhow::bail!("unchanged inputs recompiled"))?;
        assert_eq!(hit.wgsl,first.wgsl);
        std::fs::write(&sibling,"public float value(){return 0.75;}")?;
        let shadowed=cached_proof(&store,key,&compiler,&abi,source,&origin,build)?;
        assert_ne!(first.wgsl,shadowed.wgsl,"a new preferred import must never reuse stale WGSL");
        std::fs::write(&sibling,"public float value(){return 0.5;}")?;
        let edited=cached_proof(&store,key,&compiler,&abi,source,&origin,build)?;
        assert_ne!(shadowed.wgsl,edited.wgsl,"real edits must never reuse stale WGSL");
        std::fs::remove_file(&sibling)?;
        let restored=cached_proof(&store,key,&compiler,&abi,source,&origin,build)?;
        assert_eq!(restored.wgsl,first.wgsl);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn display_proof_cache_rejects_publication_races() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-display-race-{}",std::process::id()));
        let _=std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root)?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let compiler=Compiler::discover(); let abi=abi();
        let origin=root.join("main.slang"); let sibling=root.join("helper.slang");
        let source="import helper; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs inputs){return finish_view(float3(value()),ctx,1.);}";
        std::fs::write(&sibling,"public float value(){return 0.5;}")?;
        let build=|compiler:&Compiler|compiler.compile_display(&abi,source,&origin,"cachetest");
        let race_key=blake3::hash(b"display-publication-race");
        let raced:Result<vork_slang::DisplayProgram>=cached_proof(&store,race_key,&compiler,&abi,source,&origin,|compiler| {
            let program=build(compiler)?;
            // Deterministically edit after compiler reads but before publication.
            std::fs::write(&sibling,"public float value(){return 0.875;}")?;
            Ok(program)
        });
        assert!(raced.is_err(),"old output must not be associated with new bytes");
        let mut rebuilt=false;
        let fresh=cached_proof(&store,race_key,&compiler,&abi,source,&origin,|compiler| {rebuilt=true;build(compiler)})?;
        assert!(rebuilt,"a raced compilation must not publish a cache entry");
        assert!(fresh.wgsl.contains("0.875"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn removed_modules_never_publish_proofs() -> Result<()> {
        let root = std::env::temp_dir().join(format!("vork-removed-proof-{}", std::process::id()));
        let store = disk_cache::Store::for_test(root.join("cache"));
        let compiler = Compiler::discover();
        let abi = abi();
        for (module, replacement) in [("post.white_balance", "std.white_balance"), ("std.particles", "fluid.particles")] {
            let source = format!("import {module}; struct Inputs {{}}; import std.view; ViewResult view(ViewContext c, Inputs i) {{return finish_view(float3(0.25),c,1.);}}");
            for _ in 0..2 {
                let result: Result<vork_slang::DisplayProgram> = cached_proof(&store, blake3::hash(source.as_bytes()), &compiler, &abi, &source, &root.join("main.slang"), |_| {
                    panic!("removed imports must refuse before building a proof")
                });
                let error = format!("{:#}", result.err().expect("removed module accepted"));
                assert_eq!(error.lines().count(), 1, "{error}");
                assert!(error.contains(replacement) && error.contains("vork help upgrade"), "{error}");
            }
        }
        if root.exists() { std::fs::remove_dir_all(root)?; }
        Ok(())
    }

    #[test]
    fn display_proof_cache_rejects_unidentified_imports() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-proof-extra-inputs-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("helper.slang"),"public float value(){return 0.75;}")?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let compiler=Compiler::discover(); let abi=abi(); let origin=root.join("main.slang");
        let source="struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i){return finish_view(float3(0.25),ctx,1.);}";
        let actual="import helper; struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs i){return finish_view(float3(value()),ctx,1.);}";
        let key=blake3::hash(b"proof-extra-inputs");
        let result=cached_proof(&store,key,&compiler,&abi,source,&origin,|compiler|
            compiler.compile_display(&abi,actual,&origin,"extra"));
        assert!(result.is_err(),"a wrapper-only dependency must not escape the proof key");
        let mut rebuilt=false;
        let fresh=cached_proof(&store,key,&compiler,&abi,source,&origin,|compiler| {
            rebuilt=true;compiler.compile_display(&abi,source,&origin,"extra")
        })?;
        assert!(rebuilt,"an incomplete identity must not publish an artifact");
        assert!(fresh.wgsl.contains("0.25"));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn embedded_proof_cache_remains_portable_across_origins() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-portable-proof-{}",std::process::id()));
        let store=disk_cache::Store::for_test(root.clone());
        let key=blake3::hash(b"portable-display-proof");
        let source="struct Inputs {}; import std.view; ViewResult view(ViewContext ctx, Inputs inputs){return finish_view(ctx.color,ctx,1.);}";
        let abi=abi(); let compiler=Compiler::discover();
        let first_origin=root.join("setup/embedded.slang");
        let other_origin=root.join("scene/embedded.slang");
        std::fs::create_dir_all(first_origin.parent().unwrap())?;
        std::fs::create_dir_all(other_origin.parent().unwrap())?;
        let first=cached_proof(&store,key,&compiler,&abi,source,&first_origin,|compiler|
            compiler.compile_display(&abi,source,&first_origin,"portable"))?;
        let hit:vork_slang::DisplayProgram=cached_proof(&store,key,&compiler,&abi,source,&other_origin,
            |_|anyhow::bail!("embedded proof lost portability"))?;
        assert_eq!(first.wgsl,hit.wgsl);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn all_authored_stages_use_checked_cache_boundary() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-stage-smoke-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let compiler=Compiler::discover();let abi=abi();
        macro_rules! stage {($name:literal,$source:expr,$method:ident)=>{{
            let source=$source;let origin=root.join(concat!($name,".slang"));
            let key=blake3::hash($name.as_bytes());
            let _:serde_json::Value=cached_proof(&store,key,&compiler,&abi,source,&origin,|compiler|
                Ok(serde_json::to_value(compiler.$method(&abi,source,&origin,"stagesmoke")?)?))?;
            let _:serde_json::Value=cached_proof(&store,key,&compiler,&abi,source,&origin,
                |_|anyhow::bail!("unchanged {} stage recompiled",$name))?;
        }}}
        stage!("surface","struct Inputs {}; Closure surface(ShadingContext c,Inputs i){Closure o={};o.base_color=float3(0.3);return o;}",compile_surface);
        stage!("camera",vork_slang::module_source("std/camera").unwrap(),compile_camera);
        stage!("environment","struct Inputs {}; float3 environment(EnvironmentContext c,Inputs i){return float3(c.direction.y);}",compile_environment);
        stage!("medium",include_str!("../../vork-scene/tests/fixtures/medium.slang"),compile_medium);
        stage!("composite","struct Inputs {float gain=1.;}; float4 composite(CompositeContext c,Inputs i){return compositor_sample(c.uv)*i.gain;}",compile_composite);
        stage!("propagation","struct Inputs {}; PropagationStep propagation(PropagationContext c,Inputs i){PropagationStep s;s.direction=c.direction;s.distance=1.;return s;}",compile_propagation);
        let source="struct Inputs {uint amount=3;}; void compute(ComputeContext c,Inputs i){result_store(c.id.x,i.amount);}";
        let bindings="struct Tick {uint frame;float time;float dt;uint reserved;}; [[vk::binding(0,0)]] ConstantBuffer<Tick> vork_tick; struct Offset {uint4 value;}; [[vk::binding(1,0)]] ConstantBuffer<Offset> vork_offset; [[vk::binding(2,0)]] RWStructuredBuffer<uint> result; void result_store(uint i,uint v){result[i]=v;}";
        let origin=root.join("compute.slang");let key=blake3::hash(b"compute");
        let inputs=bound_program_identity(&store,&compiler,&abi,source,&origin,bindings)?;
        let compiler=compiler.with_identity(&inputs);
        let _:vork_slang::ComputeProgram=cached_proof(&store,key,&compiler,&abi,source,&origin,|compiler|
            compiler.compile_compute(source,&origin,bindings,64,[1,1,1]))?;
        let _:vork_slang::ComputeProgram=cached_proof(&store,key,&compiler,&abi,source,&origin,
            |_|anyhow::bail!("unchanged compute stage recompiled"))?;
        assert!(compiler.prepare_bound_program(&abi,"struct Inputs {}; uint bad=vork_tick.frame;",&origin,bindings).is_err(),"generated bindings must not relax authored reserved-name checks");
        std::fs::remove_dir_all(root)?;Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn closed_program_lookup_preserves_non_utf8_origins() -> Result<()> {
        use std::os::unix::ffi::OsStringExt;
        let root=std::env::temp_dir().join(format!("vork-closed-origin-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let origin=root.join(std::ffi::OsString::from_vec(b"material\xff.slang".to_vec()));
        let compiler=Compiler::discover();let abi=abi();
        let source="struct Inputs {}; Closure surface(ShadingContext c,Inputs i){Closure o={};return o;}";
        let expected=compiler.prepare_program(&abi,source,&origin)?.receipt();
        let store=disk_cache::Store::for_test(root.join("cache"));
        let actual=program_identity(&store,&compiler,&abi,source,&origin)?;
        assert_eq!(actual.identity,expected.identity);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn semantic_program_aliases_keep_ids_when_only_comments_change() -> Result<()> {
        let compiler=Compiler::discover(); let abi=abi(); let origin=Path::new("material.slang");
        let source="struct Inputs {}; Closure surface(ShadingContext ctx,Inputs i){Closure c={};c.base_color=float3(0.25);return c;}";
        let base=compiler.prepare_program(&abi,source,origin)?;
        let comment=compiler.prepare_program(&abi,&format!("// alias description\n{source}"),origin)?;
        let edited=compiler.prepare_program(&abi,&source.replace("0.25","0.75"),origin)?;
        assert_eq!(semantic_indices([base.identity(),comment.identity(),edited.identity(),base.identity()].into_iter()),vec![0,0,1,0]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_compiler_that_is_not_the_pinned_one_is_refused_before_any_cache_decision() -> Result<()> {
        // audit-3d 9: receipt and proof reuse answered before the version check, so a wrong
        // compiler was refused only on a miss. `echo` runs and reports no Slang version.
        let root=std::env::temp_dir().join(format!("vork-compiler-admission-{}",std::process::id()));
        let _=std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root)?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let cache=|key|store.path("slang-variant",1,key);
        let compiler=Compiler::new("/bin/echo");
        let refused=|error:Option<anyhow::Error>,path:&str| {
            let error=error.unwrap_or_else(||panic!("{path} accepted an unadmitted compiler"));
            assert!(error.downcast_ref::<vork_slang::CompilerUnavailable>().is_some(),"{path}: {error:#}");
        };
        refused(admit(&cache,&compiler).err(),"admit");
        let modules=[("main","[numthreads(1,1,1)] void main(){}")];
        let entries=[("main","compute")];
        let args=["-target","spirv"];
        // The lookup that never compiles: on an empty store it used to answer "a miss".
        refused(cached_native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[]).err(),"scheduling lookup");
        refused(native_input_receipt_path(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[]).err(),"receipt path");
        refused(native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[]).err(),"native identity");
        let abi=abi();
        let source="struct Inputs {}; float3 environment(EnvironmentContext c,Inputs i){return float3(c.direction.y);}";
        let origin=root.join("environment.slang");
        refused(bound_program_identity(&store,&compiler,&abi,source,&origin,"").err(),"program identity");
        let key=blake3::hash(b"admission");
        let proof:Result<serde_json::Value>=cached_proof(&store,key,&compiler,&abi,source,&origin,
            |_|anyhow::bail!("an unadmitted compiler reached the build"));
        refused(proof.err(),"proof");
        // The warm case: the pinned compiler leaves a receipt and a proof in the store. The
        // other executable, handed that valid receipt, would be answered from the proof.
        let pinned=Compiler::discover();
        let receipt=bound_program_identity(&store,&pinned,&abi,source,&origin,"")?;
        let proved:serde_json::Value=cached_proof(&store,key,&pinned.with_identity(&receipt),&abi,source,&origin,|compiler|
            Ok(serde_json::to_value(compiler.compile_environment(&abi,source,&origin,"admission")?)?))?;
        let again:serde_json::Value=cached_proof(&store,key,&pinned.with_identity(&receipt),&abi,source,&origin,
            |_|anyhow::bail!("the stored proof was not reused"))?;
        assert_eq!(proved,again);
        let warm:Result<serde_json::Value>=cached_proof(&store,key,&compiler.with_identity(&receipt),&abi,source,&origin,
            |_|anyhow::bail!("an unadmitted compiler reached the build"));
        refused(warm.err(),"stored proof with a supplied receipt");
        // A refused toolchain is never recorded as admitted. (Which store holds the pinned
        // one's record depends on the test that first checked it in this process.)
        use vork_slang::AdmissionLedger as _;
        let toolchain=compiler.toolchain()?;
        let by_input=Admissions(&|key|store.path("slang-input",1,key));
        let by_variant=Admissions(&cache);
        // Every kind of lookup in one store names the same admission entry.
        assert_eq!(by_input.entry(toolchain),by_variant.entry(toolchain));
        let entry=by_input.entry(toolchain).context("admission entry")?;
        assert!(!by_input.admitted(&entry,toolchain));
        // An admission written through one kind is read through the other.
        let pinned_toolchain=pinned.toolchain()?;
        let entry=by_variant.entry(pinned_toolchain).context("admission entry")?;
        by_variant.record(&entry,pinned_toolchain);
        assert!(by_input.admitted(&entry,pinned_toolchain));
        std::fs::remove_dir_all(&root)?;
        Ok(())
    }

    #[test]
    fn native_store_reuses_comments_and_rebuilds_imported_values() -> Result<()> {
        for target in ["spirv", "metal"] {
        let root=std::env::temp_dir().join(format!("vork-program-identity-store-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let cache=|key|store.path("slang-variant",1,key);
        let helper=root.join("helper.slang");
        std::fs::write(&helper,"public float value(){return 0.25;}")?;
        let main="import helper; RWStructuredBuffer<float> result; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){result[id.x]=value();}";
        let compiler=Compiler::discover();
        let include=root.display().to_string();
        let args=["-target",target,"-I",include.as_str()];
        let entries=[("main","compute")];
        let modules=[("main",main)];
        assert!(cached_native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?.is_none(),"cold scheduling probe must only report a miss");
        let (original,prepared)=native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?;
        // Even disjoint resolver closures must distinguish a native receipt.
        let mut other = original.clone();
        other.identity = b"different installed program".to_vec();
        let receipt_key = |args: &[&str], input: &vork_slang::InputIdentity|
            native_input_receipt_path(&cache,&compiler,&modules,"main",&entries,args,&Default::default(),std::slice::from_ref(input));
        assert_ne!(receipt_key(&args,&original)?,receipt_key(&args,&other)?);
        let cwd = std::env::current_dir()?.to_string_lossy().into_owned();
        assert_eq!(receipt_key(&["-target",target,"-I","."],&original)?,receipt_key(&["-target",target,"-I",&cwd],&original)?);
        let binary=prepared.context("first input must be captured")?.compile()?;
        let artifact=cache(blake3::hash(&original.identity));
        disk_cache::write(&artifact,&binary)?;
        let (_,prepared)=native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?;
        ensure!(prepared.is_none(),"exact unchanged input should reuse its receipt");
        assert_eq!(cached_native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?.unwrap().identity,original.identity);
        // A complete but stale receipt used to abort on an artifact miss.
        let receipt_path = native_input_receipt_path(&cache, &compiler, &modules, "main", &entries, &args, &Default::default(), &[])?;
        let mut stale = original.clone();
        stale.identity = b"stale normalized identity".to_vec();
        write_input_receipt(&receipt_path, &stale)?;
        std::fs::remove_file(&artifact)?;
        let (lookup, prepared) = native_input_identity(&cache, &compiler, &modules, "main", &entries, &args, &Default::default(), &[])?;
        assert_eq!(lookup.identity, stale.identity);
        let recovered = native_code(&lookup, prepared, &[], &receipt_path,
            |identity| cache(blake3::hash(identity)),
            || compiler.prepare_identified_modules(&modules, "main", &entries, &args, &Default::default()),
            |prepared| prepared.compile())?;
        assert_eq!(recovered, binary);
        assert_eq!(read_native_input_receipt(&receipt_path).unwrap().identity, original.identity);
        assert!(disk_cache::read(&cache(blake3::hash(&stale.identity))).is_none(), "fresh code must never be published under stale identity");
        // Both receipt and native artifact truncation are misses that heal.
        std::fs::OpenOptions::new().write(true).open(&receipt_path)?.set_len(17)?;
        std::fs::OpenOptions::new().write(true).open(&artifact)?.set_len(17)?;
        let (lookup, prepared) = native_input_identity(&cache, &compiler, &modules, "main", &entries, &args, &Default::default(), &[])?;
        assert!(prepared.is_some());
        assert_eq!(native_code(&lookup, prepared, &[], &receipt_path,
            |identity| cache(blake3::hash(identity)),
            || anyhow::bail!("captured snapshot must be retained"), |prepared| prepared.compile())?, binary);
        std::fs::write(&helper,"// explanatory comment\npublic float value() { return 0.25; }")?;
        assert!(cached_native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?.is_none(),"changed resolver input must not be reported stored");
        let (comment,prepared)=native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?;
        ensure!(prepared.is_some(),"changed bytes require compiler revalidation");
        assert_eq!(disk_cache::read(&cache(blake3::hash(&comment.identity))),Some(binary.clone()),"comment must hit the existing artifact");
        std::fs::write(&helper,"public float value(){return 0.75;}")?;
        let (edited,prepared)=native_input_identity(&cache,&compiler,&modules,"main",&entries,&args,&Default::default(),&[])?;
        assert!(disk_cache::read(&cache(blake3::hash(&edited.identity))).is_none(),"real imported edit must miss the store");
        assert_ne!(binary,prepared.context("edited input must be captured")?.compile()?);
        std::fs::remove_dir_all(root)?;
        }
        Ok(())
    }

    #[test]
    fn closed_native_inputs_relocate_without_skipping_import_precedence() -> Result<()> {
        let root=std::env::temp_dir().join(format!("vork-closed-native-{}",std::process::id()));
        let first=root.join("first");let second=root.join("second");
        std::fs::create_dir_all(&first)?;std::fs::create_dir_all(&second)?;
        let store=disk_cache::Store::for_test(root.join("cache"));
        let cache=|key|store.path("slang-variant",1,key);
        let compiler=Compiler::discover();
        let first_text=first.to_string_lossy();let second_text=second.to_string_lossy();
        let a=["-target","spirv","-I",first_text.as_ref()];
        let b=["-target","spirv","-I",second_text.as_ref()];
        let modules=[("main","import helper; RWStructuredBuffer<float> output; [numthreads(1,1,1)] void main(uint3 id:SV_DispatchThreadID){output[id.x]=value();}"),
            ("helper","public float value(){return 0.25;}")];
        let entries=[("main","compute")];let identities=Default::default();
        let (before,prepared)=native_input_identity(&cache,&compiler,&modules,"main",&entries,&a,&identities,&[])?;
        assert!(prepared.is_some());
        let (after,prepared)=native_input_identity(&cache,&compiler,&modules,"main",&entries,&b,&identities,&[])?;
        assert!(prepared.is_none(),"closed request must use the relocated receipt");
        assert_eq!(before.identity,after.identity);
        let fresh=compiler.prepare_identified_modules(&modules,"main",&entries,&b,&identities)?.receipt();
        assert_eq!(fresh.identity,after.identity,"relocation must match fresh compiler capture");
        assert!(after.valid());
        let changed=[modules[0],("helper","public float value(){return 0.5;}")];
        assert!(cached_native_input_identity(&cache,&compiler,&changed,"main",&entries,&b,&identities,&[])?.is_none());
        // Slang probes serialized modules ahead of the supplied source. Adding
        // one invalidates both the local and the portable receipt.
        let (closed_path,roots)=native_closed_input_receipt(&cache,&compiler,&modules,"main",&entries,&b,&identities,&[])?;
        assert!(read_closed_input_receipt(&closed_path,&roots).is_some());
        std::fs::write(second.join("helper.slang-module"),b"not a valid module")?;
        assert!(cached_native_input_identity(&cache,&compiler,&modules,"main",&entries,&b,&identities,&[])?.is_none());
        assert!(read_closed_input_receipt(&closed_path,&roots).is_none());
        std::fs::remove_dir_all(root)?;Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn setup_surface_artifacts_are_reused_from_another_scene_directory() -> Result<()> {
        let r = crate::test_renderer()?;
        let scene:v2::Scene=serde_json::from_value(serde_json::json!({"version":2,"nodes":[]}))?;
        let (source,origin)=material_source(&scene,Path::new("/setup"),"std/hair")?;
        let abi=abi();
        let first=compile(&r,&Compiler::discover(),&abi,&source,&origin)?;
        let (source,origin)=material_source(&scene,Path::new("/scenes/shot"),"std/hair")?;
        // Probe the store exactly as `compile` reads it: same identity, same key.
        let compiler=Compiler::discover();
        let identity=program_identity(&r.disk_cache,&compiler,&abi,&source,&origin)?;
        let second:Program=cached(&r,compile_key(&abi,&identity)?,&compiler.with_identity(&identity),&abi,&source,&origin,
            |_|anyhow::bail!("bundled hair recompiled in another scene directory"))?;
        assert_eq!(first.slang,second.slang);
        assert_eq!(first.wgsl,second.wgsl);
        Ok(())
    }

    #[test]
    fn bundled_material_origins_are_independent_of_the_scene_directory() -> Result<()> {
        let scene:v2::Scene=serde_json::from_value(serde_json::json!({"version":2,"nodes":[]}))?;
        for name in ["std/hair"] {
            let a=material_source(&scene,Path::new("/setup"),name)?;
            let b=material_source(&scene,Path::new("/scenes/shot"),name)?;
            assert_eq!(a,b);
            assert_eq!(a.1,PathBuf::from(format!("{name}.slang")));
            assert_eq!(program_key(b"program-v2",&abi(),&a.0,&a.1)?.finalize(),
                program_key(b"program-v2",&abi(),&b.0,&b.1)?.finalize());
        }
        let program:v2::Program=serde_json::from_value(serde_json::json!({"source":"import local;"}))?;
        let a=program_source(Path::new("/one"),"custom",&program)?;
        let b=program_source(Path::new("/two"),"custom",&program)?;
        assert_ne!(a.1.parent(),b.1.parent(),"authored relative imports retain their scene directory");
        Ok(())
    }
}

#[cfg(all(test, vork_private_tests))]
#[path = "programs_texture_tests_private.rs"]
mod texture_install_tests;
