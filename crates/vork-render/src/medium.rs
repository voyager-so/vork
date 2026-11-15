//! World-medium install; one reflected word buffer and the existing pipeline/cache path.
use super::*;
use ::vork_scene::v2;
use vork_slang::{Compiler, MediumProgram};

/// Sampled transport cannot bound this scene's world medium: the scene's refusal, and the one
/// refusal shader warm-up expects from a library scene.
#[derive(Debug)]
pub(crate) struct SampledRefusal(pub String);
impl std::fmt::Display for SampledRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}
impl std::error::Error for SampledRefusal {}

#[derive(Clone, Default)]
pub(crate) struct Native {
    /// Resource-free point dispatch over the same reflected programs/words.
    pub point: String,
    /// Certified plane specialization; other closed carriers use the boundary BVH.
    pub convex: bool,
    pub work_limit: Option<u32>,
    pub point_work: u32,
    pub world_only: bool,
    pub tracking: bool,
    pub sampled_error: Option<String>,
    pub modules: Vec<(String, String)>,
    pub includes: Vec<PathBuf>,
    pub homogeneous: bool,
    pub time_dependent: bool,
    /// All installed medium instances prove zero scattering for every input pose.
    pub scattering_free: bool,
    pub emission_free: bool,
    pub mesh_sources: bool,
}
impl Native {
    #[cfg(test)]
    pub(crate) fn defines(&self) -> Vec<String> {
        // Numeric oracles use the same installed point/domain capabilities.
        vec![format!("-DVORK_MEDIUM_WORLD_ONLY={}",u8::from(self.world_only))]
    }
}
pub(crate) struct Installed {
    pub native: Native,
    pub words: Vec<u32>,
    pub instances: bounded::Installed,
    pub inputs: Vec<(String, usize, vork_slang::InputLayout, f32)>,
}
pub(crate) fn install(
    r: &Renderer,
    scene: &v2::Scene,
    name: Option<&str>,
    base: &Path,
    stages: &std::collections::BTreeMap<String, programs::Stages>,
) -> Result<Option<Installed>> {
    if stages.values().any(|s| s.medium) {
        ensure!(
            scene.render.root.is_none()
                && scene.images.values().all(|image| image
                    .render
                    .as_ref()
                    .is_none_or(|render| render.root.is_none())),
            "stage 1 finite fog does not support subtree renders; render the whole scene or composite an externally rendered plate"
        );
        fn validate_instances(stages: &std::collections::BTreeMap<String, programs::Stages>, nodes: &[v2::Node]) -> Result<()> {
            for node in nodes {
                if let v2::NodeKind::Mesh {
                    materials,
                    instances: Some(instances),
                    ..
                } = &node.kind
                {
                    ensure!(
                        instances.count.is_none()
                            || !materials.iter().any(|name| stages[name].medium),
                        "stage 1 fog does not support computed carrier instance counts; bake fixed carrier instances or render the volume externally"
                    );
                }
                validate_instances(stages, &node.children)?;
            }
            Ok(())
        }
        validate_instances(stages, &scene.nodes)?;
        let mut installed=bounded::install(r, scene, name, base, stages)?;
        installed.prepare_point_work(scene)?;
        return Ok(Some(installed));
    }
    let Some(name) = name else {return Ok(None);};
    let mut installed = bounded::install(r, scene, Some(name), base, stages)?;
    // Prove termination before a vacuum identity can erase evaluation. An
    // unbounded callback is invalid even when its coefficients appear constant.
    installed.prepare_point_work(scene)?;
    if installed.instances.world_is_vacuum(scene, name)? {return Ok(None);}
    Ok(Some(installed))
}


/// A medium program's compiled module (cached by source).
pub(crate) fn compiled(r: &Renderer, source: &str, origin: &Path) -> Result<MediumProgram> {
    let abi = programs::abi();
    let mut hash = blake3::Hasher::new();
    hash.update(abi.cache_identity()?.as_bytes());
    hash.update(b"native-medium-7");
    hash.update(source.as_bytes());
    hash.update(origin.to_string_lossy().as_bytes());
    let key = hash.finalize();
    let tag = key.to_hex();
    programs::cached(
        r,
        key,
        &Compiler::discover(),
        &abi, &source, &origin,
        |compiler| compiler.compile_medium(&abi, source, origin, &tag[..12]),
    )
}

pub(crate) fn compile(
    r: &Renderer,
    scene: &v2::Scene,
    name: &str,
    base: &Path,
) -> Result<(MediumProgram, Vec<u32>)> {
    let material = scene
        .materials
        .get(name)
        .with_context(|| format!("unknown world medium {name}"))?;
    let (source, origin) = if let Some(source) = vork_slang::module_source(&material.program) {
        (
            source.to_owned(),
            PathBuf::from(format!("{}.slang", material.program)),
        )
    } else {
        let program = scene
            .programs
            .get(&material.program)
            .with_context(|| format!("unknown medium program {}", material.program))?;
        programs::program_source(base, &material.program, program)?
    };
    let abi = programs::abi();
    let mut program = compiled(r, &source, &origin)?;
    if !program.warnings.trim().is_empty() {
        eprintln!("{}", program.warnings.trim());
    }
    let words = programs::pack_inputs(
        &program.inputs,
        &material.inputs.iter().map(|(name, value)| Ok((name.clone(), serde_json::to_value(value)?))).collect::<Result<_>>()?,
        scene,
        base,
        |name, _| bail!("medium resource input {name} is unsupported in stage 1; use scalar/RGB physical coefficient programs, or render the volume externally"),
    )?;
    let dynamic = crate::materials::animated_inputs(scene)?.remove(name).unwrap_or_default();
    let proof = program.inputs.varying_words(&words, &dynamic);
    program.time_dependent=program.time_dependent_for_inputs(&abi,&proof)?;
    program.scattering_free=program.scattering_free_for_inputs(&abi,&proof)?;
    program.emission_free=program.emission_free_for_inputs(&abi,&proof)?;
    if !program.homogeneous && program.homogeneous_for_inputs(&abi,&proof)? {
        program.homogeneous=true;
        // Two kernel variants per module share the reflected material words.
        // Distinct bridge names let homogeneous and varying instances coexist.
        let entry=format!("{}_homogeneous",program.native_point);
        program.native=program.native.replace(&program.native_point,&entry);
        program.native_point=entry;
    }
    if !program.homogeneous {
        program.exponential=program.exponential_for_inputs(&abi,&proof)?;
        program.tail_slab=program.tail_slab_for_inputs(&abi,&proof)?;
        if let Some((a,b,g))=program.exponential {
            // Immutable coefficient specializations must remain distinct even
            // when they originate in the same authored module.
            let suffix=a.into_iter().chain(b).chain(g).map(|x|format!("{:08x}",x.to_bits())).collect::<String>();
            let entry=format!("{}_exponential_{suffix}",program.native_point);
            program.native=program.native.replace(&program.native_point,&entry);program.native_point=entry;
        }
    }
    Ok((program, words))
}

#[cfg(test)]
#[path = "medium_numeric_tests.rs"]
mod numeric_tests;

#[path = "medium_bounded.rs"]
mod bounded;
#[path = "medium_bound.rs"]
pub(crate) mod bound;
pub(crate) use bound::certification_times;

/// Register one reflected point program and adapt world to medium coordinates.
fn native_coefficients(program: &MediumProgram,id:u32,bounded:bool,native:&mut Native) -> String {
    let module = format!("medium_program_{id}");
    native
        .modules
        .push((module.clone(), program.native.clone()));
    for path in &program.native_includes {
        if !native.includes.contains(path) {
            native.includes.push(path.clone());
        }
    }
    let base="context.input_base";
    let point = &program.native_point;
    let position = if bounded {
        "mul(float4(position,1.),context.inverse).xyz"
    } else {
        "position"
    };
    format!("import {module};\nMediumSample volume_{id}_medium_point(float3 position,MediumTransportContext context) {{let m={point}_coefficients({position},frame.scene.x,{base},position,vork_path_wavelength);return MediumSample(m.absorption,m.scattering,m.emission,m.anisotropy);}}\n")
}

/// Numerical tests run the same native modules as production with tiny buffers.
/// Their entry is `numerical_test`: MSL reserves and renames `main`.
#[cfg(test)]
pub(crate) fn test_pipeline(r: &Renderer, source: &str, native: &Native, entries: &[wgpu::BindGroupLayoutEntry], threads: (u32, u32, u32)) -> Result<wgpu::ComputePipeline> {
    test_pipeline_defines(r,source,native,entries,threads,&[])
}
#[cfg(test)]
pub(crate) fn test_pipeline_defines(r: &Renderer, source: &str, native: &Native, entries: &[wgpu::BindGroupLayoutEntry], threads: (u32,u32,u32), extra: &[&str]) -> Result<wgpu::ComputePipeline> {
    let mut owned = programs::abi().native_modules();
    owned[0].1=programs::abi().module();
    owned.extend(native.modules.clone());
    let mut modules = crate::native::SOURCES.to_vec();
    modules.extend(owned.iter().map(|(n,s)| (n.as_str(),s.as_str())));
    // Match wgpu's dense per-resource-class Metal slots even when imported
    // declarations precede the output or an unused resource is eliminated.
    let mut source = source.to_owned();
    let mut sorted: Vec<_> = entries.iter().collect();
    sorted.sort_by_key(|e| e.binding);
    let mut slots = [0u32; 3];
    for entry in sorted {
        let (class, counter) = match entry.ty {
            wgpu::BindingType::Buffer {ty, ..} => (match ty {
                wgpu::BufferBindingType::Uniform => 'b',
                wgpu::BufferBindingType::Storage {read_only:true} => 't',
                wgpu::BufferBindingType::Storage {read_only:false} => 'u',
            }, 0),
            wgpu::BindingType::AccelerationStructure {..} => ('t', 0),
            wgpu::BindingType::Texture {..} => ('t', 1),
            wgpu::BindingType::StorageTexture {..} => ('u', 1),
            wgpu::BindingType::Sampler(_) => ('s', 2),
            _ => anyhow::bail!("unsupported numerical test resource"),
        };
        let annotation = format!("[[vk::binding({},0)]]", entry.binding);
        // Page tables declare a Metal argument buffer and a Vulkan array.
        let mut from = 0;
        while let Some(start) = source[from..].find(&annotation).map(|at| from + at) {
            let end = start + source[start..].find(';').context("numerical resource declaration")?;
            let mut declaration = source[start..end].to_owned();
            if let Some(register) = declaration.find(" : register(") {declaration.truncate(register);}
            let class = if declaration.contains("ParameterBlock<") {'b'} else {class};
            declaration.push_str(&format!(" : register({class}{})", slots[counter]));
            source.replace_range(start..end, &declaration);
            from = start + declaration.len();
        }
        slots[counter] += 1;
    }
    modules.push(("medium_test", &source));
    // Passthrough modules require the native target's representation, not a
    // backend restriction on the numerical kernel being tested.
    let metal = r.adapter_info.backend == wgpu::Backend::Metal;
    let (_, target, flags) = if metal {crate::forward_variant::METAL_TARGET} else {crate::forward_variant::SPIRV_TARGET};
    let mut args = vec!["-target", target];
    args.extend_from_slice(crate::forward_variant::SLANG_FLAGS);
    args.extend_from_slice(flags);
    let defines=native.defines();
    args.extend(defines.iter().map(String::as_str));
    args.extend_from_slice(extra);
    // Numerical transport oracles exercise the certified path arithmetic contract.
    args.extend(vork_slang::MEDIUM_FP_FLAGS);
    let include: Vec<_> = native.includes.iter().map(|p| p.display().to_string()).collect();
    for path in &include {args.extend(["-I",path]);}
    let code = Compiler::discover().compile_modules(&modules, "medium_test", &[("numerical_test","compute")], &args)?;
    let msl = metal.then(|| std::str::from_utf8(&code)).transpose()?;
    let shader = crate::native::module(&r.device, "native medium numerical test", (!metal).then_some(code.as_slice()), msl, threads);
    let bind = r.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {label: None, entries});
    let layout = r.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {label: None, bind_group_layouts: &[Some(&bind)], immediate_size: 0});
    let pipeline = r.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {label: None, layout: Some(&layout), module: &shader, entry_point: Some("numerical_test"), compilation_options: Default::default(), cache: None});
    r.errors.check()?;
    Ok(pipeline)
}

impl Installed {
    pub(crate) fn bind_inputs(&self, plan: &mut animation::Plan, words: &mut [u32], base: usize) -> Result<()> {
        for (name, offset, layout, time_offset) in &self.inputs {
            programs::bind_inputs(plan, name, layout, &mut words[*offset..*offset + layout.words], base + offset, *time_offset)?;
        }
        Ok(())
    }
}

// Receiver support does not link illumination, point programs or their exported
// transport adapters. Keep this reduction small on both native backends.
