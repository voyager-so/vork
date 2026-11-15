//! The Slang shader backend for built-in and programmed scenes (docs/design/shaders.md).
//! Depth roles use the precompiled depth module; the control roles (opaque and
//! transparent controls) use the precompiled all-capabilities generic
//! (the superset fallback) or a variant compiled at runtime by slangc, whose native
//! code is kept in the disk cache. Media link the same transport as resolve.
use crate::{
    disk_cache,
    forward_variant::{FeatureMask, ForwardKey, METAL_TARGET, NATIVE_OPTIMIZATION, SLANG_FLAGS, SPIRV_TARGET, Stage},
    native, programs,
    pipelines::{GeometryKey, Role, VariantShader, try_init},
};
use anyhow::Context;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
};

/// What a scene adds to the Slang modules: its material programs and its named
/// geometry attributes (VORK_ATTRIBUTES; then the stages carrying full varyings
/// compile at install too), and the shared mesh-medium transport.
#[derive(Clone, Default)]
pub(crate) struct SlangScene {
    pub(crate) program_trace: bool,
    pub(crate) propagation: Option<crate::propagation::Native>,
    pub(crate) camera: Option<Arc<crate::camera_program::Native>>,
    pub(crate) environment: Option<crate::environment_program::Primary>,
    pub(crate) programs: Option<Arc<programs::Native>>,
    pub(crate) attributes: u32,
    pub(crate) textures: u32,
    pub(crate) samplers: u32,
    pub(crate) aov_count: usize,
    pub(crate) ray_refraction: bool,
    pub(crate) connection_transmission: bool,
    /// Installed coverage proof, shared by table, imported and program materials.
    pub(crate) surface_coverage: bool,
    pub(crate) attribute_coverage: bool,
    pub(crate) shadow_capacity: u32,
    pub(crate) no_shadow_containment: bool,
    /// Installed geometry contains no point domains; omit compact hit decoding.
    pub(crate) no_points: bool,
    pub(crate) gaussians: bool,
    pub(crate) no_boundary_transactions: bool,
    /// All installed closures have specular_weight == 1, including program outputs.
    pub(crate) unit_ior: bool,
    /// All possible transmitting closures have unit weight; opaque weights do not occupy media.
    pub(crate) unit_boundary_ior: bool,
    /// No installed record uses glTF boundary tint semantics.
    pub(crate) no_gltf_tint: bool,
    /// No installed material can produce film, including secondary ray hits.
    pub(crate) no_film: bool,
    /// The irradiance volume (probes.slang) replaces world diffuse light.
    pub(crate) probes: bool,
    /// Raster physical-fibre resources participate in pipeline layout identity.
    pub(crate) fibre: bool,
    pub(crate) denoise: bool,
    pub(crate) field_controls: bool,
    /// Software raster interpolation must use the retained surface point before depth tests.
    pub(crate) software_raster: bool,
    pub(crate) bounded_secondary_compilation: bool,
    pub(crate) approximate_field_boundaries: bool,
    pub(crate) medium: Option<Arc<crate::medium::Native>>,
}

impl SlangScene {
    /// Straight camera escapes keep their original direction and observer through
    /// null coverage. Evaluate that radiance separately from reusable transport.
    /// Authored propagation can bend the escape and evaluates it in transport.
    pub(crate) fn path_environment_prepass(&self)->bool {
        self.environment.as_ref().is_some_and(|e|e.constant_camera.is_none()) && self.propagation.is_none()
    }

    /// Kernels that see the environment program along camera rays: raster
    /// composition, the straight-ray prepass, and propagated path escapes. Material
    /// evaluation keeps its shipped executables; its world-unlinked glass
    /// estimate reads the retained camera layer (environment.md).
    fn camera_environment(&self,kernel:Option<crate::pipelines::ProgramKernel>)->Option<&crate::environment_program::Primary> {
        use crate::pipelines::ProgramKernel;
        self.environment.as_ref().filter(|_|match kernel {
            Some(ProgramKernel::Path {..})=>self.propagation.is_some(),
            Some(ProgramKernel::PathEnvironment {..})=>true,
            Some(ProgramKernel::Decoupled {stage,..})=>stage.composes(),
            _=>false,
        })
    }

    /// Owning-cell admission only changes kernels that can encounter implicit fields.
    pub(crate) fn approximate_field_admission(&self) -> bool {
        self.field_controls && self.approximate_field_boundaries
    }

    #[cfg(test)]
    pub(crate) fn test_entry_inputs(&self,main:&str,entry:(&str,&str),kernel:Option<crate::pipelines::ProgramKernel>)->(Vec<String>,blake3::Hash) {
        self.with_entry_inputs(false,false,false,main,None,entry,kernel,None,|modules,args,_,_,transforms| {
            let mut hash=transforms.prefix((0,0,0));
            // This test fingerprint observes the actual assembled request. Runtime
            // keys use Slang's validated import closure rather than the full module list.
            for part in modules.iter().flat_map(|(name,source)|[*name,*source]).chain(args.iter().copied()) {
                hash.update(&(part.len() as u64).to_le_bytes());hash.update(part.as_bytes());
            }
            (args.iter().map(|arg|(*arg).to_owned()).collect(),hash.finalize())
        })
    }

    /// Assemble the exact request consumed by native compilation and cache lookup.
    /// Keep policy decisions here; pipeline identities must not reinterpret scene flags.
    #[allow(clippy::too_many_arguments)]
    fn with_entry_inputs<T>(
        &self, metal:bool, ray_shadows:bool, msaa:bool,
        main:&str, text:Option<&str>, entry:(&str,&str),
        kernel:Option<crate::pipelines::ProgramKernel>, program:Option<u32>,
        consume:impl FnOnce(&[(&str,&str)],&[&str],&std::collections::BTreeMap<String,Vec<u8>>,&[vork_slang::InputIdentity],EntryTransforms)->T,
    )->T {
        use crate::pipelines::{ProgramKernel, ShadeStage};
        let (_, target, flags) = if metal {
            METAL_TARGET
        } else {
            SPIRV_TARGET
        };
        let all_programs = self.table_programs(kernel,ray_shadows);
        let program=program.or_else(||if let Some(ProgramKernel::Decoupled {stage:ShadeStage::EvaluateProgram(id),..})=kernel {Some(id)} else {None});
        let primary=program.and_then(|id|all_programs.map(|p|&p.camera[id as usize-1]));
        let programs=primary.or(all_programs);
        let path_medium=self.medium.as_ref().filter(|_|matches!(kernel,Some(ProgramKernel::Path {media:Some(_),..})));
        let mut kernel_defines=vec![format!("-DVORK_AOV_COUNT={}",self.aov_count),format!("-DVORK_SHADOW_CAPACITY={}",self.shadow_capacity)];
        if self.approximate_field_admission() && !matches!(kernel,Some(ProgramKernel::Path {..})) {
            kernel_defines.push("-DVORK_APPROXIMATE_FIELD_ADMISSION".into());
        }
        if self.software_raster {kernel_defines.push("-DVORK_SOFTWARE_RASTER".into());}
        if self.program_trace {kernel_defines.push("-DVORK_PROGRAM_TRACE".into());}
        if self.program_trace && ((main=="program_depth" && entry.0=="masked_fs") || (self.software_raster && (matches!(main,"depth"|"decoupled_visibility"|"program_depth") || matches!(entry.0,"controls_fs"|"transparent_controls_fs")))) {
            kernel_defines.push("-DVORK_RAY_BINDING_GROUP=4".into());
            if metal {
                // Global/material buffers, fragment storage, and four fragment
                // geometry buffers precede visibility's retained shadow group.
                kernel_defines.extend(["-DVORK_RAY_SEQUENCE_REGISTER=t16","-DVORK_RAY_SCENE_REGISTER=t17","-DVORK_RAY_ALPHA_REGISTER=t18","-DVORK_RAY_HITS_REGISTER=b19"].map(str::to_owned));
            }
        }
        if !self.field_controls && !matches!(kernel,Some(ProgramKernel::Path {..})) {kernel_defines.push("-DVORK_NO_IMPLICIT".into());}
        // A scene's own path kernel also states two absences its installed key
        // proves (forward_variant: the union of everything the scene installs)
        // as source, where the key's constants reach too late: without them
        // Slang instantiates and inlines the implicit-field traversal and the
        // thin-film spectra into the transport loop before anything folds, and
        // SPIRV-Tools promotes that state through the loop only to discard it.
        // The kernel's identity already holds the key, so these defines add no
        // kernel: equal keys compile equal sources.
        if let Some(ProgramKernel::Path {features,..})=kernel {
            use crate::forward_variant::{FEATURE_FILM,FEATURE_IMPLICIT_GEOMETRY,FEATURE_INSTALLED};
            if features&FEATURE_INSTALLED!=0 {
                if features&FEATURE_IMPLICIT_GEOMETRY==0 {kernel_defines.push("-DVORK_NO_IMPLICIT".into());}
                if features&FEATURE_FILM==0 {kernel_defines.push("-DVORK_NO_FILM".into());}
            }
        }
        if self.unit_ior {kernel_defines.push("-DVORK_UNIT_IOR".into());}
        if self.unit_boundary_ior {kernel_defines.push("-DVORK_UNIT_BOUNDARY_IOR".into());}
        if self.no_gltf_tint {kernel_defines.push("-DVORK_NO_GLTF_TINT".into());}
        if self.no_points {kernel_defines.push("-DVORK_NO_POINTS".into());}
        if !self.gaussians {kernel_defines.push("-DVORK_NO_GAUSSIANS".into());}
        if self.no_shadow_containment {kernel_defines.push("-DVORK_NO_SHADOW_CONTAINMENT".into());}
        if self.no_boundary_transactions && matches!(kernel,Some(ProgramKernel::Path {..})) {kernel_defines.push("-DVORK_NO_BOUNDARY_TRANSACTIONS".into());}
        if !self.connection_transmission {kernel_defines.push("-DVORK_NO_CONNECTION_TRANSMISSION".into());}
        // Medium kernels already have their own outlining policy. Combining it
        // with field outlining makes the NVIDIA driver expand nested closures.
        if self.field_controls && self.medium.is_none() {kernel_defines.push("-DVORK_FIELD_OUTLINE".into());}
        if self.field_controls && matches!(entry.0,"controls_fs"|"transparent_controls_fs") {kernel_defines.push("-DVORK_IMPLICIT_CONTROLS".into());}
        #[cfg(any(test, feature="test-support"))]
        if matches!(kernel,Some(ProgramKernel::Path {..})) {
            if let Ok(mode)=std::env::var("VORK_TEST_POISON_PATH_SAMPLE") {
                kernel_defines.push(format!("-DVORK_TEST_POISON_PATH_SAMPLE={}",if mode=="all" {2}else{1}));
            }
        }
        if let Some(ProgramKernel::Path {media:Some(count),..})=kernel {
            let native=path_medium.expect("installed path medium");
            if native.convex {
                kernel_defines.push("-DVORK_MEDIUM_CONVEX".to_owned());
            }
            if let Some(limit)=native.work_limit {kernel_defines.push(format!("-DVORK_MEDIUM_WORK_LIMIT={limit}"));}
            if !native.homogeneous && native.scattering_free && native.emission_free {kernel_defines.push("-DVORK_MEDIUM_ABSORPTION_ONLY".to_owned());}
            kernel_defines.extend(["-DVORK_PATH_MEDIA".to_owned(),format!("-DVORK_PATH_MEDIUM_BOUNDS={count}"),format!("-DVORK_PATH_HETEROGENEOUS={}",u8::from(!native.homogeneous))]);
        }
        // Secondary hits can reach every installed material. The same union
        // used by camera evaluators proves the shared film response absent.
        if self.no_film && matches!(kernel,Some(ProgramKernel::Gi {..})) {
            kernel_defines.push("-DVORK_NO_FILM".into());
        }
        if let Some(ProgramKernel::Decoupled {stage,features,boundaries,..})=kernel {
            if self.no_film && (stage.table() || matches!(stage,ShadeStage::EvaluateProgram(_))) {
                kernel_defines.push("-DVORK_NO_FILM".into());
            }
            kernel_defines.extend(crate::camera_native::COMPILER_FLAGS.iter().map(|flag|(*flag).to_owned()));
            kernel_defines.push(format!("-DVORK_REFRACTION_BOUNDARIES={boundaries}"));
            kernel_defines.push("-DVORK_CAMERA_SPLIT".into());
            if matches!(stage,ShadeStage::Fused|ShadeStage::Compose) {kernel_defines.push("-DVORK_COMPOSE_STORED".into());}
            if features & crate::forward_variant::FEATURE_RECEIVER_DIFFUSION!=0 {kernel_defines.push("-DVORK_RECEIVER_DIFFUSION".into());}
            if stage==ShadeStage::Fused {kernel_defines.push("-DVORK_CAMERA_FUSED".into());}
            if matches!(stage,ShadeStage::Index|ShadeStage::Classify|ShadeStage::Prefix|ShadeStage::Scatter) {kernel_defines.push("-DVORK_CAMERA_CLASSIFY".into());}
            if stage==ShadeStage::Compose {kernel_defines.push(NATIVE_OPTIMIZATION.to_owned());}
            if stage.composes() && msaa {kernel_defines.push("-DVORK_RASTER_MSAA".into());}
        }

        // Bound secondary material graphs at every quality. Keep O2 draft-only:
        // changing optimization changes final secondary sample results.
        let surface_outlining = !metal && !self.field_controls && self.medium.is_none()
            && matches!(kernel, Some(ProgramKernel::Gi {..} | ProgramKernel::Reflections {..}));
        // Retain one surface-lighting body through Slang for camera receivers
        // too. Restore driver inlining below, without enabling path O2 or
        // changing the camera's material/Blend boundaries.
        let camera_surface_outlining = !metal && !self.field_controls && self.medium.is_none()
            && matches!(kernel, Some(ProgramKernel::Decoupled {stage,..}) if stage.table() || matches!(stage,ShadeStage::EvaluateProgram(_)));
        if surface_outlining || camera_surface_outlining {kernel_defines.push("-DVORK_SURFACE_SLANG_OUTLINE".into());}
        let transport_outlining = matches!(kernel, Some(ProgramKernel::Path {..}|ProgramKernel::PathEnvironment {..}))
            || surface_outlining;
        // Camera kernels otherwise duplicate the implicit grid walk at every
        // classify/evaluate call site. Keep their existing Slang boundaries.
        let camera_outlining = !metal && matches!(kernel, Some(ProgramKernel::Decoupled {..}));
        if transport_outlining {
            if matches!(kernel, Some(ProgramKernel::Path {..}|ProgramKernel::PathEnvironment {..})) || self.bounded_secondary_compilation {
                kernel_defines.push(NATIVE_OPTIMIZATION.to_owned());
            }
            kernel_defines.push("-DVORK_PATH_SLANG_OUTLINE".into());
        }
        if path_medium.is_some() {
            kernel_defines.extend(vork_slang::MEDIUM_FP_FLAGS.map(str::to_owned));
        }
        let adaptive=matches!(kernel,Some(ProgramKernel::Path {adaptive:true,..}|ProgramKernel::PathEnvironment {adaptive:true}));
        let mut program_modules = programs.map_or_else(Vec::new, |p| p.modules.clone());
        if matches!(kernel,Some(ProgramKernel::Path {..})) {
            if let Some(p)=&self.propagation {
                kernel_defines.push("-DVORK_PROPAGATION".into());
                for module in &p.modules {
                    if !program_modules.iter().any(|(name,_)|name==&module.0) {program_modules.push(module.clone());}
                }
            }
        }
        if matches!(kernel,Some(ProgramKernel::Path {..}|ProgramKernel::PathEnvironment {..})) {
            if let Some(camera)=&self.camera {
                kernel_defines.push("-DVORK_LIGHT_PATHS".into());
                for module in &camera.modules {
                    if !program_modules.iter().any(|(name,_)|name==&module.0) {program_modules.push(module.clone());}
                }
            }
        }

        if let Some(environment)=self.camera_environment(kernel) {
            kernel_defines.push("-DVORK_CAMERA_ENVIRONMENT".into());
            for module in &environment.modules {
                if !program_modules.iter().any(|(name,_)|name==&module.0) {program_modules.push(module.clone());}
            }
        }
        let secondary=entry.1=="compute" && (self.ray_refraction || ray_shadows && (self.connection_transmission || self.attribute_coverage || all_programs.is_some_and(|p|p.coverage || self.fibre)));
        if primary.is_some() && secondary {
            for module in &all_programs.unwrap().modules {
                if let Some(existing)=program_modules.iter_mut().find(|(name,_)|name==&module.0) {*existing=module.clone();} else {program_modules.push(module.clone());}
            }
        }
        // Secondary camera transport already receives CameraPrograms as P.
        // Shadow/probe candidate callbacks still import their shared dispatcher.
        let medium_sources=self.medium.as_ref().is_some_and(|m|m.mesh_sources);
        let program_coverage=secondary && ray_shadows && (medium_sources || self.connection_transmission || self.attribute_coverage || all_programs.is_some_and(|p|p.coverage || self.fibre));
        if let Some(p)=all_programs.filter(|_|program_coverage) {
            let p=if !self.connection_transmission && !medium_sources {p.for_coverage()} else {p};
            program_modules.push(("scene_programs".into(),p.code.replace("struct Programs :","struct ScenePrograms :")));
        }
        if program_coverage && all_programs.is_none() {
            program_modules.push(("scene_programs".into(), "import surface; typedef NoPrograms ScenePrograms;".into()));
        }
        if let Some(native)=path_medium {
            if programs.is_none() {program_modules.extend(programs::abi().native_modules());}
            program_modules.extend(native.modules.clone());
            program_modules.push(("path_medium_scene".into(),format!("import frame;\n{}",native.point)));
        }
        let attributes=if main=="decoupled_visibility" && matches!(entry.0,"fs"|"single_fs") {0} else {self.attributes};
        let define = format!("-DVORK_ATTRIBUTES={attributes}");
        // Controls do not produce lighting; only compute consumers read probes.
        let probes = self.probes && entry.1 == "compute";
        // Secondary compute kernels own group 2; decoupled colour shares the
        // raster shadow group and reads the volume after its shadow entries.
        let fibre = self.fibre && main == "decoupled_programs";
        let fibre_binding="-DVORK_FIBRE_BINDING=4";
        let probe_binding = format!("-DVORK_PROBE_BINDING={}", if main == "decoupled_programs" { 9 } else { 0 });
        // Decoupled frame, sampling and record buffers precede probes.
        let probe_register: Option<u32> = if metal && probes {
            match main { "decoupled_programs" => Some(16), _ => None }
        } else {None};
        let denoise=self.denoise && matches!(entry.0,"controls_fs"|"transparent_controls_fs");
        let mut modules = native::SOURCES.to_vec();
        for (name, source) in &program_modules {
            if let Some(module)=modules.iter_mut().find(|(n,_)|*n==name) {*module=(name,source);} else {modules.push((name,source));}
        }
        if let Some(text) = text {
            modules.push((main, text));
        }
        let mut include: Vec<String> = if secondary {all_programs} else {programs}
            .map_or(&[][..], |p| &p.include[..])
            .iter()
            .map(|d| d.display().to_string())
            .collect();
        if let Some(environment)=self.camera_environment(kernel) {include.extend(environment.include.iter().map(|p|p.display().to_string()));}
        if let Some(p)=&self.propagation {include.extend(p.include.iter().map(|p|p.display().to_string()));}
        if let Some(camera)=&self.camera {include.extend(camera.include.iter().map(|p|p.display().to_string()));}
        if let Some(native)=path_medium {include.extend(native.includes.iter().map(|p|p.display().to_string()));}
        let query_defines:Vec<_> = if metal && entry.1=="vertex" {
            vec!["-DVORK_MATERIAL_WORDS_REGISTER=t3","-DVORK_FRAME_LIGHTS_REGISTER=t30","-DVORK_FRAME_CLUSTERS_REGISTER=b29","-DVORK_POINT_WORDS_REGISTER=t2"]
        }else{Vec::new()};
        let kernel_module = match main {
            "gi_programs" => "gi_ray",
            "reflections_programs" => "reflections_hits", "probe_programs" => "probe_update",
            "path_programs_main" | "path_programs_fixed" | "path_environment_main" => "path", _ => "",
        };
        let mut native_registers = if metal {
            crate::forward_variant::hit_kernel_registers(kernel_module,
                adaptive,
                main == "probe_programs" || (self.probes && matches!(main,"gi_programs"|"reflections_programs")))
        } else {Vec::new()};
        if let Some(first) = probe_register {
            native_registers.extend(crate::forward_variant::probe_registers(first));
        }
        let args: Vec<&str> = ["-target", target]
            .into_iter()
            .chain(SLANG_FLAGS.iter().copied())
            .chain(flags.iter().copied())
            .chain(native_registers.iter().map(String::as_str))
            .chain(program_coverage.then_some("-DVORK_PROGRAM_COVERAGE"))
            .chain(self.attribute_coverage.then_some("-DVORK_ATTRIBUTE_COVERAGE"))
            .chain(kernel_defines.iter().map(String::as_str))
            .chain(query_defines.iter().copied())

            .chain((ray_shadows || (main=="decoupled_programs" && self.ray_refraction)).then_some("-DVORK_RAY_BINDINGS"))
            .chain((main=="decoupled_programs" && self.ray_refraction).then_some("-DVORK_RAY_REFRACTION"))
            .chain(denoise.then_some("-DVORK_DENOISE"))
            .chain(probes.then_some("-DVORK_PROBES"))
            .chain(fibre.then_some("-DVORK_FIBRE"))
            .chain(fibre.then_some(fibre_binding))
            .chain(probes.then_some(probe_binding.as_str()))
            .chain((self.attributes > 0).then_some(define.as_str()))
            // Metal takes no array as a stage input: stream.slang then declares
            // the attributes after the eighth as separate members.
            .chain((metal && attributes > 8).then_some("-DVORK_STAGE_SCALARS"))
            .chain(include.iter().flat_map(|d| ["-I", d.as_str()]))
            .collect();
        let identities=(if secondary {all_programs} else {programs}).map(|p|p.identities.clone()).unwrap_or_default();
        let inputs=(if secondary {all_programs} else {programs}).map(|p|p.inputs.as_slice()).unwrap_or(&[]);
        consume(&modules,&args,&identities,inputs,EntryTransforms {
            localize_blends:transport_outlining && !metal,
            inline_metal:metal && matches!(kernel,Some(ProgramKernel::Decoupled {stage:ShadeStage::Evaluate,..})),
            driver_inline:(self.field_controls || surface_outlining || transport_outlining || camera_surface_outlining) && !metal && (!camera_outlining || camera_surface_outlining),
        })
    }

    fn table_programs(&self,kernel:Option<crate::pipelines::ProgramKernel>,ray_shadows:bool)->Option<&programs::Native> {
        // Composition consumes stored closures and explicitly calls NoPrograms.
        // Do not import the scene dispatcher merely to define an unused type.
        if matches!(kernel,Some(crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::Compose,..}|crate::pipelines::ProgramKernel::PathEnvironment {..})) {return None;}
        let table=matches!(kernel,Some(crate::pipelines::ProgramKernel::Decoupled {stage,..}) if stage.table());
        self.programs.as_deref().filter(|_|!table || self.transport_programs(ray_shadows))
    }
    fn transport_programs(&self,ray_shadows:bool)->bool {
        self.programs.as_ref().is_some_and(|p|self.ray_refraction || ray_shadows && (p.coverage || self.connection_transmission) || self.fibre)
    }


    #[allow(clippy::too_many_arguments)]
    fn entry_key(&self,metal:bool,ray_shadows:bool,msaa:bool,request:&Entry,lookup_only:bool,cache:&dyn Fn(blake3::Hash)->PathBuf,prepared:&PreparedEntries)->anyhow::Result<blake3::Hash> {
        self.with_entry_inputs(metal,ray_shadows,msaa,&request.main,request.text.as_deref(),request.entry,request.kernel,request.program,|modules,args,identities,inputs,transforms| {
            let compiler=vork_slang::Compiler::discover();
            let identity=if lookup_only {
                // A scheduling lookup never compiles or prepares inputs; it admits the compiler,
                // which runs it only when the store has not seen this toolchain pass.
                // Unknown receipts are cold.
                programs::cached_native_input_identity(cache,&compiler,modules,&request.main,&[request.entry],args,identities,inputs)?
                    .context("native entry inputs have not been captured")?
            } else {
                // Registry lookup must validate external imports, too. Reuse the
                // existing resolver receipt; on a cold request capture inputs
                // without generating code or publishing a remembered pipeline.
                let (identity, snapshot)=programs::native_input_identity(cache,&compiler,modules,&request.main,&[request.entry],args,identities,inputs)?;
                let mut artifact=transforms.prefix(if metal {request.threads} else {(0,0,0)});
                artifact.update(&identity.identity);
                if let Some(snapshot)=snapshot.filter(|_|disk_cache::read(&cache(artifact.finalize())).is_none()) {
                    let path=programs::native_input_receipt_path(cache,&compiler,modules,&request.main,&[request.entry],args,identities,inputs)?;
                    prepared.lock().unwrap().insert(path,snapshot);
                }
                identity
            };
            let mut hash=transforms.prefix(if metal {request.threads} else {(0,0,0)});
            hash.update(&identity.identity);
            Ok(hash.finalize())
        })
    }


}

/// A generated entry is shared by code generation and remembered-key lookup.
struct Entry {
    main:String,
    text:Option<String>,
    entry:(&'static str,&'static str),
    threads:(u32,u32,u32),
    kernel:Option<crate::pipelines::ProgramKernel>,
    program:Option<u32>,
}

#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub(crate) struct EntryTransforms {
    pub(crate) localize_blends:bool,
    pub(crate) inline_metal:bool,
    pub(crate) driver_inline:bool,
}
impl EntryTransforms {
    pub(crate) fn prefix(self,threads:(u32,u32,u32))->blake3::Hasher {
        let mut hash=blake3::Hasher::new();
        hash.update(b"native-entry-transforms-v1");
        hash.update(&[u8::from(self.localize_blends),u8::from(self.inline_metal),u8::from(self.driver_inline)]);
        for size in [threads.0,threads.1,threads.2] {hash.update(&size.to_le_bytes());}
        hash
    }
}

type PreparedEntries = Mutex<std::collections::HashMap<PathBuf,vork_slang::PreparedModules>>;

/// A retained capture has the same resolver validation as a disk receipt. It
/// remains usable when persistence is disabled or its receipt was evicted.
fn take_prepared(
    prepared:&PreparedEntries,
    path:&std::path::Path,inputs:&[vork_slang::InputIdentity],
)->Option<vork_slang::PreparedModules> {
    let snapshot=prepared.lock().unwrap().remove(path);
    snapshot.filter(|snapshot| {
        let identity=snapshot.receipt();
        identity.valid() && inputs.iter().all(|input|identity.verify(input).is_ok())
    })
}

pub(crate) struct SlangForward {
    // Transfer cold registry captures to code generation; dropping a candidate
    // also drops snapshots for entries that will never be built.
    prepared: PreparedEntries,
    device: wgpu::Device,
    metal: bool,
    check_only: bool,
    ray_shadows: bool,
    msaa: bool,
    /// The camera vertex stage, shared by depth and forward (bit-identical depth).
    camera: wgpu::ShaderModule,
    opaque_depth: wgpu::ShaderModule,
    single_depth: [wgpu::ShaderModule; 2],
    /// Alpha-tested depth: cutout, then hashed (dithered) coverage.
    masked_depth: [wgpu::ShaderModule; 2],
    /// Decoupled visibility keys (decoupled_visibility.slang): vertex, double- and
    /// single-sided fragment, the alpha-tested vertex and fragment stages, then
    /// transparent draws' fragment lists.
    visibility: [wgpu::ShaderModule; 6],
    /// Shared geometry vertex stage for fibre captures.
    fibre_vertex: wgpu::ShaderModule,

    authored_coverage: Mutex<std::collections::HashMap<(u32,crate::pipelines::CoverageStage),Arc<OnceLock<wgpu::ShaderModule>>>>,
    fibre: [OnceLock<wgpu::ShaderModule>;crate::fibre_transport::Stage::ALL.len()],
    scene: SlangScene,
    authored_controls: Mutex<std::collections::HashMap<(ForwardKey,u32),Arc<OnceLock<wgpu::ShaderModule>>>>,
    /// The precompiled generic per fragment Stage (Stage::ALL), without and
    /// with hashed coverage (ForwardKey::generic).
    generic: [[OnceLock<wgpu::ShaderModule>; 2]; Stage::ALL.len()],
    /// Disk-cache path for a variant's native code, by its module text.
    cache: Box<dyn Fn(blake3::Hash) -> PathBuf + Send + Sync>,
}

impl SlangForward {
    pub(crate) fn new(
        r: &crate::Renderer,
        ray_shadows: bool,
        msaa: bool,
        scene: SlangScene,
    ) -> Self {
        let store = r.disk_cache.clone();
        let mut this = Self {
            prepared: Default::default(),
            device: r.device.clone(),
            check_only: r.check_only,
            metal: r.adapter_info.backend == wgpu::Backend::Metal,
            ray_shadows,
            msaa,
            camera: native::shader_module(&r.device, &native::DEPTH_MAIN_VS),
            opaque_depth: native::shader_module(&r.device, &native::DEPTH_OPAQUE_DEPTH_VS),
            single_depth: [
                native::shader_module(&r.device, &native::DEPTH_OPAQUE_SINGLE_DEPTH_VS),
                native::shader_module(&r.device, &native::DEPTH_OPAQUE_SINGLE_DEPTH_FS),
            ],
            masked_depth: [
                native::shader_module(&r.device, &native::DEPTH_DEPTH_FS),
                native::shader_module(&r.device, &native::DEPTH_HASHED_DEPTH_FS),
            ],
            visibility: [
                native::shader_module(&r.device, &native::DECOUPLED_VISIBILITY_VS),
                native::shader_module(&r.device, &native::DECOUPLED_VISIBILITY_FS),
                native::shader_module(&r.device, &native::DECOUPLED_VISIBILITY_SINGLE_FS),
                native::shader_module(&r.device, &native::DECOUPLED_VISIBILITY_MASKED_VS),
                native::shader_module(&r.device, &native::DECOUPLED_VISIBILITY_MASKED_FS),
                native::shader_module(&r.device, if scene.field_controls {&native::DECOUPLED_VISIBILITY_TRANSPARENT_FRAGMENTS_FS} else {&native::DECOUPLED_VISIBILITY_TRANSPARENT_TRIANGLES_FS}),
            ],
            fibre_vertex: native::shader_module(&r.device,&native::DEPTH_SHADOW_VS),
            authored_coverage:Default::default(),
            fibre:Default::default(),
            scene,
            authored_controls:Default::default(),
            generic: Default::default(),
            cache: Box::new(move |hash| store.path("slang-variant", 1, hash)),
        };
        let attributes=this.scene.attributes;
        if this.scene.software_raster {
            let compile=|module,entry| this.build_entry(&this.stream_entry(module,entry),false).expect("compile software raster surface").built();
            let camera=compile("depth","main_vs");
            let opaque=compile("depth","software_depth_vs");
            let single=[opaque.clone(),compile("depth","software_single_depth_fs")];
            let masked=[compile("depth","depth_fs"),compile("depth","hashed_depth_fs")];
            let visibility=[compile("decoupled_visibility","vs"),compile("decoupled_visibility","fs"),compile("decoupled_visibility","single_fs"),compile("decoupled_visibility","masked_vs"),compile("decoupled_visibility","masked_fs"),compile("decoupled_visibility","transparent_fragments_fs")];
            this.camera=camera;this.opaque_depth=opaque;this.single_depth=single;this.masked_depth=masked;this.visibility=visibility;
        }
        if attributes > 0 && !this.scene.software_raster {
            let stream = |module, entry| {
                if let Some(shader)=this.stream_shader(module,entry) {
                    native::shader_module(&this.device,shader)
                } else {
                    this.build_entry(&this.stream_entry(module,entry),false).expect("compile scene attribute stream").built()
                }
            };
            let camera=stream("depth","main_vs");
            let fibre_vertex=stream("depth","shadow_vs");
            let masked_depth=[stream("depth","depth_fs"),stream("depth","hashed_depth_fs")];
            let masked_vs=stream("decoupled_visibility","masked_vs");
            let transparent_fs=stream("decoupled_visibility",if this.scene.field_controls {"transparent_fragments_fs"} else {"transparent_triangles_fs"});
            let masked_fs=stream("decoupled_visibility","masked_fs");
            this.camera=camera;this.fibre_vertex=fibre_vertex;this.masked_depth=masked_depth;
            this.visibility[3]=masked_vs;this.visibility[5]=transparent_fs;
            this.visibility[4]=masked_fs;

        }
        this
    }
    fn generic(&self,stage:Stage,hashed:bool)->wgpu::ShaderModule {
        self.generic[stage as usize][usize::from(hashed)].get_or_init(|| {
            if let Some(shader)=self.generic_shader(stage,hashed) {
                native::shader_module(&self.device,shader)
            } else {
                let request=self.control_entry(ForwardKey::generic(stage,hashed),None);
                self.build_entry(&request,false).expect("compile scene attribute controls").built()
            }
        }).clone()
    }

    /// A coverage stage with the scene's programs: masked depth, masked shadow, or
    /// alpha-tested visibility keys (surface.slang program_covers).
    fn coverage_entry(&self, entry: &'static str, program:u32) -> Entry {
        let programs=&self.scene.programs.as_ref().unwrap().camera[program as usize-1];
        let (visibility,ret,body)=match entry {
            "depth_fs" => ("","void","v.primitive=primitive;let s=surface<Unspecialized,Programs>(v,front);if(s.discarded){discard;}"),
            "masked_fs" => ("import decoupled_fragments;","VisibilityOutput","return masked_visibility<Programs>(v,front,primitive);"),
            _ => unreachable!("unsupported coverage entry"),
        };
        let software_depth=self.scene.software_raster && entry=="depth_fs";
        let (visibility,ret,body)=if software_depth {("import software_surface;","float","v=software_surface(v.object,primitive,v.position,floor(v.position.xy)+sample_position,float2(0.));v.primitive=primitive;let s=surface<Unspecialized,Programs>(v,front);if(s.discarded){discard;}return v.position.z;")} else {(visibility,ret,body)};
        let sample=if software_depth {",float2 sample_position:SV_VulkanSamplePosition"} else {""};
        let target=if software_depth {" : SV_Depth"} else if matches!(ret,"void"|"VisibilityOutput") {""} else {" : SV_Target0"};
        let text=format!("import vork_constants;import frame;import stream;import material;import surface;import variant;import vork_shadow_coverage;{visibility}\n{}\n[shader(\"fragment\")] {ret} {entry}(Varying v,bool front:SV_IsFrontFace,uint primitive:SV_PrimitiveID{sample}){target} {{{body}}}",programs.code);
        let text=if self.scene.program_trace {format!("import vork_rays;\n{text}")} else {text};
        Entry {main:"program_depth".into(),text:Some(text),entry:(entry,"fragment"),threads:(0,0,0),kernel:None,program:Some(program)}
    }

    /// Build or probe the same request used by registry and remembered identities.
    fn build_entry(&self,request:&Entry,probe:bool)->anyhow::Result<Lookup<wgpu::ShaderModule>> {
        let (main,text,entry,threads,kernel,program)=(&*request.main,request.text.as_deref(),request.entry,request.threads,request.kernel,request.program);
        let label=entry.0;
        // Authored stages and their effect proofs were validated before this
        // backend link. The check-only renderer cannot execute this module and
        // never writes a native artifact or marks a cache entry as compiled.
        if self.check_only {
            return Ok(if probe { Lookup::Stored } else {
                Lookup::Built(native::module(&self.device, label, None, Some(""), threads))
            });
        }
        let diagnostic=kernel.and_then(crate::pipelines::ProgramKernel::camera_label);
        let label=diagnostic.as_deref().unwrap_or(label);
        self.scene.with_entry_inputs(self.metal,self.ray_shadows,self.msaa,main,text,entry,kernel,program,|modules,args,identities,inputs,transforms| {
        let compiler=vork_slang::Compiler::discover();
        // Embedded renderer modules and compiler inputs have a shader-only
        // content identity in the store, independent of host builds/devices.
        let prefix = transforms.prefix(if self.metal {threads} else {(0,0,0)});
        if probe {
            let Some(identity)=programs::cached_native_input_identity(self.cache.as_ref(),&compiler,&modules,main,&[entry],&args,&identities,inputs)? else {return Ok(Lookup::Absent);};
            let mut hash=prefix.clone();
            hash.update(&identity.identity);
            return Ok(if disk_cache::read(&(self.cache)(hash.finalize())).is_some() {Lookup::Stored} else {Lookup::Absent});
        }
        let receipt_path = programs::native_input_receipt_path(self.cache.as_ref(), &compiler, &modules, main, &[entry], &args, &identities, inputs)?;
        let (identity,prepared)=if let Some(snapshot)=take_prepared(&self.prepared,&receipt_path,inputs) {
            (snapshot.receipt(),Some(snapshot))
        } else {
            programs::native_input_identity(self.cache.as_ref(),&compiler,&modules,main,&[entry],&args,&identities,inputs)?
        };
        let code = programs::native_code(&identity, prepared, inputs, &receipt_path, |identity| {
            let mut hash = prefix.clone();
            hash.update(identity);
            (self.cache)(hash.finalize())
        }, || compiler.prepare_identified_modules(&modules, main, &[entry], &args, &identities), |prepared| {
            let start = std::time::Instant::now();
            let mut code = prepared.compile()?;
            crate::timing::creation(&self.device,label,"Slang",start.elapsed().as_secs_f64()*1000.,0.);
            if transforms.localize_blends {
                let start = std::time::Instant::now();
                let _event = vork_slang::progress::announce("SPIR-V optimization", label);
                code=compiler.localize_native_blends(&code)?;
                crate::timing::creation(&self.device,label,"SPIR-V optimization",start.elapsed().as_secs_f64()*1000.,0.);
            }
            if transforms.inline_metal {
                code=crate::camera_native::inline_metal_functions(&code);
            }
            if transforms.driver_inline { allow_driver_inlining(&mut code)?; }
            Ok(code)
        })?;
        Ok(Lookup::Built(if self.metal {
            let source=String::from_utf8(code)?;
            vork_slang::metal_source::validate(&source)?;
            native::module(&self.device, label, None, Some(&source), threads)
        } else {
            native::module(&self.device, label, Some(&code), None, (0, 0, 0))
        }))
        })
    }

    fn camera_shipped(&self,stage:crate::pipelines::ShadeStage,receiver_diffusion:bool)->Option<&'static native::Compiled> {
        // Shipped camera kernels specialize for triangle-only scenes.
        if self.scene.software_raster || self.scene.field_controls {return None;}
        use crate::pipelines::ShadeStage;
        if self.scene.attributes>crate::constants::PRECOMPILED_ATTRIBUTE_SLOTS as u32
            && !matches!(stage,ShadeStage::Index|ShadeStage::Classify|ShadeStage::Prefix|ShadeStage::Scatter) {return None;}
        // Composition evaluates an environment program along camera rays.
        if self.scene.environment.is_some() && stage.composes() {return None;}
        // Shipped evaluators read proven World profiles through the retained words.
        // Other domains and coefficient programs still link their transport.
        // Coverage needs the runtime camera payload contract for every producer
        // and consumer, including native specialized evaluators and composition.
        // Shipped camera records carry no AOVs. Classification has no surface
        // payload; every evaluator and consumer of that payload must agree on N.
        if (self.scene.aov_count!=0 || self.scene.surface_coverage) && matches!(stage,ShadeStage::EvaluateOpaque|ShadeStage::Evaluate|ShadeStage::Compose) {return None;}
        let name=match stage {
            ShadeStage::Index=>"index",ShadeStage::Classify=>"classify",ShadeStage::Prefix=>"prefix",ShadeStage::Scatter=>"scatter",
            ShadeStage::EvaluateOpaque=>"evaluate_opaque",ShadeStage::Evaluate=>"evaluate",ShadeStage::EvaluateProgram(_)|ShadeStage::Fused|ShadeStage::Occupancy|ShadeStage::Indirect=>return None,
            ShadeStage::Compose=>if receiver_diffusion {"combine_diffusion"} else {"combine"},
        };
        Some(native::CAMERAS.iter().find(|(m,r,n,_)|*m==u8::from(self.msaa && stage.composes()) && *r==u8::from(self.ray_shadows) && *n==name).unwrap().3)
    }
    /// Classification/composition are embedded; material evaluation specializes at install.
    fn camera_fixed(&self,kernel:crate::pipelines::ProgramKernel)->Option<&'static native::Compiled> {
        use crate::pipelines::{ProgramKernel,ShadeStage};
        let ProgramKernel::Decoupled {stage,features,..}=kernel else {return None;};
        if stage.table() || matches!(stage,ShadeStage::EvaluateProgram(_)|ShadeStage::Indirect)
            {return None;}
        self.camera_shipped(stage,features & crate::forward_variant::FEATURE_RECEIVER_DIFFUSION!=0)
    }

    fn surface_fixed(&self,kernel:crate::pipelines::ProgramKernel)->Option<(&'static native::Compiled,Vec<(&'static str,crate::forward_variant::Capability)>)> {
        use crate::pipelines::{ProgramKernel,ShadeStage};
        use crate::surface_native::SurfaceKernel;
        if self.scene.software_raster && matches!(kernel,ProgramKernel::Decoupled {..}) {return None;}
        if self.scene.surface_coverage && matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {..}) {return None;}
        if self.metal || self.scene.field_controls || self.scene.attributes!=0 || self.scene.aov_count!=0
            || self.scene.camera_environment(Some(kernel)).is_some() || self.scene.fibre
            || !self.scene.no_shadow_containment || self.scene.connection_transmission && (self.scene.programs.is_some() || !self.ray_shadows)
            || self.scene.table_programs(Some(kernel),self.ray_shadows).is_some() {return None;}
        let boundaries=match kernel {ProgramKernel::Decoupled {boundaries,..}=>boundaries,_=>0};
        // Shared transport has no authored medium continuation. Keep those links
        // on their existing runtime path even when the active set fits one slot.
        if boundaries>1 || self.scene.ray_refraction && self.scene.medium.is_some() {return None;}
        // Setup warms refraction with probes. Keep non-probe refraction on its
        // established runtime executable: late-specialized native arithmetic can
        // perturb narrow glass highlights before denoising in that configuration.
        if self.scene.ray_refraction && !self.scene.probes {return None;}
        // Full candidate evaluation reads hit records installed for transmission.
        // No containment alone does not provide those records: an opaque scene
        // can have only compact alpha entries, whose hit address is NO_ALPHA.
        // Caster links force even opaque geometry through the candidate reader.
        let coverage=self.scene.connection_transmission && self.scene.no_shadow_containment && self.ray_shadows;
        let (family,features,ray_shadows,subsurface,emitter_profiles)=match kernel {
            // The existing fixed probe entry owns the unspecialized zero key.
            ProgramKernel::Probes {features:0,..}=>return None,
            ProgramKernel::Gi {features} if self.scene.medium.is_none()=>(SurfaceKernel::Gi,features,false,false,false),
            ProgramKernel::Reflections {features} if self.scene.medium.is_none()=>(SurfaceKernel::Reflections,features,false,false,false),
            ProgramKernel::Probes {features,directional:true} if self.scene.medium.is_none()=>(SurfaceKernel::Directional,features,false,false,false),
            ProgramKernel::Probes {features,directional:false} if self.scene.medium.is_none()=>(SurfaceKernel::Probes,features,false,false,false),
            ProgramKernel::Decoupled {stage:ShadeStage::Evaluate|ShadeStage::EvaluateOpaque,features,subsurface,emitter_profiles,..}=>(SurfaceKernel::Evaluate,features,self.ray_shadows,subsurface,emitter_profiles),
            ProgramKernel::Decoupled {stage:ShadeStage::Indirect,features,subsurface,emitter_profiles,..}
                if self.scene.medium.is_none() && self.scene.probes && self.ray_shadows=>(SurfaceKernel::Indirect,features,self.ray_shadows,subsurface,emitter_profiles),
            ProgramKernel::Decoupled {stage:ShadeStage::Occupancy,features,subsurface,emitter_profiles,boundaries:1}
                if coverage=>(SurfaceKernel::Occupancy,features,self.ray_shadows,subsurface,emitter_profiles),
            _=>return None,
        };
        if subsurface && !self.ray_shadows {return None;}
        let shader=native::SURFACES.iter().find(|(f,r,p,c,b,_)|*f==family && *r==(family.camera() && self.ray_shadows) && *p==self.scene.probes && *c==coverage && *b==boundaries).expect("shipped surface resource interface").5;
        Some((shader,crate::forward_variant::variant_capabilities(features,ray_shadows,subsurface,emitter_profiles)))
    }

    /// The prebuilt path engine's key for `kernel`'s scene, or what the engine
    /// cannot draw. The engine is path transport over the material table with
    /// every capability and no scene code (surface_native::PathEngine), on every
    /// GPU. Each refusal names something that needs code, resources or stored
    /// state the engine does not have; none is a speed tier. A refused scene
    /// compiles its own kernel before its first frame and draws only that, so
    /// its picture never depends on which route was taken.
    fn path_engine(&self,kernel:crate::pipelines::ProgramKernel)->Result<crate::pipelines::ProgramKernel,&'static str> {
        let engine=kernel.path_engine()?;
        let scene=&self.scene;
        #[cfg(any(test,feature="test-support"))]
        if std::env::var_os("VORK_TEST_POISON_PATH_SAMPLE").is_some() {return Err("a test compiles its own path kernel");}
        // Authored code: the engine evaluates only the material table (NoPrograms).
        if scene.programs.is_some() {return Err("a material program (authored Slang) is installed");}
        if scene.medium.is_some() {return Err("a participating medium is installed");}
        if scene.propagation.is_some() {return Err("an authored ray propagation is installed");}
        if scene.program_trace {return Err("a program traces rays");}
        // The engine links the one standard camera at dispatch id 1.
        if !scene.camera.as_ref().is_some_and(|camera|camera.standard_camera) {return Err("the cameras are not exactly one std/camera");}
        // Resources and records the engine's modules do not declare.
        if scene.attributes!=0 {return Err("a material reads named vertex attributes");}
        if scene.aov_count!=0 {return Err("AOVs are rendered");}
        if scene.probes {return Err("an irradiance volume is installed");}
        if scene.fibre {return Err("hair fibres are captured");}
        if scene.gaussians {return Err("Gaussian splats are installed");}
        if !scene.no_points {return Err("point geometry is installed");}
        // The engine's shadow walk holds one absorbing solid around a connection.
        if scene.shadow_capacity>1 {return Err("absorbing solids overlap");}
        Ok(engine)
    }
    /// The prebuilt module when `kernel` is itself the engine key of this scene.
    fn path_engine_module(&self,kernel:crate::pipelines::ProgramKernel)->Option<&'static native::Compiled> {
        let crate::pipelines::ProgramKernel::Path {adaptive,boundaries,..}=kernel else {return None;};
        if self.path_engine(kernel).ok()!=Some(kernel) {return None;}
        native::PATH_ENGINES.iter().find(|(family,_)|family.adaptive==adaptive && family.boundaries==boundaries).map(|(_,shader)|*shader)
    }

    /// Each program owns its evaluation bin, so table slots hold only built-in
    /// materials: table camera stages take the scene's programs only when secondary
    /// transport (ray refraction, program shadow coverage, fibres) can reach one.
    /// Then no table kernel grows with the scene's program count.
    fn camera_entry(&self,kernel:crate::pipelines::ProgramKernel)->Entry {
        use crate::pipelines::{ProgramKernel,ShadeStage};
        if let ProgramKernel::Decoupled {stage,..}=kernel {
            let fixed=match stage {
                ShadeStage::Index=>Some(("index","camera_index")),
                ShadeStage::Classify=>Some(("classify","camera_classify")),
                ShadeStage::Prefix=>Some(("prefix","camera_prefix")),
                ShadeStage::Scatter=>Some(("scatter","camera_scatter")),
                _=>None,
            };
            if let Some((stage,entry))=fixed {
                let source=crate::camera_native::entry_source(stage);
                return Entry {main:"camera_field_classification".into(),text:Some(source),entry:(entry,"compute"),threads:(8,8,1),kernel:Some(kernel),program:None};
            }
        }
        let source=self.scene.table_programs(Some(kernel),self.ray_shadows);
        let primary=if let crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::EvaluateProgram(id),..}=kernel {source.map(|source|&source.camera[id as usize-1])} else {None};
        let code=primary.or(source).map_or_else(String::new,|source|source.code.replace("struct Programs :","struct CameraPrograms :"));
        let secondary=source.filter(|_|primary.is_some() && self.scene.ray_refraction).map_or_else(String::new,|source|source.code.clone());
        let program=if source.is_some() {"CameraPrograms"} else {"NoPrograms"};
        let transport=if secondary.is_empty() {program} else {"Programs"};
        let decode_table=!matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::EvaluateProgram(_),..});
        let fused=matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::Fused,..});
        let compose=matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::Compose,..});
        let occupancy=matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::Occupancy,..});
        let indirect=matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::Indirect,..});
        let (threads,call)=if occupancy {
            ((8,8,1),format!("prepare_refraction_occupancy<Key,{transport}>(gid);"))
        } else if indirect {
            ((vork_slang::registry::CAMERA_INDIRECT_GROUP,1,1),format!("indirect_material<Key,{program}>(gid);"))
        } else if compose {
            ((8,8,1),"if(camera_config.screen!=0u && camera_config.lit!=0u) {composite_pixel<Key,NoPrograms,CameraTransmittance,3>(gid);} else {shade_pixel<Key,NoPrograms,CameraTransmittance,CameraDiffusion,3>(gid);}".to_owned())
        } else if fused {
            ((8,8,1),"shade_pixel<Key,NoPrograms,CameraTransmittance,CameraDiffusion,3>(gid);".to_owned())
        } else {
            ((64,1,1),format!("evaluate_material<Key,{program},{decode_table},{transport}>(gid);"))
        };
        let text=format!("import camera_config;import decoupled;import surface;import frame;\n{secondary}\n{code}\n[shader(\"compute\")][numthreads({},{},{})] void decoupled_programs(uint3 gid:SV_DispatchThreadID) {{{call}}}",threads.0,threads.1,threads.2);
        let text={
            let crate::pipelines::ProgramKernel::Decoupled {features,subsurface,emitter_profiles,..}=kernel else {unreachable!("camera program key")};
            crate::forward_variant::variant_module(features,self.ray_shadows,subsurface,emitter_profiles,None,&text)
        };
        Entry {main:"decoupled_programs".into(),text:Some(text),entry:("decoupled_programs","compute"),threads,kernel:Some(kernel),program:None}
    }

    fn fibre_entry(&self,stage:crate::fibre_transport::Stage)->Entry {
        let programs=self.scene.programs.as_ref().expect("fibre entry needs compiled hair programs");
        let entry=stage.entry();
        let body:String=match stage {
            crate::fibre_transport::Stage::Accumulate=>"return fibre_accumulate<Programs>(v,front);".into(),
            crate::fibre_transport::Stage::Near=>"fibre_depth<Programs>(v,front);return uint2(v.object,v.curve_strand);".into(),
            crate::fibre_transport::Stage::Far=>"fibre_depth<Programs>(v,front);".into(),
        };
        let result=match stage {
            crate::fibre_transport::Stage::Accumulate=>"float4",
            crate::fibre_transport::Stage::Near=>"uint2", _=>"void"};
        let semantic=if result!="void" {":SV_Target0"} else {""};
        let text=format!("import fibre_capture;\n{}\n[shader(\"fragment\")] {result} {entry}(Varying v,bool front:SV_IsFrontFace,uint primitive:SV_PrimitiveID,out float depth:SV_Depth) {semantic} {{v.primitive=primitive;depth=fibre_capture_depth(v);{body}}}",programs.code);
        Entry {main:"fibre_programs".into(),text:Some(text),entry:(entry,"fragment"),threads:(0,0,0),kernel:None,program:None}
    }

    fn control_entry(&self,key:ForwardKey,program:Option<u32>)->Entry {
        let source=program.map(|program|&self.scene.programs.as_ref().unwrap().camera[program as usize-1]);
        Entry {main:if program.is_some() {key.name()} else {"scene_controls".into()},text:Some(key.module_with(source.map(|p|p.code.as_str()))),entry:key.entry(),threads:(0,0,0),kernel:None,program}
    }

    fn generic_shader(&self,stage:Stage,hashed:bool)->Option<&'static native::Compiled> {
        native::CONTROL_GENERICS.iter().find(|&&(a,d,s,h,_)|!self.scene.software_raster && !self.scene.field_controls && a==self.scene.attributes && d==self.scene.denoise && s==stage as usize && h==hashed).map(|v|v.4)
    }

    fn stream_shader(&self,main:&str,entry:&str)->Option<&'static native::Compiled> {
        native::STREAMS.iter().find(|&&(a,m,e,_)|a==self.scene.attributes && m==main && e==entry).map(|v|v.3)
    }

    fn stream_entry(&self,main:&str,entry:&'static str)->Entry {
        Entry {main:main.into(),text:None,entry:(entry,if entry.ends_with("vs") {"vertex"} else {"fragment"}),threads:(0,0,0),kernel:None,program:None}
    }

    fn program_controls(&self,stage:Stage,features:FeatureMask,program:u32)->anyhow::Result<[wgpu::ShaderModule;2]> {
        let key=ForwardKey::new(stage,features);
        let slot=self.authored_controls.lock().unwrap().entry((key,program)).or_default().clone();
        let fragment=try_init(&slot,|| {
            self.build_entry(&self.control_entry(ForwardKey::new(stage,features),Some(program)),false).map(Lookup::built)
        })?;
        Ok([self.camera.clone(),fragment.clone()])
    }

    fn table_controls(&self,stage:Stage,hashed:bool)->[wgpu::ShaderModule;2] {
        [self.camera.clone(),self.generic(stage,hashed)]
    }

}

impl VariantShader for SlangForward {
    fn geometry_input_key(&self,key:GeometryKey,lookup_only:bool)->anyhow::Result<blake3::Hash> {
        use crate::pipelines::CoverageStage;
        let mut hash=blake3::Hasher::new();
        let mut shipped=|shader:&native::Compiled| {hash.update(shader.input_key(self.metal).as_bytes());};
        let stream=|main,entry,base:&native::Compiled| {
            if self.scene.software_raster {
                if entry=="shadow_vs" {Ok(base.input_key(self.metal))}
                else {self.entry_key(&self.stream_entry(main,entry),lookup_only)}
            }
            else if self.scene.attributes==0 {Ok(base.input_key(self.metal))}
            else if let Some(shader)=self.stream_shader(main,entry) {Ok(shader.input_key(self.metal))}
            else {self.entry_key(&self.stream_entry(main,entry),lookup_only)}
        };
        match key.1 {
            Role::OpaqueDepth|Role::OpaqueSingleDepth if self.scene.software_raster=> {
                hash.update(self.entry_key(&self.stream_entry("depth","software_depth_vs"),lookup_only)?.as_bytes());
                let fragment=if key.1==Role::OpaqueDepth {"software_depth_fs"} else {"software_single_depth_fs"};
                hash.update(self.entry_key(&self.stream_entry("depth",fragment),lookup_only)?.as_bytes());
            },
            Role::Visibility|Role::VisibilitySingle if self.scene.software_raster=> {
                hash.update(self.entry_key(&self.stream_entry("decoupled_visibility","vs"),lookup_only)?.as_bytes());
                let fragment=if key.1==Role::Visibility {"fs"} else {"single_fs"};
                hash.update(self.entry_key(&self.stream_entry("decoupled_visibility",fragment),lookup_only)?.as_bytes());
            },
            Role::OpaqueDepth=>shipped(&native::DEPTH_OPAQUE_DEPTH_VS),
            Role::OpaqueSingleDepth=>{shipped(&native::DEPTH_OPAQUE_SINGLE_DEPTH_VS);shipped(&native::DEPTH_OPAQUE_SINGLE_DEPTH_FS);},
            Role::Visibility|Role::VisibilitySingle=> {
                shipped(&native::DECOUPLED_VISIBILITY_VS);
                shipped(if key.1==Role::Visibility {&native::DECOUPLED_VISIBILITY_FS} else {&native::DECOUPLED_VISIBILITY_SINGLE_FS});
            },
            Role::Fibre(stage)=> {
                hash.update(stream("depth","shadow_vs",&native::DEPTH_SHADOW_VS)?.as_bytes());
                hash.update(self.entry_key(&self.fibre_entry(stage),lookup_only)?.as_bytes());
            },
            Role::ProgramCoverage {program,stage}=> {
                let (main,entry,base,fragment)=match stage {
                    CoverageStage::Depth=>("depth","main_vs",&native::DEPTH_MAIN_VS,"depth_fs"),
                    CoverageStage::Visibility=>("decoupled_visibility","masked_vs",&native::DECOUPLED_VISIBILITY_MASKED_VS,"masked_fs"),
                };
                hash.update(stream(main,entry,base)?.as_bytes());
                hash.update(self.entry_key(&self.coverage_entry(fragment,program),lookup_only)?.as_bytes());
            },
            Role::MaskedDepth {hashed}=> {
                hash.update(stream("depth","main_vs",&native::DEPTH_MAIN_VS)?.as_bytes());
                let (entry,base)=if hashed {("hashed_depth_fs",&native::DEPTH_HASHED_DEPTH_FS)} else {("depth_fs",&native::DEPTH_DEPTH_FS)};
                hash.update(stream("depth",entry,base)?.as_bytes());
            },
            Role::VisibilityMasked|Role::TransparentFragments=> {
                hash.update(stream("decoupled_visibility","masked_vs",&native::DECOUPLED_VISIBILITY_MASKED_VS)?.as_bytes());
                let (entry,base)=if key.1==Role::VisibilityMasked {("masked_fs",&native::DECOUPLED_VISIBILITY_MASKED_FS)}
                    else if self.scene.field_controls || self.scene.software_raster {("transparent_fragments_fs",&native::DECOUPLED_VISIBILITY_TRANSPARENT_FRAGMENTS_FS)}
                    else {("transparent_triangles_fs",&native::DECOUPLED_VISIBILITY_TRANSPARENT_TRIANGLES_FS)};
                hash.update(stream("decoupled_visibility",entry,base)?.as_bytes());
            },
            Role::Controls(features)|Role::TransparentControls(features)|Role::ProgramControls {features,..}=> {
                hash.update(stream("depth","main_vs",&native::DEPTH_MAIN_VS)?.as_bytes());
                let transparent=matches!(key.1,Role::TransparentControls(_)|Role::ProgramControls {transparent:true,..});
                let stage=if transparent {Stage::Controls} else {Stage::OpaqueControls};
                let program=if let Role::ProgramControls {program,..}=key.1 {Some(program)} else {None};
                let fragment=if program.is_none() {
                    self.generic_shader(stage,ForwardKey::new(stage,features).hashed()).map(|shader|Ok(shader.input_key(self.metal)))
                        .unwrap_or_else(||self.entry_key(&self.control_entry(ForwardKey::generic(stage,ForwardKey::new(stage,features).hashed()),None),lookup_only))
                } else {self.entry_key(&self.control_entry(ForwardKey::new(stage,features),program),lookup_only)};
                hash.update(fragment?.as_bytes());
            },
        }
        Ok(hash.finalize())
    }
    fn kernel_input_key(&self,kernel:crate::pipelines::ProgramKernel,lookup_only:bool)->anyhow::Result<blake3::Hash> {
        if let Some((shader,capabilities))=self.surface_fixed(kernel) {
            let mut hash=blake3::Hasher::new();
            hash.update(shader.input_key(self.metal).as_bytes());
            hash.update(format!("{capabilities:?}").as_bytes());
            hash.update(b"surface-specialization-driver-inline-v1");
            return Ok(hash.finalize());
        }
        if let Some(shader)=self.path_engine_module(kernel).or_else(||self.camera_fixed(kernel)).or_else(||self.probe_fixed(kernel)) {
            return Ok(shader.input_key(self.metal));
        }
        let request=if matches!(kernel,crate::pipelines::ProgramKernel::Decoupled {..}) {self.camera_entry(kernel)} else {self.hit_entry(kernel)};
        self.entry_key(&request,lookup_only)
    }
    fn detached_request(&self, kernel: crate::pipelines::ProgramKernel) -> anyhow::Result<crate::specialization::Source> {
        anyhow::ensure!(self.path_engine(kernel).is_ok(), "only engine-backed path kernels detach");
        let request = self.hit_entry(kernel);
        self.scene.with_entry_inputs(self.metal,self.ray_shadows,self.msaa,&request.main,request.text.as_deref(),request.entry,request.kernel,request.program,|modules,args,identities,inputs,transforms| {
            anyhow::ensure!(inputs.is_empty(), "engine kernel has no authored input proofs");
            let compiler = vork_slang::Compiler::discover();
            let (identity, _) = programs::native_input_identity(self.cache.as_ref(), &compiler, modules, &request.main, &[request.entry], args, identities, inputs)?;
            // This process will not generate the kernel. Release the install's
            // retained capture before the CLI exits without running destructors.
            let receipt = programs::native_input_receipt_path(self.cache.as_ref(), &compiler, modules, &request.main, &[request.entry], args, identities, inputs)?;
            drop(take_prepared(&self.prepared, &receipt, inputs));
            let mut hash = transforms.prefix(if self.metal {request.threads} else {(0,0,0)});
            hash.update(&identity.identity);
            Ok(crate::specialization::Source {
                modules: modules.iter().map(|(n,s)|(n.to_string(),s.to_string())).collect(),
                args: args.iter().map(|s|s.to_string()).collect(), identities: identities.clone(),
                main: request.main.clone(), entry: request.entry.0.into(), threads: request.threads,
                metal: self.metal, transforms, identity: identity.identity, artifact: (self.cache)(hash.finalize()),
            })
        })
    }
    fn generic_shipped(&self) -> bool { self.scene.programs.is_none() }
    fn path_engine(&self,kernel:crate::pipelines::ProgramKernel)->Result<crate::pipelines::ProgramKernel,&'static str> {
        SlangForward::path_engine(self,kernel)
    }
    fn modules(&self, key: GeometryKey) -> anyhow::Result<[wgpu::ShaderModule; 2]> {
        Ok(match key.1 {
            Role::Fibre(stage) => {
                let index=stage.index();
                let fragment=try_init(&self.fibre[index],|| {
                    self.build_entry(&self.fibre_entry(stage),false).map(Lookup::built)
                })?;
                [self.fibre_vertex.clone(),fragment.clone()]
            },
            Role::ProgramCoverage {program,stage} => {
                use crate::pipelines::CoverageStage;
                let slot=self.authored_coverage.lock().unwrap().entry((program,stage)).or_default().clone();
                let (vertex,entry)=match stage {
                    CoverageStage::Depth=>(self.camera.clone(),"depth_fs"),
                    CoverageStage::Visibility=>(self.visibility[3].clone(),"masked_fs"),
                };
                [vertex,try_init(&slot,||self.build_entry(&self.coverage_entry(entry,program),false).map(Lookup::built))?.clone()]
            },

            // Position-only: no fragment stage.
            Role::OpaqueDepth => [self.opaque_depth.clone(), if self.scene.software_raster {self.build_entry(&self.stream_entry("depth","software_depth_fs"),false)?.built()} else {self.opaque_depth.clone()}],
            Role::OpaqueSingleDepth => self.single_depth.clone(),
            Role::Visibility | Role::VisibilitySingle | Role::VisibilityMasked | Role::TransparentFragments => {
                let [vertex, double, single, masked_vertex, masked, fragments] = &self.visibility;
                match key.1 {
                    Role::Visibility => [vertex.clone(), double.clone()],
                    Role::VisibilitySingle => [vertex.clone(), single.clone()],
                    Role::TransparentFragments => [masked_vertex.clone(), fragments.clone()],
                    _ => [masked_vertex.clone(), masked.clone()],
                }
            }
            Role::MaskedDepth {hashed} => [self.camera.clone(), self.masked_depth[usize::from(hashed)].clone()],
            Role::Controls(features) => self.table_controls(Stage::OpaqueControls,ForwardKey::new(Stage::OpaqueControls,features).hashed()),
            Role::TransparentControls(features) => self.table_controls(Stage::Controls,ForwardKey::new(Stage::Controls,features).hashed()),
            Role::ProgramControls {program,features,transparent} => self.program_controls(if transparent {Stage::Controls} else {Stage::OpaqueControls},features,program)?,
        })
    }
    fn camera_generic(&self, kernel: crate::pipelines::ProgramKernel) -> Option<&'static native::Compiled> {
        use crate::pipelines::ProgramKernel;
        // Coverage must use the installed material specialization, including
        // imported alpha and program output proofs. The opaque generic is not
        // a coverage fallback; an authored-table preflight cannot prove this.
        // Shipped evaluators link no shadow connection through transmitting
        // surfaces: they would shadow opaque receivers as if the glass were solid.
        if self.scene.surface_coverage || self.scene.transport_programs(self.ray_shadows) || self.scene.connection_transmission || matches!(kernel,ProgramKernel::Decoupled {boundaries,..} if boundaries>0) {return None;}
        let ProgramKernel::Decoupled {stage,features,..}=kernel else {return None;};
        if !stage.table() {return None;}
        self.camera_shipped(stage,features & crate::forward_variant::FEATURE_RECEIVER_DIFFUSION!=0)
    }
    fn program_kernel(&self, kernel: crate::pipelines::ProgramKernel) -> anyhow::Result<(wgpu::ShaderModule, &'static str)> {
        Ok(self.kernel_module(kernel, false)?.built())
    }
    fn kernel_stored(&self, kernel: crate::pipelines::ProgramKernel) -> anyhow::Result<bool> {
        Ok(!matches!(self.kernel_module(kernel, true)?, Lookup::Absent))
    }
    fn constants(&self, key: GeometryKey) -> Vec<(&'static str, f64)> {
        match key.1 {
            Role::Fibre(_) | Role::ProgramCoverage {..} => Vec::new(),

            Role::OpaqueDepth
            | Role::OpaqueSingleDepth
            | Role::Visibility
            | Role::VisibilitySingle
            | Role::VisibilityMasked
            | Role::TransparentFragments
            | Role::MaskedDepth {..}
            | Role::ProgramControls {..}
            | Role::Controls(_)
            | Role::TransparentControls(_) => Vec::new(),
        }
    }
}

impl SlangForward {
    fn probe_fixed(&self,kernel:crate::pipelines::ProgramKernel)->Option<&'static native::Compiled> {
        if let crate::pipelines::ProgramKernel::Probes {features:0,directional}=kernel
            && self.scene.programs.is_none() && self.scene.medium.is_none() && self.scene.attributes==0 {
            Some(if directional {&native::PROBE_UPDATE_DIRECTIONAL_TRACE} else {&native::PROBE_UPDATE_TRACE})
        } else {None}
    }

    fn hit_entry(&self,kernel:crate::pipelines::ProgramKernel)->Entry {
        use crate::pipelines::ProgramKernel;
        let (native, module, entry, kernel_call) = match kernel {
            ProgramKernel::PathEnvironment {adaptive} => return Entry {
                main:"path_environment_main".into(),
                text:Some(format!("import path; [shader(\"compute\")][numthreads(8,8,1)] void path_environment_main(uint3 gid:SV_DispatchThreadID) {{path_environment_pixel<{adaptive}>(gid);}}")),
                entry:("path_environment_main","compute"),threads:(8,8,1),kernel:Some(kernel),program:None,
            },
            // A scene's own kernel dispatches like the prebuilt engine it replaces.
            ProgramKernel::Path { adaptive: true, .. } => (Some(native::PATH_ENGINES[0].1), "path", "path_programs_main", "path_pixel<true,Key,"),
            ProgramKernel::Path { adaptive: false, .. } => (Some(native::PATH_ENGINES[0].1), "path", "path_programs_fixed", "path_pixel<false,Key,"),
            ProgramKernel::Gi { .. } => (Some(&native::GI_RAY_COMPUTE), "gi_ray", "gi_programs", "gi_pixel<Key,"),
            ProgramKernel::Reflections { .. } => (Some(&native::REFLECTIONS_HITS_COMPUTE), "reflections_hits", "reflections_programs", "reflections_pixel<Key,"),
            ProgramKernel::Probes {directional,..} => if directional {
                (Some(&native::PROBE_UPDATE_DIRECTIONAL_TRACE),"probe_update","probe_programs","probe_directional_trace<Key,")
            } else {(Some(&native::PROBE_UPDATE_TRACE),"probe_update","probe_programs","probe_trace<Key,")},
            ProgramKernel::Decoupled {..} => unreachable!("camera kernel handled above"),
        };
        let kernel_call=if let ProgramKernel::Path {random_walk,..}=kernel {
            kernel_call.replace("Key,",&format!("{random_walk},Key,"))
        } else {kernel_call.to_owned()};
        // Programs or named attributes (VORK_ATTRIBUTES) need a per-scene compile;
        // every other kernel but the probe update (precompiled with VORK_PROBES)
        // is specialized to the scene's features.
        let (program, code) = self.scene.programs.as_deref().map_or(("NoPrograms", ""), |p| ("Programs", p.code.as_str()));
        // The registered kernel's threadgroup, on both native targets.
        let threads = native.map_or((8,8,1),|n|n.threads);
        let (parameters, arguments) = match kernel {
            ProgramKernel::Probes { .. } => ("uint3 group:SV_GroupID,uint thread:SV_GroupIndex", "group,thread"),
            _ => ("uint3 gid:SV_DispatchThreadID", "gid"),
        };
        let type_arguments = match kernel {
            ProgramKernel::Path { boundaries, catchers, .. } => format!("{program},{boundaries},{catchers}"),
            _ => program.to_string(),
        };
        let text = format!(
            "// Generated by slang_forward.rs: {module} with the scene's programs and attributes at ray hits.\n\
             import vork_constants;\nimport {module};\nimport surface;\nimport frame;\n{code}\
             [shader(\"compute\")]\n[numthreads({},{},{})]\n\
             void {entry}({parameters}) {{{kernel_call}{type_arguments}>({arguments});}}\n",
            threads.0, threads.1, threads.2
        );
        let specialization = match kernel {
            ProgramKernel::Path { features, emitter_profiles, .. } => Some((features, emitter_profiles, false)),
            ProgramKernel::Reflections { features } | ProgramKernel::Gi { features } | ProgramKernel::Probes {features,..} => Some((features, false, false)),

            _ => None,
        };
        let text = if let Some((features, emitter_profiles, ray_shadows)) = specialization {
            crate::forward_variant::variant_module(features, ray_shadows, false, emitter_profiles, None, &text)
        } else { text };
        Entry {main:entry.into(),text:Some(text),entry:(entry,"compute"),threads,kernel:Some(kernel),program:None}
    }

    fn entry_key(&self,request:&Entry,lookup_only:bool)->anyhow::Result<blake3::Hash> {
        self.scene.entry_key(self.metal,self.ray_shadows,self.msaa,request,lookup_only,self.cache.as_ref(),&self.prepared)
    }

    /// A ray-hit or camera kernel's module, or with `probe` whether it needs no
    /// compile (shipped, or already in the store).
    fn kernel_module(&self, kernel: crate::pipelines::ProgramKernel, probe: bool) -> anyhow::Result<Lookup<(wgpu::ShaderModule, &'static str)>> {
        use crate::pipelines::ProgramKernel;
        let shipped = |shader: &'static native::Compiled| Ok(if probe {Lookup::Stored} else {Lookup::Built((native::shader_module(&self.device,shader),shader.entry))});
        if let Some((shader,capabilities))=self.surface_fixed(kernel) {
            return Ok(if probe {Lookup::Stored} else {Lookup::Built((native::surface_module(&self.device,shader,&capabilities)?,shader.entry))});
        }
        if let Some(shader)=self.path_engine_module(kernel) {return shipped(shader);}
        if let ProgramKernel::Decoupled {..}=kernel {
            if let Some(shader)=self.camera_fixed(kernel) {return shipped(shader);}
            let request=self.camera_entry(kernel);
            return Ok(self.build_entry(&request,probe)?.map(|module|(module,request.entry.0)));
        }
        if let Some(shader)=self.probe_fixed(kernel) {return shipped(shader);}
        let request=self.hit_entry(kernel);
        Ok(self.build_entry(&request,probe)?.map(|module|(module,request.entry.0)))
    }
}

/// A module request's result: built, or (probing) whether the store holds it.
pub(crate) enum Lookup<T> { Absent, Stored, Built(T) }
impl<T> Lookup<T> {
    fn built(self) -> T {
        match self { Self::Built(value) => value, _ => unreachable!("a compile request builds") }
    }
    fn map<U>(self, f: impl FnOnce(T) -> U) -> Lookup<U> {
        match self { Self::Built(value) => Lookup::Built(f(value)), Self::Stored => Lookup::Stored, Self::Absent => Lookup::Absent }
    }
}

pub(crate) use vork_slang::spirv::allow_driver_inlining;

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn shipped_surface_warm_keys_follow_executables() -> anyhow::Result<()> {
        use super::*;
        use crate::pipelines::ProgramKernel;
        let r = crate::test_renderer()?;
        if r.adapter_info.backend!=wgpu::Backend::Vulkan {return Ok(());}
        let base=SlangScene {probes:true,no_shadow_containment:true,..Default::default()};
        let a=SlangForward::new(&r,true,false,base.clone());
        let mut changed=base.clone();changed.denoise=true;changed.no_shadow_containment=true;
        changed.camera=Some(Default::default());
        let b=SlangForward::new(&r,true,false,changed);
        let kernel=ProgramKernel::Gi {features:3};
        assert!(a.kernel_input_key(kernel,true).ok().is_some());
        assert_eq!(a.kernel_input_key(kernel,true).ok(),b.kernel_input_key(kernel,true).ok());
        assert_ne!(a.kernel_input_key(kernel,true).ok(),a.kernel_input_key(ProgramKernel::Reflections {features:3},true).ok());
        assert_ne!(a.kernel_input_key(kernel,true).ok(),a.kernel_input_key(ProgramKernel::Gi {features:crate::forward_variant::FEATURE_SPECIALIZED},true).ok());
        let mut ordinary=base.clone();ordinary.textures=8;ordinary.samplers=4;
        let ordinary=SlangForward::new(&r,true,false,ordinary);
        assert_eq!(a.kernel_input_key(kernel,true).ok(),ordinary.kernel_input_key(kernel,true).ok(),"ordinary descriptor counts share setup's capacity");
        let mut textured=base.clone();textured.textures=vork_slang::registry::MATERIAL_TEXTURE_CAPACITY+1;
        let textures=SlangForward::new(&r,true,false,textured);
        assert_eq!(a.kernel_input_key(kernel,true).ok(),textures.kernel_input_key(kernel,true).ok(),"descriptor capacities belong to pipeline layouts, not shipped shader bytes");
        let mut thin=base.clone();thin.connection_transmission=true;thin.ray_refraction=true;
        let thin=SlangForward::new(&r,true,false,thin);
        assert_ne!(a.kernel_input_key(kernel,true).ok(),thin.kernel_input_key(kernel,true).ok(),"transmitting connections require the full hit reader");
        assert_eq!(a.surface_fixed(kernel).unwrap().0.name,"surface_gi_0_1_0_0");
        assert_eq!(thin.surface_fixed(kernel).unwrap().0.name,"surface_gi_0_1_1_0");
        let mut unshadowed=base.clone();unshadowed.connection_transmission=true;
        let unshadowed=SlangForward::new(&r,false,false,unshadowed);
        assert!(unshadowed.surface_fixed(kernel).is_none(),"compact shipped hit lighting requires the no-transmission proof independently of primary ray bindings");
        let transport=ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::Evaluate,
            features:3,subsurface:false,emitter_profiles:false,boundaries:1};
        assert!(thin.kernel_input_key(transport,true).ok().is_some(),"ordinary single-boundary transport is shipped");
        let mut unprobed=base.clone();unprobed.ray_refraction=true;unprobed.connection_transmission=true;unprobed.probes=false;
        let unprobed=SlangForward::new(&r,true,false,unprobed);
        assert!(unprobed.surface_fixed(transport).is_none(),"non-probe refraction retains its validated runtime executable");

        let scene=vork_scene::v2::from_test_value(serde_json::json!({"version":2,"nodes":[]}))?;
        let instance=programs::Instance {name:"thin-shadow-test".into(),file:None,
            source:Some("import vork; struct Inputs {}; Closure surface(ShadingContext ctx,Inputs p) {Closure c={};c.transmission_weight=1.;c.geometry_thin_walled=true;return c;}".into()),inputs:Default::default(),images:Vec::new()};
        let compiled=programs::install(&r,&scene,std::iter::once(("thin",&instance)),std::path::Path::new("."))?.unwrap();
        let mut authored=base.clone();authored.connection_transmission=true;authored.programs=compiled.native;
        let authored=SlangForward::new(&r,true,false,authored);
        assert!(authored.surface_fixed(ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::EvaluateOpaque,features:3,subsurface:false,emitter_profiles:false,boundaries:0}).is_none(),"authored transmitting shadows must keep their dispatcher");
        assert!(!authored.scene.programs.as_ref().unwrap().coverage,"constant coverage does not remove transmission");
        assert!(authored.scene.table_programs(Some(transport),authored.ray_shadows).is_some(),"table receivers must evaluate transmitting programs even at constant coverage");
        let mut contained=base.clone();contained.no_shadow_containment=false;
        let contained=SlangForward::new(&r,true,false,contained);
        assert!(contained.surface_fixed(kernel).is_none(),"shipped surfaces require the no-containment proof independently of candidate layout");
        let mut transmitting=base;transmitting.connection_transmission=true;transmitting.no_shadow_containment=false;
        let c=SlangForward::new(&r,true,false,transmitting);
        assert!(c.surface_fixed(kernel).is_none(),"transmission must retain its different executable");
        let camera=ProgramKernel::Decoupled {stage:crate::pipelines::ShadeStage::EvaluateOpaque,
            features:3,subsurface:false,emitter_profiles:false,boundaries:0};
        assert!(a.camera_generic(camera).is_some());
        assert!(c.camera_generic(camera).is_none(),"opaque receivers still need transmitting shadow connections");
        // The prebuilt path engine: one key per family, on this and every GPU, for
        // any table-material scene; no scene proof or material mix selects it.
        let table=SlangScene {no_points:true,camera:Some(Arc::new(crate::camera_program::Native {standard_camera:true,..Default::default()})),..Default::default()};
        let ordinary=SlangForward::new(&r,false,false,table.clone());
        for adaptive in [false,true] { for boundaries in [0,1] {
            let path=|features|ProgramKernel::Path {adaptive,random_walk:false,catchers:false,boundaries,media:None,features,emitter_profiles:false};
            let engine=path(crate::forward_variant::FEATURE_SPECIALIZED).path_engine().expect("a prebuilt family");
            for features in [crate::forward_variant::FEATURE_SPECIALIZED,
                crate::forward_variant::FEATURE_SPECIALIZED|crate::forward_variant::FEATURE_COAT|crate::forward_variant::FEATURE_COAT_ANISOTROPY|crate::forward_variant::FEATURE_TRANSMISSION] {
                assert_eq!(ordinary.path_engine(path(features)),Ok(engine),"every material mix draws the one engine key");
            }
            assert_eq!(ordinary.path_engine(engine),Ok(engine));
            assert_eq!(ordinary.path_engine_module(engine).expect("prebuilt module").name,crate::surface_native::PathEngine {adaptive,boundaries}.name());
            assert!(ordinary.path_engine_module(path(crate::forward_variant::FEATURE_SPECIALIZED)).is_none(),"a scene's own key is never the prebuilt module");
            assert_eq!(ordinary.kernel_input_key(engine,true).ok(),Some(ordinary.path_engine_module(engine).unwrap().input_key(ordinary.metal)));
            for proven in [SlangScene {unit_ior:true,unit_boundary_ior:true,no_gltf_tint:true,..table.clone()},
                SlangScene {no_shadow_containment:true,no_boundary_transactions:true,..table.clone()},
                SlangScene {connection_transmission:true,shadow_capacity:1,..table.clone()}] {
                assert_eq!(SlangForward::new(&r,false,false,proven).path_engine(path(crate::forward_variant::FEATURE_SPECIALIZED)),Ok(engine),"scene proofs do not select an engine");
            }
            assert!(ordinary.path_engine(path(crate::forward_variant::FEATURE_SPECIALIZED|crate::forward_variant::FEATURE_IMPLICIT_GEOMETRY)).is_err(),"implicit fields need their own kernel");
            let refused=|scene:SlangScene|SlangForward::new(&r,false,false,scene).path_engine(path(crate::forward_variant::FEATURE_SPECIALIZED)).is_err();
            assert!(!refused(SlangScene {surface_coverage:true,connection_transmission:true,..table.clone()}),"table cutouts and fractional opacity are material values the engine reads");
            assert!(refused(SlangScene {camera:Some(Default::default()),..table.clone()}),"authored cameras must not lose their entry");
            assert!(refused(SlangScene {programs:authored.scene.programs.clone(),..table.clone()}),"code materials need their evaluator");
            assert!(refused(SlangScene {shadow_capacity:2,..table.clone()}),"overlapping absorbing solids exceed the engine's shadow walk");
            assert!(refused(SlangScene {no_points:false,..table.clone()}),"points need the point-capable hit decoder");
            assert!(refused(SlangScene {attributes:1,..table.clone()}) && refused(SlangScene {aov_count:1,..table.clone()}));
        }}
        r.errors.check()
    }

    use super::*;

    /// Install-only measurement on wgpu's no-op backend: no GPU or frame work.
    #[test]
    #[ignore = "CPU shader install benchmark; set VORK_INSTALL_SCENES to a JSON array of scene paths"]
    fn cpu_shader_install_measurement() -> anyhow::Result<()> {
        let paths:Vec<PathBuf>=serde_json::from_str(&std::env::var("VORK_INSTALL_SCENES")?)?;
        let profile=std::env::var("VORK_INSTALL_PROFILE").unwrap_or_else(|_|"std/draft".into());
        let settings=vork_scene::Profile::load(&profile)?.resolve(&Default::default())?;
        let kind=std::env::var("VORK_INSTALL_KIND").unwrap_or_else(|_|"still".into());
        let renderer=pollster::block_on(crate::Renderer::create(crate::DeviceNeeds::ALL, true))?;
        for path in paths {
            let mut scene=crate::load_with_settings(&path,settings.clone(),&profile,None)?;
            let options=match kind.as_str() {
                "still"=>crate::shot::install_plan(&renderer,&mut scene,None,
                    crate::ReadbackOutputs {color:true,..Default::default()},true,false,&[],vork_time::Request::Numeric(2.5),Default::default())?.options,
                "check"=>crate::InstallOptions {variants:crate::pipelines::VariantCompile::Setup,..Default::default()},
                _=>anyhow::bail!("VORK_INSTALL_KIND must be still or check"),
            };
            let creations=crate::timing::creations(&renderer.device).len();
            let start=std::time::Instant::now();
            let installed=renderer.install_with_options(scene,path.parent().unwrap_or(std::path::Path::new(".")),
                options)?;
            let (passes,truncated)=vork_slang::progress::passes(start,std::time::Instant::now());
            anyhow::ensure!(!truncated,"install compiler timeline truncated");
            let frontend=passes.iter().filter(|(kind,_,_)|*kind==vork_slang::progress::Pass::FrontEnd).count();
            let codegen=passes.len()-frontend;
            let busy=|pass|passes.iter().filter(|(kind,_,_)|*kind==pass).map(|(_,start,end)|(*end-*start).as_secs_f64()*1000.).sum::<f64>();
            println!("CPU_INSTALL {}",serde_json::json!({"scene":path,"kind":kind,"install_ms":installed.install_time.as_secs_f64()*1000.,"pipeline_ms":installed.pipeline_ms,"frontend_passes":frontend,"codegen_passes":codegen,"frontend_busy_ms":busy(vork_slang::progress::Pass::FrontEnd),"codegen_busy_ms":busy(vork_slang::progress::Pass::CodeGeneration),"shader_creation":crate::timing::creations(&renderer.device).into_iter().skip(creations).collect::<Vec<_>>()}));
        }
        Ok(())
    }

    #[test]
    fn entry_keys_validate_imports_entry_target_and_transformations() -> anyhow::Result<()> {
        let root=std::env::temp_dir().join(format!("vork-entry-identity-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()));
        std::fs::create_dir(&root)?;
        let imported=root.join("identity_external.slang");
        std::fs::write(&imported,"public uint external_value() {return 1u;}\n")?;
        let scene=SlangScene {camera:Some(Arc::new(crate::camera_program::Native {modules:vec![],include:vec![root.clone()],..Default::default()})),..Default::default()};
        let cache=|key|root.join(format!("{key}.entry"));
        let mut entry=Entry {main:"identity_main".into(),text:Some("import identity_external; RWStructuredBuffer<uint> result; [shader(\"compute\")][numthreads(1,1,1)] void main() {result[0]=external_value();} [shader(\"compute\")][numthreads(1,1,1)] void other() {result[0]=external_value();}".into()),entry:("main","compute"),threads:(1,1,1),kernel:None,program:None};
        let prepared=Mutex::new(Default::default());
        let first=scene.entry_key(false,false,false,&entry,false,&cache,&prepared)?;
        assert_eq!(scene.entry_key(false,false,false,&entry,true,&cache,&prepared)?,first);
        assert_eq!(prepared.lock().unwrap().len(),1,"registry identity must retain its cold frontend snapshot");
        scene.with_entry_inputs(false,false,false,&entry.main,entry.text.as_deref(),entry.entry,entry.kernel,entry.program,|modules,args,identities,inputs,transforms| -> anyhow::Result<()> {
            let compiler=vork_slang::Compiler::discover();
            let receipt=programs::native_input_receipt_path(&cache,&compiler,modules,&entry.main,&[entry.entry],args,identities,inputs)?;
            let (identity,recaptured)=programs::native_input_identity(&cache,&compiler,modules,&entry.main,&[entry.entry],args,identities,inputs)?;
            assert!(recaptured.is_none());
            std::fs::remove_file(&receipt)?;
            let snapshot=take_prepared(&prepared,&receipt,inputs).expect("identity capture survives receipt eviction and transfers to compilation");
            let artifact=|identity:&[u8]| {let mut hash=transforms.prefix((0,0,0));hash.update(identity);cache(hash.finalize())};
            let code=programs::native_code(&identity,Some(snapshot),inputs,&receipt,artifact,
                ||anyhow::bail!("registry preparation must not run twice"),|snapshot|snapshot.compile())?;
            let fresh=compiler.prepare_identified_modules(modules,&entry.main,&[entry.entry],args,identities)?.compile()?;
            assert_eq!(code,fresh,"transferring the snapshot must preserve executable bytes");
            programs::native_input_identity(&cache,&compiler,modules,&entry.main,&[entry.entry],args,identities,inputs)?;
            Ok(())
        })?;
        // Unconsumed MSAA state must not split this compute entry.
        assert_eq!(scene.entry_key(false,false,true,&entry,false,&cache,&prepared)?,first);
        std::fs::write(&imported,"public uint external_value() {return 2u;}\n")?;
        assert!(scene.entry_key(false,false,false,&entry,true,&cache,&prepared).is_err(),"a stale resolver receipt must not schedule warm");
        let edited=scene.entry_key(false,false,false,&entry,false,&cache,&prepared)?;
        assert_ne!(edited,first,"external imported bytes belong to registry and remembered identities");
        entry.entry=("other","compute");
        assert_ne!(scene.entry_key(false,false,false,&entry,false,&cache,&prepared)?,edited,"entry point is consumed");
        entry.entry=("main","compute");
        let metal=scene.entry_key(true,false,false,&entry,false,&cache,&prepared)?;
        assert_ne!(metal,edited,"target and compiler options are consumed");
        entry.threads=(2,1,1);
        assert_eq!(scene.entry_key(false,false,false,&entry,false,&cache,&prepared)?,edited,"Vulkan gets its threadgroup from SPIR-V, not native module metadata");
        assert_ne!(scene.entry_key(true,false,false,&entry,false,&cache,&prepared)?,metal,"Metal consumes native threadgroup metadata");
        let pending:Vec<_>=prepared.lock().unwrap().keys().cloned().collect();
        assert!(!pending.is_empty());
        std::fs::write(&imported,"public uint external_value() {return 3u;}\n")?;
        for receipt in pending {
            assert!(take_prepared(&prepared,&receipt,&[]).is_none(),"an imported edit invalidates retained snapshots as well as disk receipts");
        }
        let ordinary=EntryTransforms {localize_blends:false,inline_metal:false,driver_inline:false}.prefix((1,1,1)).finalize();
        for transform in [EntryTransforms {localize_blends:true,inline_metal:false,driver_inline:false},EntryTransforms {localize_blends:false,inline_metal:true,driver_inline:false},EntryTransforms {localize_blends:false,inline_metal:false,driver_inline:true}] {
            assert_ne!(ordinary,transform.prefix((1,1,1)).finalize());
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn environment_program_does_not_change_path_transport_inputs() {
        use crate::pipelines::ProgramKernel;
        let base=SlangScene::default();
        let environment=crate::environment_program::Primary {
            spatial:false,uses_time:false,
            modules:vec![("camera_environment".into(),"public float3 primary_environment(uint base,float3 origin,float3 direction,float time) {return float3(1.);}".into())],
            include:vec![PathBuf::from("environment-imports")],words:vec![],
            layout:vork_slang::InputLayout {fields:vec![],words:0},material:"world".into(),image_data:None,constant_camera:None,
        };
        let scene=SlangScene {environment:Some(environment),..base.clone()};
        let mut edited=scene.clone();
        edited.environment.as_mut().unwrap().modules[0].1.push_str(" // another environment");
        assert!(scene.path_environment_prepass());
        let propagated=SlangScene {propagation:Some(crate::propagation::Native {modules:vec![],include:vec![],words:vec![]}),..scene.clone()};
        assert!(!propagated.path_environment_prepass());
        for adaptive in [false,true] {
            let kernel=Some(ProgramKernel::Path {adaptive,catchers:false,random_walk:false,features:0,boundaries:0,emitter_profiles:false,media:None});
            let main=if adaptive {"path_programs_main"} else {"path_programs_fixed"};
            assert_eq!(base.test_entry_inputs(main,(main,"compute"),kernel),scene.test_entry_inputs(main,(main,"compute"),kernel),
                "an environment program must not displace the prepared transport executable");
            assert_eq!(scene.test_entry_inputs(main,(main,"compute"),kernel),edited.test_entry_inputs(main,(main,"compute"),kernel));
            assert!(propagated.camera_environment(kernel).is_some(),"propagation evaluates the actual escape direction");
            let prepass=Some(ProgramKernel::PathEnvironment {adaptive});
            assert_ne!(scene.test_entry_inputs("path_environment_main",("path_environment_main","compute"),prepass),
                edited.test_entry_inputs("path_environment_main",("path_environment_main","compute"),prepass),
                "the separate camera evaluator must retain the environment's identity");
        }
    }

    #[test]
    fn camera_film_pruning_requires_the_whole_scene_proof() {
        use crate::pipelines::{ProgramKernel,ShadeStage};
        let unknown=SlangScene::default();
        let absent=SlangScene {no_film:true,..unknown.clone()};
        for stage in [ShadeStage::Evaluate,ShadeStage::EvaluateOpaque,ShadeStage::EvaluateProgram(1),ShadeStage::Fused] {
            // A receiver without film can still refract toward another material.
            // Conversely a widened fallback key can reserve film the scene lacks.
            for features in [crate::forward_variant::FEATURE_SPECIALIZED,crate::forward_variant::FEATURE_FILM] {
                let kernel=Some(ProgramKernel::Decoupled {stage,features,subsurface:false,emitter_profiles:false,boundaries:1});
                let args=|scene:&SlangScene|scene.test_entry_inputs("decoupled_programs",("evaluate","compute"),kernel);
                let (ordinary,key)=args(&unknown);
                let (pruned,pruned_key)=args(&absent);
                assert!(!ordinary.iter().any(|arg|arg=="-DVORK_NO_FILM"));
                assert!(pruned.iter().any(|arg|arg=="-DVORK_NO_FILM"));
                assert_ne!(key,pruned_key,"source pruning belongs to compiler input identity");
            }
        }
        for kernel in [ProgramKernel::Gi {features:0}] {
            let request=|scene:&SlangScene|scene.test_entry_inputs("gi_programs",("main","compute"),Some(kernel));
            let (ordinary,key)=request(&unknown);
            let (pruned,pruned_key)=request(&absent);
            assert!(!ordinary.iter().any(|arg|arg=="-DVORK_NO_FILM"));
            assert!(pruned.iter().any(|arg|arg=="-DVORK_NO_FILM"));
            assert_ne!(key,pruned_key,"secondary film pruning belongs to compiler input identity");
        }
        let reflections=Some(ProgramKernel::Reflections {features:0});
        assert_eq!(unknown.test_entry_inputs("reflections_programs",("main","compute"),reflections),
            absent.test_entry_inputs("reflections_programs",("main","compute"),reflections),
            "reflection compilation keeps its existing GPU form");
        let path=Some(ProgramKernel::Path {adaptive:false,catchers:false,random_walk:false,features:0,boundaries:1,emitter_profiles:false,media:None});
        assert_eq!(unknown.test_entry_inputs("path_programs_fixed",("path_programs_fixed","compute"),path),
            absent.test_entry_inputs("path_programs_fixed",("path_programs_fixed","compute"),path),
            "path compilation retains its existing installed-key proof");
    }

    #[test]
    fn flags_follow_the_entries_that_emit_them() {
        use crate::pipelines::ProgramKernel;
        let base=SlangScene::default();
        let path=ProgramKernel::Path {adaptive:false,catchers:false,random_walk:false,features:0,boundaries:0,emitter_profiles:false,media:None};
        for (main,entry,kernel,is_compute,is_path) in [
            ("depth",("main_vs","vertex"),None,false,false),
            ("scene_controls",("controls_fs","fragment"),None,false,false),
            ("program_depth",("masked_fs","fragment"),None,false,false),
            ("gi_programs",("gi_programs","compute"),Some(ProgramKernel::Gi {features:0}),true,false),
            ("reflections_programs",("reflections_programs","compute"),Some(ProgramKernel::Reflections {features:0}),true,false),
            ("path_programs_fixed",("path_programs_fixed","compute"),Some(path),true,true),
        ] {
            let (_,ordinary)=base.test_entry_inputs(main,entry,kernel);
            for (changed,define,consumed) in [
                (SlangScene {no_boundary_transactions:true,..base.clone()},"-DVORK_NO_BOUNDARY_TRANSACTIONS",is_path),
                (SlangScene {probes:true,..base.clone()},"-DVORK_PROBES",is_compute),
                (SlangScene {unit_ior:true,..base.clone()},"-DVORK_UNIT_IOR",true),
                (SlangScene {unit_boundary_ior:true,..base.clone()},"-DVORK_UNIT_BOUNDARY_IOR",true),
                (SlangScene {no_gltf_tint:true,..base.clone()},"-DVORK_NO_GLTF_TINT",true),
            ] {
                let (args,key)=changed.test_entry_inputs(main,entry,kernel);
                assert_eq!(args.iter().any(|arg|arg==define),consumed,"{main}: {define}");
                assert_eq!(ordinary!=key,consumed,"identity must follow emitted input: {main}: {define}");
            }
            for changed in [SlangScene {field_controls:true,..base.clone()},SlangScene {connection_transmission:true,..base.clone()}] {
                assert_ne!(ordinary,changed.test_entry_inputs(main,entry,kernel).1,"{main}: consumed code/outlining input");
            }
        }
    }

    /// Metal refuses an array as a vertex attribute or a fragment varying. A
    /// scene with more than eight named attributes (a Gaussian capture with SH
    /// degree 2 or 3 binds 9 or 16) failed to create its raster stream modules.
    #[test]
    fn metal_stage_inputs_past_eight_attributes_are_not_arrays() -> anyhow::Result<()> {
        for attributes in [9,16,19] {
            let scene=SlangScene {attributes,..Default::default()};
            for entry in [("main_vs","vertex"),("shadow_vs","vertex"),("depth_fs","fragment")] {
                let code=scene.with_entry_inputs(true,false,false,"depth",None,entry,None,None,|modules,args,_,_,_| {
                    assert!(args.contains(&"-DVORK_STAGE_SCALARS"),"{args:?}");
                    vork_slang::Compiler::discover().compile_modules(modules,"depth",&[entry],args)
                })?;
                let source=String::from_utf8(code)?;
                vork_slang::metal_source::validate(&source)?;
                for line in source.lines().filter(|line|line.contains("[[attribute(") || line.contains("[[user(")) {
                    assert!(!line.contains("array<"),"{attributes} attributes, {}: {line}",entry.0);
                }
                if entry.0=="main_vs" {
                    let varyings=source.lines().filter(|line|line.contains("[[user(VARYING_ATTRIBUTE")).count();
                    assert_eq!(varyings,attributes as usize,"one varying per named attribute");
                }
            }
            let (vulkan,_)=scene.test_entry_inputs("depth",("main_vs","vertex"),None);
            assert!(!vulkan.iter().any(|arg|arg=="-DVORK_STAGE_SCALARS"),"{vulkan:?}");
        }
        let (eight,_)=SlangScene {attributes:8,..Default::default()}.test_entry_inputs("depth",("main_vs","vertex"),None);
        assert!(!eight.iter().any(|arg|arg=="-DVORK_STAGE_SCALARS"));
        Ok(())
    }

    #[test]
    fn driver_inlining_clears_only_dont_inline() {
        let words = [0x07230203u32, 0x00010500, 0, 32, 0,
            (5 << 16) | 54, 1, 2, 0xe, 3, (1 << 16) | 56,
            (3 << 16) | 71, 2, 42]; // Preserve NoContraction decorations.
        let mut code: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        allow_driver_inlining(&mut code).unwrap();
        let mut expected = words;
        expected[8] = 0xc; // Preserve Pure and Const; remove only DontInline.
        assert_eq!(code, expected.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
        for tail in [vec![0u32], vec![(5 << 16) | 54, 1]] {
            let mut bad: Vec<u8> = words[..5].iter().copied().chain(tail).flat_map(u32::to_le_bytes).collect();
            assert!(allow_driver_inlining(&mut bad).is_err());
        }
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn camera_profile_follows_installed_transport() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let kernel=crate::pipelines::ProgramKernel::Decoupled {
            stage:crate::pipelines::ShadeStage::Classify, features:0,
            subsurface:false, emitter_profiles:false, boundaries:0,
        };
        for rays in [false,true] {
            let shader=SlangForward::new(&r,rays,false,SlangScene::default());
            assert_eq!(shader.camera_fixed(kernel).unwrap().name,
                format!("camera_0_{}_classify",u8::from(rays)));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn concurrent_layout_requests_share_one_fragment_module() -> anyhow::Result<()> {
        let r = crate::test_renderer()?;
        let shader = SlangForward::new(&r, false, false, SlangScene::default());
        let key = ForwardKey::new(Stage::OpaqueControls, 0);
        let barrier = std::sync::Barrier::new(4);
        let modules = std::thread::scope(|scope| {
            (0..4)
                .map(|_| scope.spawn(|| {
                    barrier.wait();
                    shader.generic(key.stage,false)
                }))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|job| job.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(modules.iter().all(|module| module == &modules[0]));
        assert_eq!(shader.generic(key.stage,false), modules[0]);
        Ok(())
    }
}
