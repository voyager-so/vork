use alloc::{borrow::ToOwned as _, string::String, sync::Arc, vec::Vec};
use core::ptr::NonNull;
use parking_lot::Mutex;
use std::time;

use bytemuck::TransparentWrapper;
use objc2::{
    available,
    rc::{autoreleasepool, Retained},
    runtime::ProtocolObject,
};
use objc2_foundation::{ns_string, NSArray, NSError, NSProcessInfo, NSRange, NSString, NSUInteger, NSURL};
use objc2_metal::{
    MTLAccelerationStructure, MTLAccelerationStructureInstanceOptions, MTLBinaryArchive,
    MTLBinaryArchiveDescriptor, MTLBuffer,
    MTLCaptureManager, MTLCaptureScope, MTLCommandBuffer, MTLCommandBufferStatus,
    MTLCompileOptions, MTLComputePipelineDescriptor, MTLComputePipelineState,
    MTLCounterSampleBufferDescriptor, MTLCounterSet, MTLDepthClipMode, MTLDepthStencilDescriptor,
    MTLDevice, MTLFunction, MTLIndirectAccelerationStructureInstanceDescriptor, MTLLanguageVersion,
    MTLLibrary, MTLMeshRenderPipelineDescriptor, MTLMutability, MTLPackedFloat3, MTLPackedFloat4x3,
    MTLPipelineBufferDescriptorArray, MTLPipelineOption, MTLPixelFormat, MTLPrimitiveTopologyClass,
    MTLRenderPipelineColorAttachmentDescriptorArray, MTLRenderPipelineDescriptor, MTLResource,
    MTLResourceID, MTLResourceOptions, MTLSamplerAddressMode, MTLSamplerDescriptor,
    MTLSamplerMipFilter, MTLSamplerState, MTLSize, MTLStencilDescriptor, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTriangleFillMode, MTLVertexDescriptor,
    MTLVertexStepFunction,
};

use super::{compile_admission, conv, PassthroughShader, ShaderModuleSource};
use crate::{auxil::map_naga_stage, TlasInstance};

type DeviceResult<T> = Result<T, crate::DeviceError>;

// Tier-2 argument buffers contain one 64-bit resource ID or device address per
// declared slot, even when PARTIALLY_BOUND_BINDING_ARRAY supplies only a prefix.
fn new_argument_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    count: core::num::NonZeroU32,
) -> Retained<ProtocolObject<dyn MTLBuffer>> {
    device
        .newBufferWithLength_options(
            size_of::<MTLResourceID>() * count.get() as usize,
            MTLResourceOptions::HazardTrackingModeUntracked | MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
}

fn fill_argument_buffer_tail<T: Copy>(contents: &mut [T], supplied: usize) {
    // Core rejects empty binding arrays. Reuse a supplied, already tracked
    // resource so every tail slot has a valid handle, including samplers.
    assert!(supplied > 0 && supplied <= contents.len());
    let fallback = contents[0];
    contents[supplied..].fill(fallback);
}

/// Slang's single-field ParameterBlock tables have flexible resource tails.
/// Compile them at pipeline creation using the *declared* binding capacity,
/// never the device limit or the populated prefix. Fixed-size tables and data
/// buffers are not rewritten. Reject unfamiliar declarations rather than guess
/// an ABI. This accepts the pinned Slang form `Table constant* name [[buffer(N)]]`.
fn specialize_argument_arrays(source: &str, counts: &[(u32, u32)]) -> Result<String, String> {
    let mut edits = Vec::new();
    for (start, _) in source.match_indices("struct ") {
        let rest = &source[start + "struct ".len()..];
        let Some((head, rest)) = rest.split_once('{') else { continue };
        if head.contains(';') { continue; } // A forward declaration, not a definition.
        let Some((body, _)) = rest.split_once('}') else { continue };
        if !body.contains("[];") { continue; }
        let name = head.trim();
        let field = body.trim();
        if name.split_whitespace().count() != 1 || field.matches(';').count() != 1
            || !(field.starts_with("texture") || field.starts_with("depth")
                || field.starts_with("sampler ") || field.contains(" device*"))
        {
            return Err(format!("Unsupported flexible Metal argument table: {name}"));
        }
        let parameter = format!("{name} constant*");
        let mut capacity = None;
        for tail in source.split(&parameter).skip(1) {
            // A helper's table pointer has no buffer attribute. Only an entry
            // point's physical slot connects this declaration to the layout.
            let tail = tail.trim_start();
            let Some((_, after_name)) = tail.split_once(char::is_whitespace) else { continue };
            let Some(slot) = after_name.trim_start().strip_prefix("[[buffer(") else { continue };
            let slot: u32 = slot.split_once(")]]").ok_or("Malformed Metal buffer attribute")?.0
                .parse().map_err(|_| "Malformed Metal buffer slot")?;
            let count = counts.iter().find(|&&(index, _)| index == slot)
                .ok_or_else(|| format!("Flexible Metal table {name} at buffer {slot} has no binding-array capacity"))?.1;
            if count == 0 || capacity.is_some_and(|previous| previous != count) {
                return Err(format!("Conflicting Metal argument capacities for {name}"));
            }
            capacity = Some(count);
        }
        let count = capacity.ok_or_else(|| format!("Flexible Metal table {name} has no bound entry parameter"))?;
        let offset = source.len() - rest.len() + body.find("[];").unwrap();
        edits.push((offset, count));
    }
    let mut output = source.to_owned();
    for (offset, count) in edits.into_iter().rev() {
        output.replace_range(offset..offset + 2, &format!("[{count}]"));
    }
    if output.contains("[];") {
        return Err("Unresolved flexible Metal array declaration".into());
    }
    Ok(output)
}

struct CompiledShader {
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    /// Bytes of source behind `library`: what compiling a pipeline state from it is charged.
    source_bytes: u64,
    function: Retained<ProtocolObject<dyn MTLFunction>>,
    wg_size: MTLSize,
    wg_memory_sizes: Vec<u32>,

    /// Bindings of WGSL `storage` globals that contain variable-sized arrays.
    ///
    /// In order to implement bounds checks and the `arrayLength` function for
    /// WGSL runtime-sized arrays, we pass the entry point a struct with a
    /// member for each global variable that contains such an array. That member
    /// is a `u32` holding the variable's total size in bytes---which is simply
    /// the size of the `Buffer` supplying that variable's contents for the
    /// draw call.
    sized_bindings: Vec<naga::ResourceBinding>,

    immutable_buffer_mask: usize,
}

fn create_stencil_desc(
    face: &wgt::StencilFaceState,
    read_mask: u32,
    write_mask: u32,
) -> Retained<MTLStencilDescriptor> {
    let desc = MTLStencilDescriptor::new();
    desc.setStencilCompareFunction(conv::map_compare_function(face.compare));
    desc.setReadMask(read_mask);
    desc.setWriteMask(write_mask);
    desc.setStencilFailureOperation(conv::map_stencil_op(face.fail_op));
    desc.setDepthFailureOperation(conv::map_stencil_op(face.depth_fail_op));
    desc.setDepthStencilPassOperation(conv::map_stencil_op(face.pass_op));
    desc
}

fn create_depth_stencil_desc(
    state: &wgt::DepthStencilState,
) -> Retained<MTLDepthStencilDescriptor> {
    let desc = MTLDepthStencilDescriptor::new();
    desc.setDepthCompareFunction(conv::map_compare_function(
        state.depth_compare.unwrap_or_default(),
    ));
    desc.setDepthWriteEnabled(state.depth_write_enabled.unwrap_or_default());
    let s = &state.stencil;
    if s.is_enabled() {
        let front_desc = create_stencil_desc(&s.front, s.read_mask, s.write_mask);
        desc.setFrontFaceStencil(Some(&front_desc));
        let back_desc = create_stencil_desc(&s.back, s.read_mask, s.write_mask);
        desc.setBackFaceStencil(Some(&back_desc));
    }
    desc
}

const fn convert_vertex_format_to_naga(format: wgt::VertexFormat) -> naga::back::msl::VertexFormat {
    match format {
        wgt::VertexFormat::Uint8 => naga::back::msl::VertexFormat::Uint8,
        wgt::VertexFormat::Uint8x2 => naga::back::msl::VertexFormat::Uint8x2,
        wgt::VertexFormat::Uint8x4 => naga::back::msl::VertexFormat::Uint8x4,
        wgt::VertexFormat::Sint8 => naga::back::msl::VertexFormat::Sint8,
        wgt::VertexFormat::Sint8x2 => naga::back::msl::VertexFormat::Sint8x2,
        wgt::VertexFormat::Sint8x4 => naga::back::msl::VertexFormat::Sint8x4,
        wgt::VertexFormat::Unorm8 => naga::back::msl::VertexFormat::Unorm8,
        wgt::VertexFormat::Unorm8x2 => naga::back::msl::VertexFormat::Unorm8x2,
        wgt::VertexFormat::Unorm8x4 => naga::back::msl::VertexFormat::Unorm8x4,
        wgt::VertexFormat::Snorm8 => naga::back::msl::VertexFormat::Snorm8,
        wgt::VertexFormat::Snorm8x2 => naga::back::msl::VertexFormat::Snorm8x2,
        wgt::VertexFormat::Snorm8x4 => naga::back::msl::VertexFormat::Snorm8x4,
        wgt::VertexFormat::Uint16 => naga::back::msl::VertexFormat::Uint16,
        wgt::VertexFormat::Uint16x2 => naga::back::msl::VertexFormat::Uint16x2,
        wgt::VertexFormat::Uint16x4 => naga::back::msl::VertexFormat::Uint16x4,
        wgt::VertexFormat::Sint16 => naga::back::msl::VertexFormat::Sint16,
        wgt::VertexFormat::Sint16x2 => naga::back::msl::VertexFormat::Sint16x2,
        wgt::VertexFormat::Sint16x4 => naga::back::msl::VertexFormat::Sint16x4,
        wgt::VertexFormat::Unorm16 => naga::back::msl::VertexFormat::Unorm16,
        wgt::VertexFormat::Unorm16x2 => naga::back::msl::VertexFormat::Unorm16x2,
        wgt::VertexFormat::Unorm16x4 => naga::back::msl::VertexFormat::Unorm16x4,
        wgt::VertexFormat::Snorm16 => naga::back::msl::VertexFormat::Snorm16,
        wgt::VertexFormat::Snorm16x2 => naga::back::msl::VertexFormat::Snorm16x2,
        wgt::VertexFormat::Snorm16x4 => naga::back::msl::VertexFormat::Snorm16x4,
        wgt::VertexFormat::Float16 => naga::back::msl::VertexFormat::Float16,
        wgt::VertexFormat::Float16x2 => naga::back::msl::VertexFormat::Float16x2,
        wgt::VertexFormat::Float16x4 => naga::back::msl::VertexFormat::Float16x4,
        wgt::VertexFormat::Float32 => naga::back::msl::VertexFormat::Float32,
        wgt::VertexFormat::Float32x2 => naga::back::msl::VertexFormat::Float32x2,
        wgt::VertexFormat::Float32x3 => naga::back::msl::VertexFormat::Float32x3,
        wgt::VertexFormat::Float32x4 => naga::back::msl::VertexFormat::Float32x4,
        wgt::VertexFormat::Uint32 => naga::back::msl::VertexFormat::Uint32,
        wgt::VertexFormat::Uint32x2 => naga::back::msl::VertexFormat::Uint32x2,
        wgt::VertexFormat::Uint32x3 => naga::back::msl::VertexFormat::Uint32x3,
        wgt::VertexFormat::Uint32x4 => naga::back::msl::VertexFormat::Uint32x4,
        wgt::VertexFormat::Sint32 => naga::back::msl::VertexFormat::Sint32,
        wgt::VertexFormat::Sint32x2 => naga::back::msl::VertexFormat::Sint32x2,
        wgt::VertexFormat::Sint32x3 => naga::back::msl::VertexFormat::Sint32x3,
        wgt::VertexFormat::Sint32x4 => naga::back::msl::VertexFormat::Sint32x4,
        wgt::VertexFormat::Unorm10_10_10_2 => naga::back::msl::VertexFormat::Unorm10_10_10_2,
        wgt::VertexFormat::Unorm8x4Bgra => naga::back::msl::VertexFormat::Unorm8x4Bgra,

        wgt::VertexFormat::Float64
        | wgt::VertexFormat::Float64x2
        | wgt::VertexFormat::Float64x3
        | wgt::VertexFormat::Float64x4 => {
            unimplemented!()
        }
    }
}

/// A unique temporary path for binary archive bytes handed between Metal and wgpu.
fn archive_path() -> std::path::PathBuf {
    static NEXT: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);
    let next = NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(alloc::format!("wgpu-metal-archive-{}-{next}.bin", std::process::id()))
}

fn file_url(path: &std::path::Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

/// Create a pipeline through a binary archive. A hit loads the stored binary
/// without compiling, so it is not admitted. A miss compiles exactly as without
/// an archive, under the process's compile admission, then records the result
/// in the archive so the next process that supplies the serialized archive hits,
/// and counts the addition. It never waits for a serialization: an addition that
/// meets one is deferred to whoever next holds the archive's lock, the next
/// serialization included, and is recorded there under its own admission.
fn through_archive<P>(
    cache: Option<&super::PipelineCache>,
    additions: &core::sync::atomic::AtomicUsize,
    admission: &compile_admission::Admission,
    label: crate::Label<'_>,
    source_bytes: u64,
    create: impl Fn(Option<&NSArray<ProtocolObject<dyn MTLBinaryArchive>>>, MTLPipelineOption) -> Result<P, Retained<NSError>>,
    add: impl Fn(&ProtocolObject<dyn MTLBinaryArchive>) -> Result<(), Retained<NSError>> + Clone + 'static,
) -> Result<P, String> {
    let compile = || {
        create(None, MTLPipelineOption::empty())
            .map_err(|error| compile_admission::Failure::new(&error, alloc::format!("{error:?}")))
    };
    let Some(cache) = cache else {
        return admission.compile(label, source_bytes, compile);
    };
    let archives = NSArray::from_retained_slice(&[cache.archive.clone()]);
    if let Ok(pipeline) = create(Some(&archives), MTLPipelineOption::FailOnBinaryArchiveMiss) {
        return Ok(pipeline);
    }
    // Recording a pipeline in the archive is a compiler request too: it reuses
    // the binary just compiled only while the driver still holds it. Both run
    // under one admission, and a retry repeats both.
    let pipeline = admission.compile(label, source_bytes, || {
        let pipeline = compile()?;
        // An addition is counted only once it is in the archive or in the deferred queue.
        // A serialization drains that queue under the lock, so a caller that reads the
        // count and then serializes has written every addition the count included.
        // The lock is only tried: an admitted compilation never waits for it, which
        // is what lets a drain wait for admission while holding it.
        let Some(_guard) = cache.files.try_lock() else {
            let (archive, add) = (cache.archive.clone(), add.clone());
            cache.deferred.lock().push(super::DeferredAddition(compile_admission::Deferred::new(label, source_bytes, move || {
                add(&archive).map_err(|error| compile_admission::Failure::new(&error, alloc::format!("{error:?}")))
            })));
            additions.fetch_add(1, core::sync::atomic::Ordering::Release);
            return Ok(pipeline);
        };
        add(&cache.archive).map_err(|error| {
            compile_admission::Failure::new(
                &error,
                alloc::format!("the pipeline compiled but the Metal binary archive refused it: {error:?}"),
            )
        })?;
        additions.fetch_add(1, core::sync::atomic::Ordering::Release);
        Ok(pipeline)
    })?;
    // The permit is released. Record what earlier pipelines deferred, each under
    // its own admission; never from inside an admitted attempt, whose permit
    // would account for the wrong work and could wait on itself.
    if let Some(_guard) = cache.files.try_lock() {
        cache.record_deferred(admission);
    }
    Ok(pipeline)
}

/// FNV-1a over the inputs, seeded so two seeds give 16 independent bytes.
fn fnv(seed: u64, parts: &[&[u8]]) -> u64 {
    parts.iter().flat_map(|part| part.iter().chain(&[0xff])).fold(seed, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl super::Device {
    fn compile_passthrough_msl(&self, label: crate::Label<'_>, source: &str) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
        let options = MTLCompileOptions::new();
        // Match Naga modules: honor invariant vertex positions in passthrough MSL.
        if available!(macos = 11.0, ios = 13.0, tvos = 14.0, visionos = 1.0) {
            options.setPreserveInvariance(true);
        }
        // The source pragma makes arithmetic safe, but metal_stdlib picks
        // its function implementations (sqrt, rsqrt, trig, normalize...)
        // from the compile options. Fast ones carry fast-math flags, so
        // their results depend on the surrounding code. A module that
        // declares safe math compiles with precise library functions.
        if source
            .lines()
            .take_while(|line| line.starts_with("#pragma"))
            .any(|line| line.trim() == "#pragma METAL fp math_mode(safe)")
        {
            if available!(macos = 15.0, ios = 18.0, tvos = 18.0, visionos = 2.0) {
                options.setMathMode(objc2_metal::MTLMathMode::Safe);
                options.setMathFloatingPointFunctions(
                    objc2_metal::MTLMathFloatingPointFunctions::Precise,
                );
            } else {
                #[allow(deprecated)]
                options.setFastMathEnabled(false);
            }
        }
        // Obtain the device from shared
        let device = &self.shared.device;
        let source_string = NSString::from_str(source);
        compile_admission::service().compile(label, source.len() as u64, || {
            device
                .newLibraryWithSource_options_error(&source_string, Some(&options))
                .map_err(|e| compile_admission::Failure::new(&e, format!("MSL: {e:?}")))
        })
    }

    fn load_shader(
        &self,
        stage: &crate::ProgrammableStage<super::ShaderModule>,
        vertex_buffer_mappings: &[naga::back::msl::VertexBufferMapping],
        layout: &super::PipelineLayout,
        primitive_class: MTLPrimitiveTopologyClass,
        naga_stage: naga::ShaderStage,
        label: crate::Label<'_>,
    ) -> Result<CompiledShader, crate::PipelineError> {
        match stage.module.source {
            ShaderModuleSource::Naga(ref naga_shader) => {
                let stage_bit = map_naga_stage(naga_stage);
                let (module, module_info) = naga::back::pipeline_constants::process_overrides(
                    &naga_shader.module,
                    &naga_shader.info,
                    Some((naga_stage, stage.entry_point)),
                    stage.constants,
                )
                .map_err(|e| {
                    crate::PipelineError::PipelineConstants(stage_bit, format!("MSL: {e:?}"))
                })?;

                let ep_resources = &layout.per_stage_map[naga_stage];

                let bounds_check_policy = if stage.module.bounds_checks.bounds_checks {
                    naga::proc::BoundsCheckPolicy::Restrict
                } else {
                    naga::proc::BoundsCheckPolicy::Unchecked
                };

                let options = naga::back::msl::Options {
                    lang_version: match self.shared.private_caps.msl_version {
                        #[allow(deprecated)]
                        MTLLanguageVersion::Version1_0 => (1, 0),
                        MTLLanguageVersion::Version1_1 => (1, 1),
                        MTLLanguageVersion::Version1_2 => (1, 2),
                        MTLLanguageVersion::Version2_0 => (2, 0),
                        MTLLanguageVersion::Version2_1 => (2, 1),
                        MTLLanguageVersion::Version2_2 => (2, 2),
                        MTLLanguageVersion::Version2_3 => (2, 3),
                        MTLLanguageVersion::Version2_4 => (2, 4),
                        MTLLanguageVersion::Version3_0 => (3, 0),
                        MTLLanguageVersion::Version3_1 => (3, 1),
                        MTLLanguageVersion::Version3_2 => (3, 2),
                        // Newer version, fall back to 3.2
                        _ => (3, 2),
                    },
                    inline_samplers: Default::default(),
                    spirv_cross_compatibility: false,
                    fake_missing_bindings: false,
                    per_entry_point_map: naga::back::msl::EntryPointResourceMap::from([(
                        stage.entry_point.to_owned(),
                        ep_resources.clone(),
                    )]),
                    bounds_check_policies: naga::proc::BoundsCheckPolicies {
                        index: bounds_check_policy,
                        buffer: bounds_check_policy,
                        image_load: bounds_check_policy,
                        // TODO: support bounds checks on binding arrays
                        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
                    },
                    zero_initialize_workgroup_memory: stage.zero_initialize_workgroup_memory,
                    force_loop_bounding: stage.module.bounds_checks.force_loop_bounding,
                };

                let pipeline_options = naga::back::msl::PipelineOptions {
                    entry_point: Some((naga_stage, stage.entry_point.to_owned())),
                    allow_and_force_point_size: match primitive_class {
                        MTLPrimitiveTopologyClass::Point => true,
                        _ => false,
                    },
                    vertex_pulling_transform: true,
                    vertex_buffer_mappings: vertex_buffer_mappings.to_vec(),
                };

                let (source, info) = naga::back::msl::write_string(
                    &module,
                    &module_info,
                    &options,
                    &pipeline_options,
                )
                .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("MSL: {e:?}")))?;

                log::debug!(
                    "Naga generated shader for entry point '{}' and stage {:?}\n{}",
                    stage.entry_point,
                    naga_stage,
                    &source
                );

                let options = MTLCompileOptions::new();
                options.setLanguageVersion(self.shared.private_caps.msl_version);
                // WGSL defines no fast-math license, and SPIR-V backends do not
                // reassociate it. Metal's default fast math let GPU-composed
                // placements differ from the same CPU-authored transforms.
                if available!(macos = 15.0, ios = 18.0, tvos = 18.0, visionos = 2.0) {
                    options.setMathMode(objc2_metal::MTLMathMode::Safe);
                    options.setMathFloatingPointFunctions(
                        objc2_metal::MTLMathFloatingPointFunctions::Precise,
                    );
                } else {
                    #[allow(deprecated)]
                    options.setFastMathEnabled(false);
                }

                // https://developer.apple.com/documentation/metal/mtlcompileoptions/preserveinvariance
                if available!(macos = 11.0, ios = 13.0, tvos = 14.0, visionos = 1.0) {
                    options.setPreserveInvariance(true);
                }

                let source_string = NSString::from_str(&source);
                let library = compile_admission::service()
                    .compile(label, source.len() as u64, || {
                        self.shared
                            .device
                            .newLibraryWithSource_options_error(&source_string, Some(&options))
                            .map_err(|err| compile_admission::Failure::new(&err, format!("Metal: {err}")))
                    })
                    .map_err(|err| {
                        log::debug!("Naga generated shader:\n{source}");
                        crate::PipelineError::Linkage(stage_bit, err)
                    })?;

                let ep_index = module
                    .entry_points
                    .iter()
                    .position(|ep| ep.stage == naga_stage && ep.name == stage.entry_point)
                    .ok_or(crate::PipelineError::EntryPoint(naga_stage))?;
                let ep = &module.entry_points[ep_index];
                let translated_ep_name = info.entry_point_names[0]
                    .as_ref()
                    .map_err(|e| crate::PipelineError::Linkage(stage_bit, format!("{e}")))?;

                let wg_size = MTLSize {
                    width: ep.workgroup_size[0] as _,
                    height: ep.workgroup_size[1] as _,
                    depth: ep.workgroup_size[2] as _,
                };

                let function = library
                    .newFunctionWithName(&NSString::from_str(translated_ep_name))
                    .ok_or_else(|| {
                        log::error!("Function '{translated_ep_name}' does not exist");
                        crate::PipelineError::EntryPoint(naga_stage)
                    })?;

                // collect sizes indices, immutable buffers, and work group memory sizes
                let ep_info = &module_info.get_entry_point(ep_index);
                let mut wg_memory_sizes = Vec::new();
                let mut sized_bindings = Vec::new();
                let mut immutable_buffer_mask = 0;
                for (var_handle, var) in module.global_variables.iter() {
                    match var.space {
                        naga::AddressSpace::WorkGroup => {
                            if !ep_info[var_handle].is_empty() {
                                let size = module.types[var.ty].inner.size(module.to_ctx());
                                wg_memory_sizes.push(size);
                            }
                        }
                        naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. } => {
                            let br = match var.binding {
                                Some(br) => br,
                                None => continue,
                            };
                            let storage_access_store = match var.space {
                                naga::AddressSpace::Storage { access } => {
                                    access.contains(naga::StorageAccess::STORE)
                                }
                                _ => false,
                            };

                            // check for an immutable buffer
                            if !ep_info[var_handle].is_empty() && !storage_access_store {
                                let slot = ep_resources.resources[&br].buffer.unwrap();
                                immutable_buffer_mask |= 1 << slot;
                            }

                            let mut dynamic_array_container_ty = var.ty;
                            if let naga::TypeInner::Struct { ref members, .. } =
                                module.types[var.ty].inner
                            {
                                dynamic_array_container_ty = members.last().unwrap().ty;
                            }
                            if let naga::TypeInner::Array {
                                size: naga::ArraySize::Dynamic,
                                ..
                            } = module.types[dynamic_array_container_ty].inner
                            {
                                sized_bindings.push(br);
                            }
                        }
                        _ => {}
                    }
                }

                Ok(CompiledShader {
                    library,
                    source_bytes: source.len() as u64,
                    function,
                    wg_size,
                    wg_memory_sizes,
                    sized_bindings,
                    immutable_buffer_mask,
                })
            }
            ShaderModuleSource::Passthrough(ref shader) => {
                let library = if let Some(source) = &shader.source {
                    let source = specialize_argument_arrays(source, &layout.argument_array_counts[naga_stage])
                        .map_err(|e| crate::PipelineError::Linkage(map_naga_stage(naga_stage), e))?;
                    self.compile_passthrough_msl(label, &source)
                        .map_err(|e| crate::PipelineError::Linkage(map_naga_stage(naga_stage), e))?
                } else {
                    shader.library.as_ref().unwrap().clone()
                };
                let function = library.newFunctionWithName(&NSString::from_str(stage.entry_point))
                    .ok_or(crate::PipelineError::EntryPoint(naga_stage))?;
                Ok(CompiledShader {
                    library,
                    source_bytes: shader.source_bytes,
                    function,
                    wg_size: MTLSize {
                        width: shader.num_workgroups.0 as usize,
                        height: shader.num_workgroups.1 as usize,
                        depth: shader.num_workgroups.2 as usize,
                    },
                    wg_memory_sizes: vec![],
                    sized_bindings: vec![],
                    immutable_buffer_mask: 0,
                })
            }
        }
    }

    fn set_buffers_mutability(
        buffers: &MTLPipelineBufferDescriptorArray,
        mut immutable_mask: usize,
    ) {
        while immutable_mask != 0 {
            let slot = immutable_mask.trailing_zeros();
            immutable_mask ^= 1 << slot;
            unsafe { buffers.objectAtIndexedSubscript(slot as usize) }
                .setMutability(MTLMutability::Immutable);
        }
    }

    pub unsafe fn texture_from_raw(
        raw: Retained<ProtocolObject<dyn MTLTexture>>,
        format: wgt::TextureFormat,
        raw_type: MTLTextureType,
        array_layers: u32,
        mip_levels: u32,
        copy_size: crate::CopyExtent,
    ) -> super::Texture {
        super::Texture {
            raw,
            format,
            raw_type,
            array_layers,
            mip_levels,
            copy_size,
        }
    }

    pub unsafe fn device_from_raw(
        raw: Retained<ProtocolObject<dyn MTLDevice>>,
        features: wgt::Features,
    ) -> super::Device {
        let capabilities_query = super::CapabilitiesQuery::new(&raw);
        let shared = super::AdapterShared::new(raw, &capabilities_query);
        super::Device {
            shared: Arc::new(shared),
            features,
            counters: Default::default(),
            archive_additions: Default::default(),
        }
    }

    pub unsafe fn buffer_from_raw(
        raw: Retained<ProtocolObject<dyn MTLBuffer>>,
        size: wgt::BufferAddress,
    ) -> super::Buffer {
        super::Buffer { raw, size }
    }

    pub fn raw_device(&self) -> &Retained<ProtocolObject<dyn MTLDevice>> {
        &self.shared.device
    }

    /// How many pipelines this device has added to a binary archive, or queued for it behind a
    /// serialization. An owner with one pipeline cache on the device reads the count and then
    /// serializes: that serialization holds every addition the count included. Seeing the same
    /// count again, it knows the archive holds nothing new: every pipeline since was an archive
    /// hit, and serializing again would only rewrite the same bytes.
    pub fn archive_additions(&self) -> usize {
        self.archive_additions.load(core::sync::atomic::Ordering::Acquire)
    }
}

impl crate::Device for super::Device {
    type A = super::Api;

    unsafe fn create_buffer(&self, desc: &crate::BufferDescriptor) -> DeviceResult<super::Buffer> {
        let map_read = desc.usage.contains(wgt::BufferUses::MAP_READ);
        let map_write = desc.usage.contains(wgt::BufferUses::MAP_WRITE);

        let mut options = MTLResourceOptions::empty();
        options |= if map_read || map_write {
            // `crate::MemoryFlags::PREFER_COHERENT` is ignored here
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        };
        options.set(MTLResourceOptions::CPUCacheModeWriteCombined, map_write);

        //TODO: HazardTrackingModeUntracked

        autoreleasepool(|_| {
            let raw = self
                .shared
                .device
                .newBufferWithLength_options(desc.size as usize, options)
                .ok_or(crate::DeviceError::OutOfMemory)?;
            if let Some(label) = desc.label {
                raw.setLabel(Some(&NSString::from_str(label)));
            }
            self.counters.buffers.add(1);
            Ok(super::Buffer {
                raw,
                size: desc.size,
            })
        })
    }
    unsafe fn destroy_buffer(&self, _buffer: super::Buffer) {
        self.counters.buffers.sub(1);
    }

    unsafe fn add_raw_buffer(&self, _buffer: &super::Buffer) {
        self.counters.buffers.add(1);
    }

    unsafe fn map_buffer(
        &self,
        buffer: &super::Buffer,
        range: crate::MemoryRange,
    ) -> DeviceResult<crate::BufferMapping> {
        let ptr = buffer.raw.contents().cast::<u8>();
        Ok(crate::BufferMapping {
            ptr: NonNull::new(unsafe { ptr.as_ptr().offset(range.start as isize) }).unwrap(),
            is_coherent: true,
        })
    }

    unsafe fn unmap_buffer(&self, _buffer: &super::Buffer) {}
    unsafe fn flush_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}
    unsafe fn invalidate_mapped_ranges<I>(&self, _buffer: &super::Buffer, _ranges: I) {}

    unsafe fn create_texture(
        &self,
        desc: &crate::TextureDescriptor,
    ) -> DeviceResult<super::Texture> {
        let mtl_format = self
            .shared
            .private_texture_format_caps
            .map_format(desc.format);

        autoreleasepool(|_| {
            let descriptor = MTLTextureDescriptor::new();

            let mtl_type = match desc.dimension {
                wgt::TextureDimension::D1 => MTLTextureType::Type1D,
                wgt::TextureDimension::D2 => {
                    if desc.sample_count > 1 {
                        unsafe { descriptor.setSampleCount(desc.sample_count as usize) };
                        MTLTextureType::Type2DMultisample
                    } else if desc.size.depth_or_array_layers > 1 {
                        unsafe {
                            descriptor.setArrayLength(desc.size.depth_or_array_layers as usize)
                        };
                        MTLTextureType::Type2DArray
                    } else {
                        MTLTextureType::Type2D
                    }
                }
                wgt::TextureDimension::D3 => {
                    unsafe { descriptor.setDepth(desc.size.depth_or_array_layers as usize) };
                    MTLTextureType::Type3D
                }
            };

            let mtl_storage_mode = if desc.usage.contains(wgt::TextureUses::TRANSIENT)
                && self.shared.private_caps.supports_memoryless_storage
            {
                MTLStorageMode::Memoryless
            } else {
                MTLStorageMode::Private
            };

            descriptor.setTextureType(mtl_type);
            unsafe { descriptor.setWidth(desc.size.width as usize) };
            unsafe { descriptor.setHeight(desc.size.height as usize) };
            unsafe { descriptor.setMipmapLevelCount(desc.mip_level_count as usize) };
            descriptor.setPixelFormat(mtl_format);
            descriptor.setUsage(conv::map_texture_usage(desc.format, desc.usage));
            descriptor.setStorageMode(mtl_storage_mode);

            let raw = self
                .shared
                .device
                .newTextureWithDescriptor(&descriptor)
                .ok_or(crate::DeviceError::OutOfMemory)?;
            if let Some(label) = desc.label {
                raw.setLabel(Some(&NSString::from_str(label)));
            }

            self.counters.textures.add(1);

            Ok(super::Texture {
                raw,
                format: desc.format,
                raw_type: mtl_type,
                mip_levels: desc.mip_level_count,
                array_layers: desc.array_layer_count(),
                copy_size: desc.copy_extent(),
            })
        })
    }

    unsafe fn destroy_texture(&self, _texture: super::Texture) {
        self.counters.textures.sub(1);
    }

    unsafe fn add_raw_texture(&self, _texture: &super::Texture) {
        self.counters.textures.add(1);
    }

    unsafe fn create_texture_view(
        &self,
        texture: &super::Texture,
        desc: &crate::TextureViewDescriptor,
    ) -> DeviceResult<super::TextureView> {
        let raw_type = if texture.raw_type == MTLTextureType::Type2DMultisample {
            texture.raw_type
        } else {
            conv::map_texture_view_dimension(desc.dimension)
        };

        let aspects = crate::FormatAspects::new(texture.format, desc.range.aspect);

        let raw_format = self
            .shared
            .private_texture_format_caps
            .map_view_format(desc.format, aspects);

        let format_equal = raw_format
            == self
                .shared
                .private_texture_format_caps
                .map_format(texture.format);
        let type_equal = raw_type == texture.raw_type;
        let range_full_resource =
            desc.range
                .is_full_resource(desc.format, texture.mip_levels, texture.array_layers);

        let raw = if format_equal && type_equal && range_full_resource {
            // Some images are marked as framebuffer-only, and we can't create aliases of them.
            // Also helps working around Metal bugs with aliased array textures.
            texture.raw.to_owned()
        } else {
            let mip_level_count = desc
                .range
                .mip_level_count
                .unwrap_or(texture.mip_levels - desc.range.base_mip_level);
            let array_layer_count = desc
                .range
                .array_layer_count
                .unwrap_or(texture.array_layers - desc.range.base_array_layer);

            autoreleasepool(|_| {
                let level_range = NSRange {
                    location: desc.range.base_mip_level as _,
                    length: mip_level_count as _,
                };
                let slice_range = NSRange {
                    location: desc.range.base_array_layer as _,
                    length: array_layer_count as _,
                };
                let raw = unsafe {
                    texture
                        .raw
                        .newTextureViewWithPixelFormat_textureType_levels_slices(
                            raw_format,
                            raw_type,
                            level_range,
                            slice_range,
                        )
                        .unwrap()
                };
                if let Some(label) = desc.label {
                    raw.setLabel(Some(&NSString::from_str(label)));
                }
                raw
            })
        };

        self.counters.texture_views.add(1);

        Ok(super::TextureView { raw, aspects })
    }

    unsafe fn destroy_texture_view(&self, _view: super::TextureView) {
        self.counters.texture_views.sub(1);
    }

    unsafe fn create_sampler(
        &self,
        desc: &crate::SamplerDescriptor,
    ) -> DeviceResult<super::Sampler> {
        autoreleasepool(|_| {
            let descriptor = MTLSamplerDescriptor::new();

            descriptor.setMinFilter(conv::map_filter_mode(desc.min_filter));
            descriptor.setMagFilter(conv::map_filter_mode(desc.mag_filter));
            descriptor.setMipFilter(match desc.mipmap_filter {
                wgt::MipmapFilterMode::Nearest if desc.lod_clamp == (0.0..0.0) => {
                    MTLSamplerMipFilter::NotMipmapped
                }
                wgt::MipmapFilterMode::Nearest => MTLSamplerMipFilter::Nearest,
                wgt::MipmapFilterMode::Linear => MTLSamplerMipFilter::Linear,
            });

            let [s, t, r] = desc.address_modes;
            descriptor.setSAddressMode(conv::map_address_mode(s));
            descriptor.setTAddressMode(conv::map_address_mode(t));
            descriptor.setRAddressMode(conv::map_address_mode(r));

            // Anisotropy is always supported on mac up to 16x
            descriptor.setMaxAnisotropy(desc.anisotropy_clamp as _);

            descriptor.setLodMinClamp(desc.lod_clamp.start);
            descriptor.setLodMaxClamp(desc.lod_clamp.end);

            if let Some(fun) = desc.compare {
                descriptor.setCompareFunction(conv::map_compare_function(fun));
            }

            if let Some(border_color) = desc.border_color {
                if let wgt::SamplerBorderColor::Zero = border_color {
                    if s == wgt::AddressMode::ClampToBorder {
                        descriptor.setSAddressMode(MTLSamplerAddressMode::ClampToZero);
                    }

                    if t == wgt::AddressMode::ClampToBorder {
                        descriptor.setTAddressMode(MTLSamplerAddressMode::ClampToZero);
                    }

                    if r == wgt::AddressMode::ClampToBorder {
                        descriptor.setRAddressMode(MTLSamplerAddressMode::ClampToZero);
                    }
                } else {
                    descriptor.setBorderColor(conv::map_border_color(border_color));
                }
            }

            if let Some(label) = desc.label {
                descriptor.setLabel(Some(&NSString::from_str(label)));
            }
            if self.features.contains(wgt::Features::TEXTURE_BINDING_ARRAY) {
                descriptor.setSupportArgumentBuffers(true);
            }
            let raw = self
                .shared
                .device
                .newSamplerStateWithDescriptor(&descriptor)
                .unwrap();

            self.counters.samplers.add(1);

            Ok(super::Sampler { raw })
        })
    }
    unsafe fn destroy_sampler(&self, _sampler: super::Sampler) {
        self.counters.samplers.sub(1);
    }

    unsafe fn create_command_encoder(
        &self,
        desc: &crate::CommandEncoderDescriptor<super::Queue>,
    ) -> Result<super::CommandEncoder, crate::DeviceError> {
        self.counters.command_encoders.add(1);
        Ok(super::CommandEncoder {
            shared: Arc::clone(&self.shared),
            queue_shared: Arc::clone(&desc.queue.shared),
            raw_cmd_buf: None,
            timestamp_event: None,
            timestamp_value: 0,
            timestamp_barrier_needed: true,
            state: super::CommandState::default(),
            temp: super::Temp::default(),
            counters: Arc::clone(&self.counters),
        })
    }

    unsafe fn create_bind_group_layout(
        &self,
        desc: &crate::BindGroupLayoutDescriptor,
    ) -> DeviceResult<super::BindGroupLayout> {
        self.counters.bind_group_layouts.add(1);

        Ok(super::BindGroupLayout {
            entries: Arc::from(desc.entries),
        })
    }

    unsafe fn destroy_bind_group_layout(&self, _bg_layout: super::BindGroupLayout) {
        self.counters.bind_group_layouts.sub(1);
    }

    unsafe fn create_pipeline_layout(
        &self,
        desc: &crate::PipelineLayoutDescriptor<super::BindGroupLayout>,
    ) -> DeviceResult<super::PipelineLayout> {
        #[derive(Debug)]
        struct StageInfo {
            stage: naga::ShaderStage,
            counters: super::ResourceData<super::ResourceIndex>,
            pc_buffer: Option<super::ResourceIndex>,
            pc_limit: u32,
            sizes_buffer: Option<super::ResourceIndex>,
            need_sizes_buffer: bool,
            resources: naga::back::msl::BindingMap,
            argument_array_counts: Vec<(u32, u32)>,
        }

        let mut stage_data = super::NAGA_STAGES.map(|stage| StageInfo {
            stage,
            counters: super::ResourceData::default(),
            pc_buffer: None,
            pc_limit: 0,
            sizes_buffer: None,
            need_sizes_buffer: false,
            resources: Default::default(),
            argument_array_counts: Vec::new(),
        });
        let mut bind_group_infos = [const { None }; crate::MAX_BIND_GROUPS];

        // First, place the immediates
        for info in stage_data.iter_mut() {
            info.pc_limit = desc.immediate_size;

            // handle the immediate data buffer assignment and shader overrides
            if info.pc_limit != 0 {
                info.pc_buffer = Some(info.counters.buffers);
                info.counters.buffers += 1;
            }
        }

        // Second, place the described resources
        for (group_index, bgl) in desc.bind_group_layouts.iter().enumerate() {
            let Some(bgl) = bgl else {
                continue;
            };

            // remember where the resources for this set start at each shader stage
            let base_resource_indices = stage_data.map_ref(|info| info.counters.clone());

            for entry in bgl.entries.iter() {
                if let wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { .. },
                    ..
                } = entry.ty
                {
                    for info in stage_data.iter_mut() {
                        if entry.visibility.contains(map_naga_stage(info.stage)) {
                            info.need_sizes_buffer = true;
                        }
                    }
                }

                for info in stage_data.iter_mut() {
                    if !entry.visibility.contains(map_naga_stage(info.stage)) {
                        continue;
                    }

                    let mut target = naga::back::msl::BindTarget::default();
                    // Bindless path
                    if let Some(count) = entry.count {
                        info.argument_array_counts.push((info.counters.buffers, count.get()));
                        target.buffer = Some(info.counters.buffers as _);
                    } else {
                        match entry.ty {
                            wgt::BindingType::Buffer { ty, .. } => {
                                target.buffer = Some(info.counters.buffers as _);
                                if let wgt::BufferBindingType::Storage { read_only } = ty {
                                    target.mutable = !read_only;
                                }
                            }
                            wgt::BindingType::Sampler { .. } => {
                                target.sampler =
                                    Some(naga::back::msl::BindSamplerTarget::Resource(
                                        info.counters.samplers as _,
                                    ));
                            }
                            wgt::BindingType::Texture { .. } => {
                                target.texture = Some(info.counters.textures as _);
                            }
                            wgt::BindingType::StorageTexture { access, .. } => {
                                target.texture = Some(info.counters.textures as _);
                                target.mutable = match access {
                                    wgt::StorageTextureAccess::ReadOnly => false,
                                    wgt::StorageTextureAccess::WriteOnly => true,
                                    wgt::StorageTextureAccess::ReadWrite => true,
                                    wgt::StorageTextureAccess::Atomic => true,
                                };
                            }
                            wgt::BindingType::AccelerationStructure { .. } => {
                                target.buffer = Some(info.counters.buffers as _);
                            }
                            wgt::BindingType::ExternalTexture => {
                                target.external_texture =
                                    Some(naga::back::msl::BindExternalTextureTarget {
                                        planes: [
                                            info.counters.textures as _,
                                            (info.counters.textures + 1) as _,
                                            (info.counters.textures + 2) as _,
                                        ],
                                        params: info.counters.buffers as _,
                                    });
                            }
                        }
                    }

                    let [buffers, textures, samplers] = entry.ty.metal_resource_slots(entry.count);
                    info.counters.buffers += u32::from(buffers);
                    info.counters.textures += u32::from(textures);
                    info.counters.samplers += u32::from(samplers);

                    let br = naga::ResourceBinding {
                        group: group_index as u32,
                        binding: entry.binding,
                    };
                    info.resources.insert(br, target);
                }
            }

            bind_group_infos[group_index] = Some(super::BindGroupLayoutInfo {
                base_resource_indices,
            });
        }

        // Finally, make sure we fit the limits
        for info in stage_data.iter_mut() {
            if info.need_sizes_buffer || info.stage == naga::ShaderStage::Vertex {
                // Set aside space for the sizes_buffer, which is required
                // for variable-length buffers, or to support vertex pulling.
                info.sizes_buffer = Some(info.counters.buffers);
                info.counters.buffers += 1;
            }

            if info.counters.buffers > self.shared.private_caps.max_buffers_per_stage
                || info.counters.textures > self.shared.private_caps.max_textures_per_stage
                || info.counters.samplers > self.shared.private_caps.max_samplers_per_stage
            {
                let caps = &self.shared.private_caps;
                return Err(crate::DeviceError::ResourceLimit(format!(
                    "pipeline layout {:?} binds {} buffers, {} textures and {} samplers to the {:?} stage; \
                     Metal allows {} buffers, {} textures and {} samplers per stage",
                    desc.label, info.counters.buffers, info.counters.textures, info.counters.samplers,
                    info.stage, caps.max_buffers_per_stage, caps.max_textures_per_stage, caps.max_samplers_per_stage,
                )));
            }
        }

        let immediates_infos = stage_data.map_ref(|info| {
            info.pc_buffer.map(|buffer_index| super::ImmediateDataInfo {
                count: info.pc_limit,
                buffer_index,
            })
        });

        let total_counters = stage_data.map_ref(|info| info.counters.clone());

        let argument_array_counts = stage_data.map_ref(|info| info.argument_array_counts.clone());
        let per_stage_map = stage_data.map(|info| naga::back::msl::EntryPointResources {
            immediates_buffer: info
                .pc_buffer
                .map(|buffer_index| buffer_index as naga::back::msl::Slot),
            sizes_buffer: info
                .sizes_buffer
                .map(|buffer_index| buffer_index as naga::back::msl::Slot),
            resources: info.resources,
        });

        self.counters.pipeline_layouts.add(1);

        Ok(super::PipelineLayout {
            bind_group_infos,
            immediates_infos,
            total_counters,
            total_immediates: desc.immediate_size,
            per_stage_map,
            argument_array_counts,
        })
    }

    unsafe fn destroy_pipeline_layout(&self, _pipeline_layout: super::PipelineLayout) {
        self.counters.pipeline_layouts.sub(1);
    }

    unsafe fn create_bind_group(
        &self,
        desc: &crate::BindGroupDescriptor<
            super::BindGroupLayout,
            super::Buffer,
            super::Sampler,
            super::TextureView,
            super::AccelerationStructure,
        >,
    ) -> DeviceResult<super::BindGroup> {
        autoreleasepool(|_| {
            let mut bg = super::BindGroup::default();
            for (&stage, counter) in super::NAGA_STAGES.iter().zip(bg.counters.iter_mut()) {
                let stage_bit = map_naga_stage(stage);
                let mut dynamic_offsets_count = 0u32;
                let layout_and_entry_iter = desc.entries.iter().map(|entry| {
                    let layout = desc
                        .layout
                        .entries
                        .iter()
                        .find(|layout_entry| layout_entry.binding == entry.binding)
                        .expect("internal error: no layout entry found with binding slot");
                    (entry, layout)
                });
                for (entry, layout) in layout_and_entry_iter {
                    // Bindless path
                    if let Some(declared_count) = layout.count {
                        if !layout.visibility.contains(stage_bit) {
                            continue;
                        }

                        let count = entry.count;

                        let stages = conv::map_render_stages(layout.visibility);
                        let uses = conv::map_resource_usage(&layout.ty);

                        let buffer = new_argument_buffer(&self.shared.device, declared_count);

                        let contents: &mut [MTLResourceID] = unsafe {
                            core::slice::from_raw_parts_mut(
                                buffer.contents().cast().as_ptr(),
                                declared_count.get() as usize,
                            )
                        };

                        match layout.ty {
                            wgt::BindingType::Buffer { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + count as usize;
                                for (idx, source) in desc.buffers[start..end].iter().enumerate() {
                                    // Tier-2 argument buffers store device pointers directly.
                                    // Slang's ParameterBlock of StructuredBuffers uses this ABI.
                                    unsafe {
                                        contents
                                            .as_mut_ptr()
                                            .cast::<u64>()
                                            .add(idx)
                                            .write(source.buffer.raw.gpuAddress() + source.offset);
                                    }
                                    let use_info = bg
                                        .resources_to_use
                                        .entry(source.buffer.as_raw().cast())
                                        .or_default();
                                    use_info.stages |= stages;
                                    use_info.uses |= uses;
                                    use_info.visible_in_compute |=
                                        layout.visibility.contains(wgt::ShaderStages::COMPUTE);
                                }
                            }
                            wgt::BindingType::Texture { .. }
                            | wgt::BindingType::StorageTexture { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + count as usize;
                                let textures = &desc.textures[start..end];

                                for (idx, tex) in textures.iter().enumerate() {
                                    contents[idx] = tex.view.raw.gpuResourceID();

                                    let use_info = bg
                                        .resources_to_use
                                        .entry(tex.view.as_raw().cast())
                                        .or_default();
                                    use_info.stages |= stages;
                                    use_info.uses |= uses;
                                    use_info.visible_in_compute |=
                                        layout.visibility.contains(wgt::ShaderStages::COMPUTE);
                                }
                            }
                            wgt::BindingType::Sampler { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + count as usize;
                                let samplers = &desc.samplers[start..end];

                                for (idx, &sampler) in samplers.iter().enumerate() {
                                    contents[idx] = sampler.raw.gpuResourceID();
                                    // Samplers aren't resources like buffers and textures, so don't
                                    // need to be passed to useResource
                                }
                            }
                            wgt::BindingType::AccelerationStructure { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + count as usize;
                                let acceleration_structures =
                                    &desc.acceleration_structures[start..end];

                                for (idx, &acceleration_structure) in
                                    acceleration_structures.iter().enumerate()
                                {
                                    contents[idx] = acceleration_structure.raw.gpuResourceID();

                                    let use_info = bg
                                        .resources_to_use
                                        .entry(acceleration_structure.as_raw().cast())
                                        .or_default();
                                    use_info.stages |= stages;
                                    use_info.uses |= uses;
                                    use_info.visible_in_compute |=
                                        layout.visibility.contains(wgt::ShaderStages::COMPUTE);
                                }
                            }
                            _ => {
                                unimplemented!();
                            }
                        }

                        fill_argument_buffer_tail(contents, count as usize);

                        bg.buffers.push(super::BufferLikeResource::Buffer {
                            ptr: NonNull::from(&*buffer),
                            offset: 0,
                            dynamic_index: None,
                            binding_size: None,
                            binding_location: layout.binding,
                        });
                        counter.buffers += 1;

                        bg.argument_buffers.push(buffer)
                    }
                    // Bindfull path
                    else {
                        if let wgt::BindingType::Buffer {
                            has_dynamic_offset: true,
                            ..
                        } = layout.ty
                        {
                            dynamic_offsets_count += 1;
                        }
                        if !layout.visibility.contains(stage_bit) {
                            continue;
                        }
                        match layout.ty {
                            wgt::BindingType::Buffer {
                                ty,
                                has_dynamic_offset,
                                ..
                            } => {
                                let start = entry.resource_index as usize;
                                let end = start + 1;
                                bg.buffers
                                    .extend(desc.buffers[start..end].iter().map(|source| {
                                        // Given the restrictions on `BufferBinding::offset`,
                                        // this should never be `None`.
                                        let remaining_size = wgt::BufferSize::new(
                                            source.buffer.size - source.offset,
                                        );
                                        let binding_size = match ty {
                                            wgt::BufferBindingType::Storage { .. } => {
                                                source.size.or(remaining_size)
                                            }
                                            _ => None,
                                        };
                                        super::BufferLikeResource::Buffer {
                                            ptr: source.buffer.as_raw(),
                                            offset: source.offset,
                                            dynamic_index: if has_dynamic_offset {
                                                Some(dynamic_offsets_count - 1)
                                            } else {
                                                None
                                            },
                                            binding_size,
                                            binding_location: layout.binding,
                                        }
                                    }));
                                counter.buffers += 1;
                            }
                            wgt::BindingType::Sampler { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + 1;
                                bg.samplers.extend(
                                    desc.samplers[start..end].iter().map(|samp| samp.as_raw()),
                                );
                                counter.samplers += 1;
                            }
                            wgt::BindingType::Texture { .. }
                            | wgt::BindingType::StorageTexture { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + 1;
                                bg.textures.extend(
                                    desc.textures[start..end]
                                        .iter()
                                        .map(|tex| tex.view.as_raw()),
                                );
                                counter.textures += 1;
                            }
                            wgt::BindingType::AccelerationStructure { .. } => {
                                let start = entry.resource_index as usize;
                                let end = start + 1;
                                bg.buffers.extend(
                                    desc.acceleration_structures[start..end].iter().map(
                                        |acceleration_structure| {
                                            super::BufferLikeResource::AccelerationStructure(
                                                acceleration_structure.as_raw(),
                                            )
                                        },
                                    ),
                                );
                                counter.buffers += 1;
                            }
                            wgt::BindingType::ExternalTexture => {
                                // We don't yet support binding arrays of external textures.
                                // https://github.com/gfx-rs/wgpu/issues/8027
                                assert_eq!(entry.count, 1);
                                let external_texture =
                                    &desc.external_textures[entry.resource_index as usize];
                                bg.textures.extend(
                                    external_texture
                                        .planes
                                        .iter()
                                        .map(|plane| plane.view.as_raw()),
                                );
                                bg.buffers.push(super::BufferLikeResource::Buffer {
                                    ptr: external_texture.params.buffer.as_raw(),
                                    offset: external_texture.params.offset,
                                    dynamic_index: None,
                                    binding_size: None,
                                    binding_location: layout.binding,
                                });
                                counter.textures += 3;
                                counter.buffers += 1;
                            }
                        }
                    }
                }
            }

            self.counters.bind_groups.add(1);

            Ok(bg)
        })
    }

    unsafe fn destroy_bind_group(&self, _group: super::BindGroup) {
        self.counters.bind_groups.sub(1);
    }

    unsafe fn create_shader_module(
        &self,
        desc: &crate::ShaderModuleDescriptor,
        shader: crate::ShaderInput,
    ) -> Result<super::ShaderModule, crate::ShaderError> {
        self.counters.shader_modules.add(1);

        match shader {
            crate::ShaderInput::Naga(naga) => Ok(super::ShaderModule {
                source: ShaderModuleSource::Naga(naga),
                bounds_checks: desc.runtime_checks,
            }),
            crate::ShaderInput::MetalLib {
                file,
                num_workgroups,
            } => {
                // SAFETY: this creates a reference to `file` that is dropped before `file` is dropped.
                let library = super::library_from_metallib::new_library_from_metallib_bytes(
                    &self.shared.device,
                    file,
                )
                .map_err(|e| crate::ShaderError::Compilation(format!("Metallib: {e:?}")))?;
                Ok(super::ShaderModule {
                    source: ShaderModuleSource::Passthrough(PassthroughShader {
                        library: Some(library),
                        source: None,
                        source_bytes: file.len() as u64,
                        num_workgroups,
                    }),
                    // This goes unused for passthrough shaders
                    bounds_checks: wgt::ShaderRuntimeChecks::unchecked(),
                })
            }
            crate::ShaderInput::Msl {
                shader: source,
                num_workgroups,
            } => {
                let flexible = source.contains("[];");
                let library = if flexible { None } else {
                    Some(self.compile_passthrough_msl(desc.label, source).map_err(crate::ShaderError::Compilation)?)
                };

                Ok(super::ShaderModule {
                    source: ShaderModuleSource::Passthrough(PassthroughShader {
                        library,
                        source: flexible.then(|| source.to_owned()),
                        source_bytes: source.len() as u64,
                        num_workgroups,
                    }),
                    // This goes unused for passthrough shaders
                    bounds_checks: wgt::ShaderRuntimeChecks::unchecked(),
                })
            }
            crate::ShaderInput::SpirV(_)
            | crate::ShaderInput::Dxil { .. }
            | crate::ShaderInput::Hlsl { .. }
            | crate::ShaderInput::Glsl { .. } => unreachable!(),
        }
    }

    unsafe fn destroy_shader_module(&self, _module: super::ShaderModule) {
        self.counters.shader_modules.sub(1);
    }

    unsafe fn create_render_pipeline(
        &self,
        desc: &crate::RenderPipelineDescriptor<
            super::PipelineLayout,
            super::ShaderModule,
            super::PipelineCache,
        >,
    ) -> Result<super::RenderPipeline, crate::PipelineError> {
        autoreleasepool(|_| {
            // Source behind every stage: what compiling the pipeline state is charged.
            let mut source_bytes = 0;
            enum MetalGenericRenderPipelineDescriptor {
                Standard(Retained<MTLRenderPipelineDescriptor>),
                Mesh(Retained<MTLMeshRenderPipelineDescriptor>),
            }
            macro_rules! descriptor_fn {
                ($descriptor:ident . $method:ident $( ( $($args:expr),* ) )? ) => {
                    match $descriptor {
                        MetalGenericRenderPipelineDescriptor::Standard(ref inner) => inner.$method$(($($args),*))?,
                        MetalGenericRenderPipelineDescriptor::Mesh(ref inner) => inner.$method$(($($args),*))?,
                    }
                };
            }
            #[allow(non_snake_case)]
            impl MetalGenericRenderPipelineDescriptor {
                unsafe fn setFragmentFunction(
                    &self,
                    function: Option<&ProtocolObject<dyn MTLFunction>>,
                ) {
                    unsafe { descriptor_fn!(self.setFragmentFunction(function)) };
                }
                fn fragmentBuffers(&self) -> Retained<MTLPipelineBufferDescriptorArray> {
                    descriptor_fn!(self.fragmentBuffers())
                }
                fn setDepthAttachmentPixelFormat(&self, pixel_format: MTLPixelFormat) {
                    descriptor_fn!(self.setDepthAttachmentPixelFormat(pixel_format));
                }
                fn colorAttachments(
                    &self,
                ) -> Retained<MTLRenderPipelineColorAttachmentDescriptorArray> {
                    descriptor_fn!(self.colorAttachments())
                }
                fn setStencilAttachmentPixelFormat(&self, pixel_format: MTLPixelFormat) {
                    descriptor_fn!(self.setStencilAttachmentPixelFormat(pixel_format));
                }
                fn setAlphaToCoverageEnabled(&self, enabled: bool) {
                    descriptor_fn!(self.setAlphaToCoverageEnabled(enabled));
                }
                fn setLabel(&self, label: Option<&NSString>) {
                    descriptor_fn!(self.setLabel(label));
                }
                unsafe fn setMaxVertexAmplificationCount(&self, count: NSUInteger) {
                    unsafe { descriptor_fn!(self.setMaxVertexAmplificationCount(count)) }
                }
            }

            // https://developer.apple.com/documentation/metal/mtlpipelinebufferdescriptor/mutability
            let supports_mutability =
                available!(macos = 10.13, ios = 11.0, tvos = 11.0, visionos = 1.0);

            let (primitive_class, raw_primitive_type) =
                conv::map_primitive_topology(desc.primitive.topology);

            let vs_info;
            let ts_info;
            let ms_info;

            // Create the pipeline descriptor and do vertex/mesh pipeline specific setup
            let descriptor = match desc.vertex_processor {
                crate::VertexProcessor::Standard {
                    vertex_buffers,
                    ref vertex_stage,
                } => {
                    // Vertex pipeline specific setup

                    let descriptor = MTLRenderPipelineDescriptor::new();
                    ts_info = None;
                    ms_info = None;

                    // Collect vertex buffer mappings
                    let mut vertex_buffer_mappings =
                        Vec::<naga::back::msl::VertexBufferMapping>::new();
                    for (i, vbl) in vertex_buffers.iter().enumerate() {
                        let mut attributes = Vec::<naga::back::msl::AttributeMapping>::new();
                        for attribute in vbl.attributes.iter() {
                            attributes.push(naga::back::msl::AttributeMapping {
                                shader_location: attribute.shader_location,
                                offset: attribute.offset as u32,
                                format: convert_vertex_format_to_naga(attribute.format),
                            });
                        }

                        let mapping = naga::back::msl::VertexBufferMapping {
                            id: self.shared.private_caps.max_vertex_buffers - 1 - i as u32,
                            stride: if vbl.array_stride > 0 {
                                vbl.array_stride.try_into().unwrap()
                            } else {
                                vbl.attributes
                                    .iter()
                                    .map(|attribute| attribute.offset + attribute.format.size())
                                    .max()
                                    .unwrap_or(0)
                                    .try_into()
                                    .unwrap()
                            },
                            step_mode: match (vbl.array_stride == 0, vbl.step_mode) {
                                (true, _) => naga::back::msl::VertexBufferStepMode::Constant,
                                (false, wgt::VertexStepMode::Vertex) => {
                                    naga::back::msl::VertexBufferStepMode::ByVertex
                                }
                                (false, wgt::VertexStepMode::Instance) => {
                                    naga::back::msl::VertexBufferStepMode::ByInstance
                                }
                            },
                            attributes,
                        };
                        vertex_buffer_mappings.push(mapping);
                    }

                    // Setup vertex shader
                    {
                        let vs = self.load_shader(
                            vertex_stage,
                            &vertex_buffer_mappings,
                            desc.layout,
                            primitive_class,
                            naga::ShaderStage::Vertex,
                            desc.label,
                        )?;
                        source_bytes += vs.source_bytes;

                        descriptor.setVertexFunction(Some(&vs.function));

                        if supports_mutability {
                            Self::set_buffers_mutability(
                                &descriptor.vertexBuffers(),
                                vs.immutable_buffer_mask,
                            );
                        }

                        vs_info = Some(super::PipelineStageInfo {
                            immediates: desc.layout.immediates_infos.vs,
                            sizes_slot: desc.layout.per_stage_map.vs.sizes_buffer,
                            sized_bindings: vs.sized_bindings,
                            vertex_buffer_mappings,
                            library: Some(vs.library),
                            raw_wg_size: MTLSize {
                                width: 0,
                                height: 0,
                                depth: 0,
                            },
                            work_group_memory_sizes: vec![],
                        });
                    }

                    // Validate vertex buffer count
                    if desc.layout.total_counters.vs.buffers + (vertex_buffers.len() as u32)
                        > self.shared.private_caps.max_vertex_buffers
                    {
                        let msg = format!(
                            "pipeline needs too many buffers in the vertex stage: {} vertex and {} layout",
                            vertex_buffers.len(),
                            desc.layout.total_counters.vs.buffers
                        );
                        return Err(crate::PipelineError::Linkage(
                            wgt::ShaderStages::VERTEX,
                            msg,
                        ));
                    }

                    // Set the pipeline vertex buffer info
                    if !vertex_buffers.is_empty() {
                        let vertex_descriptor = MTLVertexDescriptor::new();
                        for (i, vb) in vertex_buffers.iter().enumerate() {
                            let buffer_index =
                                self.shared.private_caps.max_vertex_buffers as usize - 1 - i;
                            let buffer_desc = unsafe {
                                vertex_descriptor
                                    .layouts()
                                    .objectAtIndexedSubscript(buffer_index)
                            };

                            // Metal expects the stride to be the actual size of the attributes.
                            // The semantics of array_stride == 0 can be achieved by setting
                            // the step function to constant and rate to 0.
                            if vb.array_stride == 0 {
                                let stride = vb
                                    .attributes
                                    .iter()
                                    .map(|attribute| attribute.offset + attribute.format.size())
                                    .max()
                                    .unwrap_or(0);
                                unsafe {
                                    buffer_desc.setStride(wgt::math::align_to(stride as _, 4))
                                };
                                buffer_desc.setStepFunction(MTLVertexStepFunction::Constant);
                                unsafe { buffer_desc.setStepRate(0) };
                            } else {
                                unsafe { buffer_desc.setStride(vb.array_stride as _) };
                                buffer_desc.setStepFunction(conv::map_step_mode(vb.step_mode));
                            }

                            for at in vb.attributes {
                                let attribute_desc = unsafe {
                                    vertex_descriptor
                                        .attributes()
                                        .objectAtIndexedSubscript(at.shader_location as _)
                                };
                                attribute_desc.setFormat(conv::map_vertex_format(at.format));
                                unsafe { attribute_desc.setBufferIndex(buffer_index) };
                                unsafe { attribute_desc.setOffset(at.offset as _) };
                            }
                        }
                        descriptor.setVertexDescriptor(Some(&vertex_descriptor));
                    }

                    MetalGenericRenderPipelineDescriptor::Standard(descriptor)
                }
                crate::VertexProcessor::Mesh {
                    ref task_stage,
                    ref mesh_stage,
                } => {
                    // Mesh pipeline specific setup

                    vs_info = None;
                    let descriptor = MTLMeshRenderPipelineDescriptor::new();

                    // Setup task stage
                    if let Some(ref task_stage) = task_stage {
                        let ts = self.load_shader(
                            task_stage,
                            &[],
                            desc.layout,
                            primitive_class,
                            naga::ShaderStage::Task,
                            desc.label,
                        )?;
                        source_bytes += ts.source_bytes;
                        unsafe { descriptor.setObjectFunction(Some(&ts.function)) };
                        if supports_mutability {
                            Self::set_buffers_mutability(
                                &descriptor.meshBuffers(),
                                ts.immutable_buffer_mask,
                            );
                        }
                        ts_info = Some(super::PipelineStageInfo {
                            immediates: desc.layout.immediates_infos.ts,
                            sizes_slot: desc.layout.per_stage_map.ts.sizes_buffer,
                            sized_bindings: ts.sized_bindings,
                            vertex_buffer_mappings: vec![],
                            library: Some(ts.library),
                            raw_wg_size: ts.wg_size,
                            work_group_memory_sizes: ts.wg_memory_sizes,
                        });
                    } else {
                        ts_info = None;
                    }

                    // Setup mesh stage
                    {
                        let ms = self.load_shader(
                            mesh_stage,
                            &[],
                            desc.layout,
                            primitive_class,
                            naga::ShaderStage::Mesh,
                            desc.label,
                        )?;
                        source_bytes += ms.source_bytes;
                        unsafe { descriptor.setMeshFunction(Some(&ms.function)) };
                        if supports_mutability {
                            Self::set_buffers_mutability(
                                &descriptor.meshBuffers(),
                                ms.immutable_buffer_mask,
                            );
                        }
                        ms_info = Some(super::PipelineStageInfo {
                            immediates: desc.layout.immediates_infos.ms,
                            sizes_slot: desc.layout.per_stage_map.ms.sizes_buffer,
                            sized_bindings: ms.sized_bindings,
                            vertex_buffer_mappings: vec![],
                            library: Some(ms.library),
                            raw_wg_size: ms.wg_size,
                            work_group_memory_sizes: ms.wg_memory_sizes,
                        });
                    }

                    MetalGenericRenderPipelineDescriptor::Mesh(descriptor)
                }
            };

            let raw_triangle_fill_mode = match desc.primitive.polygon_mode {
                wgt::PolygonMode::Fill => MTLTriangleFillMode::Fill,
                wgt::PolygonMode::Line => MTLTriangleFillMode::Lines,
                wgt::PolygonMode::Point => panic!(
                    "{:?} is not enabled for this backend",
                    wgt::Features::POLYGON_MODE_POINT
                ),
            };

            // Fragment shader
            let fs_info = match desc.fragment_stage {
                Some(ref stage) => {
                    let fs = self.load_shader(
                        stage,
                        &[],
                        desc.layout,
                        primitive_class,
                        naga::ShaderStage::Fragment,
                        desc.label,
                    )?;
                    source_bytes += fs.source_bytes;

                    unsafe { descriptor.setFragmentFunction(Some(&fs.function)) };
                    if supports_mutability {
                        Self::set_buffers_mutability(
                            &descriptor.fragmentBuffers(),
                            fs.immutable_buffer_mask,
                        );
                    }

                    Some(super::PipelineStageInfo {
                        immediates: desc.layout.immediates_infos.fs,
                        sizes_slot: desc.layout.per_stage_map.fs.sizes_buffer,
                        sized_bindings: fs.sized_bindings,
                        vertex_buffer_mappings: vec![],
                        library: Some(fs.library),
                        raw_wg_size: MTLSize {
                            width: 0,
                            height: 0,
                            depth: 0,
                        },
                        work_group_memory_sizes: vec![],
                    })
                }
                None => {
                    // TODO: This is a workaround for what appears to be a Metal validation bug
                    // A pixel format is required even though no attachments are provided
                    if desc.color_targets.is_empty() && desc.depth_stencil.is_none() {
                        descriptor.setDepthAttachmentPixelFormat(MTLPixelFormat::Depth32Float);
                    }
                    None
                }
            };

            // Setup pipeline color attachments
            for (i, ct) in desc.color_targets.iter().enumerate() {
                let at_descriptor =
                    unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(i) };
                let ct = if let Some(color_target) = ct.as_ref() {
                    color_target
                } else {
                    at_descriptor.setPixelFormat(MTLPixelFormat::Invalid);
                    continue;
                };

                let raw_format = self
                    .shared
                    .private_texture_format_caps
                    .map_format(ct.format);
                at_descriptor.setPixelFormat(raw_format);
                at_descriptor.setWriteMask(conv::map_color_write(ct.write_mask));

                if let Some(ref blend) = ct.blend {
                    at_descriptor.setBlendingEnabled(true);
                    let (color_op, color_src, color_dst) = conv::map_blend_component(&blend.color);
                    let (alpha_op, alpha_src, alpha_dst) = conv::map_blend_component(&blend.alpha);

                    at_descriptor.setRgbBlendOperation(color_op);
                    at_descriptor.setSourceRGBBlendFactor(color_src);
                    at_descriptor.setDestinationRGBBlendFactor(color_dst);

                    at_descriptor.setAlphaBlendOperation(alpha_op);
                    at_descriptor.setSourceAlphaBlendFactor(alpha_src);
                    at_descriptor.setDestinationAlphaBlendFactor(alpha_dst);
                }
            }

            // Setup depth stencil state
            let depth_stencil = match desc.depth_stencil {
                Some(ref ds) => {
                    let raw_format = self
                        .shared
                        .private_texture_format_caps
                        .map_format(ds.format);
                    let aspects = crate::FormatAspects::from(ds.format);
                    if aspects.contains(crate::FormatAspects::DEPTH) {
                        descriptor.setDepthAttachmentPixelFormat(raw_format);
                    }
                    if aspects.contains(crate::FormatAspects::STENCIL) {
                        descriptor.setStencilAttachmentPixelFormat(raw_format);
                    }

                    let ds_descriptor = create_depth_stencil_desc(ds);
                    let raw = self
                        .shared
                        .device
                        .newDepthStencilStateWithDescriptor(&ds_descriptor)
                        .unwrap();
                    Some((raw, ds.bias))
                }
                None => None,
            };

            // Setup multisample state
            if desc.multisample.count != 1 {
                //TODO: handle sample mask
                match descriptor {
                    MetalGenericRenderPipelineDescriptor::Standard(ref inner) => {
                        #[allow(deprecated)]
                        inner.setSampleCount(desc.multisample.count as _);
                    }
                    MetalGenericRenderPipelineDescriptor::Mesh(ref inner) => {
                        unsafe { inner.setRasterSampleCount(desc.multisample.count as _) };
                    }
                }
                descriptor.setAlphaToCoverageEnabled(desc.multisample.alpha_to_coverage_enabled);
                //descriptor.set_alpha_to_one_enabled(desc.multisample.alpha_to_one_enabled);
            }

            // Set debug label
            if let Some(name) = desc.label {
                descriptor.setLabel(Some(&NSString::from_str(name)));
            }
            if let Some(mv) = desc.multiview_mask {
                unsafe {
                    descriptor.setMaxVertexAmplificationCount(mv.get().count_ones() as usize)
                };
            }

            // Create the pipeline from descriptor
            let raw = match descriptor {
                MetalGenericRenderPipelineDescriptor::Standard(d) => through_archive(
                    desc.cache,
                    &self.archive_additions,
                    compile_admission::service(),
                    desc.label,
                    source_bytes,
                    |archives, options| {
                        d.setBinaryArchives(archives);
                        self.shared
                            .device
                            .newRenderPipelineStateWithDescriptor_options_reflection_error(
                                &d, options, None,
                            )
                    },
                    {
                        let d = d.clone();
                        move |archive: &ProtocolObject<dyn MTLBinaryArchive>| archive.addRenderPipelineFunctionsWithDescriptor_error(&d)
                    },
                ),
                MetalGenericRenderPipelineDescriptor::Mesh(d) => {
                    compile_admission::service().compile(desc.label, source_bytes, || {
                        self.shared
                            .device
                            .newRenderPipelineStateWithMeshDescriptor_options_reflection_error(
                                &d,
                                MTLPipelineOption::empty(),
                                None,
                            )
                            .map_err(|e| compile_admission::Failure::new(&e, format!("{e:?}")))
                    })
                }
            }
            .map_err(|e| {
                crate::PipelineError::Linkage(
                    wgt::ShaderStages::VERTEX | wgt::ShaderStages::FRAGMENT,
                    format!("new_render_pipeline_state: {e}"),
                )
            })?;

            self.counters.render_pipelines.add(1);

            Ok(super::RenderPipeline {
                raw,
                vs_info,
                fs_info,
                ts_info,
                ms_info,
                raw_primitive_type,
                raw_triangle_fill_mode,
                raw_front_winding: conv::map_winding(desc.primitive.front_face),
                raw_cull_mode: conv::map_cull_mode(desc.primitive.cull_mode),
                raw_depth_clip_mode: if self.features.contains(wgt::Features::DEPTH_CLIP_CONTROL) {
                    Some(if desc.primitive.unclipped_depth {
                        MTLDepthClipMode::Clamp
                    } else {
                        MTLDepthClipMode::Clip
                    })
                } else {
                    None
                },
                depth_stencil,
            })
        })
    }

    unsafe fn destroy_render_pipeline(&self, _pipeline: super::RenderPipeline) {
        self.counters.render_pipelines.sub(1);
    }

    unsafe fn create_compute_pipeline(
        &self,
        desc: &crate::ComputePipelineDescriptor<
            super::PipelineLayout,
            super::ShaderModule,
            super::PipelineCache,
        >,
    ) -> Result<super::ComputePipeline, crate::PipelineError> {
        autoreleasepool(|_| {
            let descriptor = MTLComputePipelineDescriptor::new();

            let cs = self.load_shader(
                &desc.stage,
                &[],
                desc.layout,
                MTLPrimitiveTopologyClass::Unspecified,
                naga::ShaderStage::Compute,
                desc.label,
            )?;

            descriptor.setComputeFunction(Some(&cs.function));
            let cs_source_bytes = cs.source_bytes;

            // https://developer.apple.com/documentation/metal/mtlpipelinebufferdescriptor/mutability
            if available!(macos = 10.13, ios = 11.0, tvos = 11.0, visionos = 1.0) {
                Self::set_buffers_mutability(&descriptor.buffers(), cs.immutable_buffer_mask);
            }

            let cs_info = super::PipelineStageInfo {
                library: Some(cs.library),
                immediates: desc.layout.immediates_infos.cs,
                sizes_slot: desc.layout.per_stage_map.cs.sizes_buffer,
                sized_bindings: cs.sized_bindings,
                vertex_buffer_mappings: vec![],
                raw_wg_size: cs.wg_size,
                work_group_memory_sizes: cs.wg_memory_sizes,
            };

            if let Some(name) = desc.label {
                descriptor.setLabel(Some(&NSString::from_str(name)));
            }

            let raw = through_archive(
                desc.cache,
                &self.archive_additions,
                compile_admission::service(),
                desc.label,
                cs_source_bytes,
                |archives, options| {
                    descriptor.setBinaryArchives(archives);
                    self.shared
                        .device
                        .newComputePipelineStateWithDescriptor_options_reflection_error(
                            &descriptor,
                            options,
                            None,
                        )
                },
                {
                    let descriptor = descriptor.clone();
                    move |archive: &ProtocolObject<dyn MTLBinaryArchive>| archive.addComputePipelineFunctionsWithDescriptor_error(&descriptor)
                },
            );

            let raw: Retained<ProtocolObject<dyn MTLComputePipelineState>> =
                raw.map_err(|e| {
                    crate::PipelineError::Linkage(
                        wgt::ShaderStages::COMPUTE,
                        format!("new_compute_pipeline_state: {e}"),
                    )
                })?;

            self.counters.compute_pipelines.add(1);

            Ok(super::ComputePipeline { raw, cs_info })
        })
    }

    unsafe fn destroy_compute_pipeline(&self, _pipeline: super::ComputePipeline) {
        self.counters.compute_pipelines.sub(1);
    }

    unsafe fn create_pipeline_cache(
        &self,
        desc: &crate::PipelineCacheDescriptor<'_>,
    ) -> Result<super::PipelineCache, crate::PipelineCacheError> {
        autoreleasepool(|_| {
            let descriptor = MTLBinaryArchiveDescriptor::new();
            let source = match desc.data {
                Some(data) => {
                    if !super::archive_files_open() {
                        return Err(crate::PipelineCacheError::Rejected(
                            "the process released its Metal binary archive files for exit".into(),
                        ));
                    }
                    let path = archive_path();
                    std::fs::write(&path, data).map_err(|error| {
                        crate::PipelineCacheError::Rejected(alloc::format!(
                            "cannot stage the Metal binary archive at {}: {error}",
                            path.display()
                        ))
                    })?;
                    descriptor.setUrl(Some(&file_url(&path)));
                    Some(path)
                }
                None => None,
            };
            match self.shared.device.newBinaryArchiveWithDescriptor_error(&descriptor) {
                Ok(archive) => {
                    let files = Arc::new(Mutex::new(source.into_iter().collect()));
                    super::register_archive_files(&files);
                    Ok(super::PipelineCache { archive, files, deferred: Default::default() })
                }
                Err(error) => {
                    if let Some(path) = &source {
                        let _ = std::fs::remove_file(path);
                    }
                    Err(crate::PipelineCacheError::Rejected(alloc::format!(
                        "Metal could not open the binary archive: {}",
                        error.localizedDescription()
                    )))
                }
            }
        })
    }
    fn pipeline_cache_validation_key(&self) -> Option<[u8; 16]> {
        // The OS build supplies the Metal compiler; the device name the GPU.
        let os = NSProcessInfo::processInfo().operatingSystemVersionString();
        let os = alloc::string::ToString::to_string(&*os);
        let name = alloc::string::ToString::to_string(&*self.shared.device.name());
        let parts: [&[u8]; 3] = [b"wgpu-metal-binary-archive-v1", name.as_bytes(), os.as_bytes()];
        let mut key = [0; 16];
        key[..8].copy_from_slice(&fnv(0xcbf2_9ce4_8422_2325, &parts).to_be_bytes());
        key[8..].copy_from_slice(&fnv(0x8422_2325_cbf2_9ce4, &parts).to_be_bytes());
        Some(key)
    }
    unsafe fn pipeline_cache_get_data(&self, cache: &super::PipelineCache) -> Option<Vec<u8>> {
        autoreleasepool(|_| {
            let mut files = cache.files.lock();
            // After release_archive_files_for_exit no backing file may appear.
            if !super::archive_files_open() {
                return None;
            }
            cache.record_deferred(compile_admission::service());
            let path = archive_path();
            let serialized = cache.archive.serializeToURL_error(&file_url(&path));
            // Serialization rebinds the archive to this file, even when reading it back fails.
            files.push(path.clone());
            serialized.ok().and_then(|()| std::fs::read(&path).ok())
        })
    }
    unsafe fn destroy_pipeline_cache(&self, _: super::PipelineCache) {}

    unsafe fn create_query_set(
        &self,
        desc: &wgt::QuerySetDescriptor<crate::Label>,
    ) -> DeviceResult<super::QuerySet> {
        autoreleasepool(|_| {
            match desc.ty {
                wgt::QueryType::Occlusion => {
                    let size = desc.count as u64 * crate::QUERY_SIZE;
                    let options = MTLResourceOptions::empty();
                    //TODO: HazardTrackingModeUntracked
                    let raw_buffer = self
                        .shared
                        .device
                        .newBufferWithLength_options(size as usize, options)
                        .unwrap();
                    if let Some(label) = desc.label {
                        raw_buffer.setLabel(Some(&NSString::from_str(label)));
                    }
                    Ok(super::QuerySet {
                        raw_buffer,
                        counter_sample_buffer: None,
                        ty: desc.ty,
                    })
                }
                wgt::QueryType::Timestamp => {
                    let size = desc.count as u64 * crate::QUERY_SIZE;
                    let device = &self.shared.device;
                    let destination_buffer = device
                        .newBufferWithLength_options(size as usize, MTLResourceOptions::empty())
                        .unwrap();

                    let csb_desc = MTLCounterSampleBufferDescriptor::new();
                    csb_desc.setStorageMode(MTLStorageMode::Shared);
                    unsafe { csb_desc.setSampleCount(desc.count as _) };
                    if let Some(label) = desc.label {
                        csb_desc.setLabel(&NSString::from_str(label));
                    }

                    let counter_sets = device.counterSets().unwrap();
                    let timestamp_counter = match counter_sets
                        .iter()
                        .find(|cs| &*cs.name() == ns_string!("timestamp"))
                    {
                        Some(counter) => counter,
                        None => {
                            log::error!("Failed to obtain timestamp counter set.");
                            return Err(crate::DeviceError::Unexpected);
                        }
                    };
                    csb_desc.setCounterSet(Some(&timestamp_counter));

                    let counter_sample_buffer =
                        match device.newCounterSampleBufferWithDescriptor_error(&csb_desc) {
                            Ok(buffer) => buffer,
                            Err(err) => {
                                log::error!("Failed to create counter sample buffer: {err:?}");
                                return Err(crate::DeviceError::Unexpected);
                            }
                        };

                    self.counters.query_sets.add(1);

                    Ok(super::QuerySet {
                        raw_buffer: destination_buffer,
                        counter_sample_buffer: Some(counter_sample_buffer),
                        ty: desc.ty,
                    })
                }
                _ => {
                    todo!()
                }
            }
        })
    }

    unsafe fn destroy_query_set(&self, _set: super::QuerySet) {
        self.counters.query_sets.sub(1);
    }

    unsafe fn create_fence(&self) -> DeviceResult<super::Fence> {
        self.counters.fences.add(1);
        // https://developer.apple.com/documentation/metal/mtlsharedevent
        let shared_event = if available!(macos = 10.14, ios = 12.0, tvos = 12.0, visionos = 1.0) {
            self.shared.device.newSharedEvent() // This should be supported on said devices, but some sandbox environments may still restrict it, making it return `None`.
        } else {
            None
        };
        Ok(super::Fence {
            completion: None,
            completed_value: 0,
            pending_signals: Vec::new(),
            shared_event,
        })
    }

    unsafe fn destroy_fence(&self, _fence: super::Fence) {
        self.counters.fences.sub(1);
    }

    unsafe fn get_fence_value(&self, fence: &super::Fence) -> DeviceResult<crate::FenceValue> {
        fence.get_latest()
    }
    unsafe fn wait(
        &self,
        fence: &super::Fence,
        wait_value: crate::FenceValue,
        timeout: Option<core::time::Duration>,
    ) -> DeviceResult<bool> {
        let start = time::Instant::now();
        loop {
            // Poll every buffer even when this fence value was already reached.
            // A device failure must precede readback, not become a successful wait.
            if fence.get_latest()? >= wait_value { return Ok(true); }
            // The queue serial that signals this value: every buffer up to it must finish.
            let Some(serial) = fence.pending_signals.iter().find(|(value, _)| *value >= wait_value).map(|(_, serial)| *serial) else {
                return Err(crate::DeviceError::LostWithReason(format!(
                    "Metal has no active command buffer for fence value {wait_value}"
                )));
            };
            let remaining = match timeout {
                Some(timeout) => match timeout.checked_sub(start.elapsed()) {
                    Some(remaining) if !remaining.is_zero() => Some(remaining),
                    _ => return Ok(false),
                },
                None => None,
            };
            // A pending signal was pushed by a submission, which set the fence's queue. Sleep until the queue's
            // completed handlers have reported every buffer up to that serial (one wake-up, when the last of them
            // finishes), then poll again: the statuses are what publishes the fence.
            let Some(completion) = fence.completion.as_ref() else { return Ok(false) };
            completion.wait_for(serial, remaining);
        }
    }

    unsafe fn start_graphics_debugger_capture(&self) -> bool {
        // https://developer.apple.com/documentation/metal/mtlcapturemanager
        if !available!(macos = 10.13, ios = 11.0, tvos = 11.0, visionos = 1.0) {
            return false;
        }
        let device = &self.shared.device;
        let shared_capture_manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        let default_capture_scope = shared_capture_manager.newCaptureScopeWithDevice(device);
        shared_capture_manager.setDefaultCaptureScope(Some(&default_capture_scope));
        #[allow(deprecated)]
        shared_capture_manager.startCaptureWithScope(&default_capture_scope);
        default_capture_scope.beginScope();
        true
    }

    unsafe fn stop_graphics_debugger_capture(&self) {
        let shared_capture_manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if let Some(default_capture_scope) = shared_capture_manager.defaultCaptureScope() {
            default_capture_scope.endScope();
        }
        shared_capture_manager.stopCapture();
    }

    unsafe fn get_acceleration_structure_build_sizes(
        &self,
        descriptor: &crate::GetAccelerationStructureBuildSizesDescriptor<super::Buffer>,
    ) -> crate::AccelerationStructureBuildSizes {
        let acceleration_structure_descriptor =
            conv::map_acceleration_structure_descriptor(descriptor.entries, descriptor.flags);
        let info = self
            .shared
            .device
            .accelerationStructureSizesWithDescriptor(&acceleration_structure_descriptor);
        crate::AccelerationStructureBuildSizes {
            acceleration_structure_size: info.accelerationStructureSize as u64,
            update_scratch_size: info.refitScratchBufferSize as u64,
            build_scratch_size: info.buildScratchBufferSize as u64,
        }
    }

    unsafe fn get_acceleration_structure_device_address(
        &self,
        acceleration_structure: &super::AccelerationStructure,
    ) -> wgt::BufferAddress {
        acceleration_structure.raw.gpuResourceID().to_raw()
    }

    unsafe fn create_acceleration_structure(
        &self,
        descriptor: &crate::AccelerationStructureDescriptor,
    ) -> Result<super::AccelerationStructure, crate::DeviceError> {
        // self.counters.acceleration_structures.add(1);
        autoreleasepool(|_| {
            Ok(super::AccelerationStructure {
                raw: self
                    .shared
                    .device
                    .newAccelerationStructureWithSize(descriptor.size as usize)
                    .ok_or(crate::DeviceError::OutOfMemory)?,
            })
        })
    }

    unsafe fn destroy_acceleration_structure(
        &self,
        _acceleration_structure: super::AccelerationStructure,
    ) {
        // self.counters.acceleration_structures.sub(1);
    }

    fn tlas_instance_layout(&self) -> Option<crate::TlasInstanceLayout> {
        Some(crate::TlasInstanceLayout { stride: core::mem::size_of::<MTLIndirectAccelerationStructureInstanceDescriptor>() as u32, transform: [0,3,6,9,1,4,7,10,2,5,8,11], custom_index: (core::mem::offset_of!(MTLIndirectAccelerationStructureInstanceDescriptor, userID) / 4) as u32, mask: (core::mem::offset_of!(MTLIndirectAccelerationStructureInstanceDescriptor, mask) / 4) as u32, mask_shift: 0, triangle_flags: (core::mem::offset_of!(MTLIndirectAccelerationStructureInstanceDescriptor, options) / 4) as u32, disable_triangle_culling: MTLAccelerationStructureInstanceOptions::DisableTriangleCulling.bits() as u32, force_non_opaque: MTLAccelerationStructureInstanceOptions::NonOpaque.bits() as u32 })
    }

    fn tlas_instance_to_bytes(&self, instance: TlasInstance) -> Vec<u8> {
        let temp = MTLIndirectAccelerationStructureInstanceDescriptor {
            transformationMatrix: MTLPackedFloat4x3 {
                columns: [
                    MTLPackedFloat3 {
                        x: instance.transform[0],
                        y: instance.transform[4],
                        z: instance.transform[8],
                    },
                    MTLPackedFloat3 {
                        x: instance.transform[1],
                        y: instance.transform[5],
                        z: instance.transform[9],
                    },
                    MTLPackedFloat3 {
                        x: instance.transform[2],
                        y: instance.transform[6],
                        z: instance.transform[10],
                    },
                    MTLPackedFloat3 {
                        x: instance.transform[3],
                        y: instance.transform[7],
                        z: instance.transform[11],
                    },
                ],
            },
            options: MTLAccelerationStructureInstanceOptions::None,
            mask: instance.mask as u32,
            intersectionFunctionTableOffset: 0,
            userID: instance.custom_data,
            accelerationStructureID: unsafe { MTLResourceID::from_raw(instance.blas_address) },
        };

        wgt::bytemuck_wrapper!(unsafe struct Desc(MTLIndirectAccelerationStructureInstanceDescriptor));

        bytemuck::bytes_of(&Desc::wrap(temp)).to_vec()
    }

    fn get_internal_counters(&self) -> wgt::HalCounters {
        self.counters.as_ref().clone()
    }

    fn check_if_oom(&self) -> Result<(), crate::DeviceError> {
        // TODO: see https://github.com/gfx-rs/wgpu/issues/7460

        Ok(())
    }
}

#[cfg(test)]
mod argument_buffer_tests {
    use super::*;
    #[test]
    fn flexible_argument_tables_use_each_binding_capacity() {
        let source = r#"
struct Textures;
struct Textures { texture2d<float> handles[]; };
struct Samplers { sampler handles[]; };
struct Buffers { uint device* buffers[]; };
struct Fixed { uint device* buffers[3]; };
kernel void check(Textures constant* t [[buffer(14)]], Samplers constant* s [[buffer(15)]],
                  Buffers constant* b [[buffer(21)]], Fixed constant* f [[buffer(22)]]) {}
"#;
        for (textures, samplers, buffers) in [(1, 1, 1), (1025, 65, 7)] {
            let actual = specialize_argument_arrays(source, &[(14, textures), (15, samplers), (21, buffers)]).unwrap();
            assert!(actual.contains(&format!("texture2d<float> handles[{textures}];")));
            assert!(actual.contains(&format!("sampler handles[{samplers}];")));
            assert!(actual.contains(&format!("uint device* buffers[{buffers}];")));
            assert!(actual.contains("uint device* buffers[3];"));
            assert!(!actual.contains("[];"));
        }
        assert!(specialize_argument_arrays(source, &[(14, 1)]).is_err());
        let conflict = source.replace("Fixed constant* f", "Textures constant* f");
        assert!(specialize_argument_arrays(&conflict, &[(14, 1), (15, 1), (21, 1), (22, 2)]).is_err());
        let unknown = "struct Mixed { uint count; sampler handles[]; };";
        assert!(specialize_argument_arrays(unknown, &[]).is_err());
    }



    /// Execute the last declared slot, including a padded tail. Real resource
    /// handles and a nonzero buffer offset exercise the native Tier-2 ABI.
    #[test]
    #[allow(deprecated)] // Match the HAL's sampled-texture residency usage.
    fn native_argument_prefix_tail_execution() {
        use objc2_metal::{MTLCommandQueue, MTLComputeCommandEncoder,
            MTLCommandEncoder, MTLResourceUsage, MTLRegion, MTLOrigin, MTLTextureUsage};
        autoreleasepool(|_| {
            let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required");
            let queue = device.newCommandQueue().expect("Metal command queue required");
            let texture_desc = MTLTextureDescriptor::new();
            texture_desc.setPixelFormat(MTLPixelFormat::RGBA32Float);
            texture_desc.setStorageMode(MTLStorageMode::Shared);
            texture_desc.setUsage(MTLTextureUsage::ShaderRead);
            unsafe { texture_desc.setWidth(1); texture_desc.setHeight(1); }
            let texture = device.newTextureWithDescriptor(&texture_desc).unwrap();
            let pixel = [3.0f32; 4];
            unsafe { texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                MTLRegion { origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize { width: 1, height: 1, depth: 1 } },
                0, NonNull::new(pixel.as_ptr().cast_mut().cast()).unwrap(), 16); }
            let sampler_desc = MTLSamplerDescriptor::new();
            sampler_desc.setSupportArgumentBuffers(true);
            let sampler = device.newSamplerStateWithDescriptor(&sampler_desc).unwrap();
            let data = device.newBufferWithLength_options((1024 + 64) * 4, MTLResourceOptions::StorageModeShared).unwrap();
            for index in 0..1024 {
                unsafe { data.contents().cast::<u32>().as_ptr().add(64 + index).write(17 + index as u32); }
            }
            for capacity in [16, 64, 1024] {
                let source = format!(r#"
#include <metal_stdlib>
using namespace metal;
struct Args {{ array<texture2d<float>, {capacity}> t [[id(0)]];
    array<sampler, {capacity}> s [[id({capacity})]];
    array<device uint*, {capacity}> b [[id({twice})]]; }};
kernel void check(constant Args& a [[buffer(0)]], device uint* out [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {{
    out[i] = a.b[i][0] + uint(a.t[i].sample(a.s[i], float2(0.5)).x);
}}
"#, twice = capacity * 2);
                let library = device.newLibraryWithSource_options_error(&NSString::from_str(&source), None).unwrap();
                let function = library.newFunctionWithName(&NSString::from_str("check")).unwrap();
                let pipeline = device.newComputePipelineStateWithFunction_error(&function).unwrap();
                for supplied in [1, 2, capacity] {
                    // Three contiguous arrays use the same production allocation/tail helper.
                    let args = new_argument_buffer(&device, core::num::NonZeroU32::new(capacity * 3).unwrap());
                    assert_eq!(args.length(), capacity as usize * 3 * 8);
                    let slots = unsafe { core::slice::from_raw_parts_mut(args.contents().cast::<u64>().as_ptr(), capacity as usize * 3) };
                    for (kind, (array, handle)) in slots.chunks_exact_mut(capacity as usize).zip([
                        texture.gpuResourceID().to_raw(), sampler.gpuResourceID().to_raw(), data.gpuAddress() + 256]).enumerate() {
                        array.fill(u64::MAX);
                        for (index, slot) in array[..supplied as usize].iter_mut().enumerate() {
                            *slot = handle + if kind == 2 { index as u64 * 4 } else { 0 };
                        }
                        fill_argument_buffer_tail(array, supplied as usize);
                    }
                    let output = device.newBufferWithLength_options(capacity as usize * 4, MTLResourceOptions::StorageModeShared).unwrap();
                    let command = queue.commandBuffer().unwrap();
                    let encoder = command.computeCommandEncoder().unwrap();
                    encoder.setComputePipelineState(&pipeline);
                    unsafe {
                        encoder.setBuffer_offset_atIndex(Some(&args), 0, 0);
                        encoder.setBuffer_offset_atIndex(Some(&output), 0, 1);
                    }
                    encoder.useResource_usage(ProtocolObject::from_ref(&*data), MTLResourceUsage::Read);
                    encoder.useResource_usage(ProtocolObject::from_ref(&*texture), MTLResourceUsage::Sample);
                    encoder.dispatchThreads_threadsPerThreadgroup(
                        MTLSize { width: capacity as usize, height: 1, depth: 1 },
                        MTLSize { width: 1, height: 1, depth: 1 });
                    encoder.endEncoding();
                    command.commit();
                    command.waitUntilCompleted();
                    assert_eq!(command.status(), MTLCommandBufferStatus::Completed, "{:?}", command.error());
                    let result = unsafe { core::slice::from_raw_parts(output.contents().cast::<u32>().as_ptr(), capacity as usize) };
                    for (index, &value) in result.iter().enumerate() {
                        let expected = 20 + if index < supplied as usize { index as u32 } else { 0 };
                        assert_eq!(value, expected, "capacity={capacity} prefix={supplied} slot={index}");
                    }
                }
            }
        });
    }

    #[test]
    fn partially_bound_arrays_have_full_storage_and_valid_tail() {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal device required");
        // Texture, sampler, and buffer array capacities; test both one and
        // multiple supplied handles, plus the fully bound case.
        for capacity in [1024, 64, 16] {
            for supplied in [1, 2, capacity] {
                let buffer =
                    new_argument_buffer(&device, core::num::NonZeroU32::new(capacity).unwrap());
                assert_eq!(buffer.length(), capacity as usize * 8);
                let contents = unsafe {
                    core::slice::from_raw_parts_mut(
                        buffer.contents().cast::<u64>().as_ptr(),
                        capacity as usize,
                    )
                };
                // Poison all slots so the test catches an uninitialized tail.
                contents.fill(u64::MAX);
                for (index, handle) in contents[..supplied as usize].iter_mut().enumerate() {
                    *handle = 100 + index as u64;
                }
                fill_argument_buffer_tail(contents, supplied as usize);
                for (index, &handle) in contents.iter().enumerate() {
                    assert_eq!(
                        handle,
                        if index < supplied as usize {
                            100 + index as u64
                        } else {
                            100
                        }
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod binary_archive_tests {
    use super::*;
    use crate::Device as _;

    fn add(descriptor: &Retained<MTLComputePipelineDescriptor>) -> impl Fn(&ProtocolObject<dyn MTLBinaryArchive>) -> Result<(), Retained<NSError>> + Clone + 'static {
        let descriptor = descriptor.clone();
        move |archive: &ProtocolObject<dyn MTLBinaryArchive>| archive.addComputePipelineFunctionsWithDescriptor_error(&descriptor)
    }

    /// A process serializes its archive more than once: a session after each render that
    /// compiled something, a 3D shot after preparation and again at its end. Pipelines added
    /// after one serialization must be in the next, and one that met a serialization in
    /// progress (deferred) must be in the one after it.
    #[test]
    fn additions_after_a_serialization_are_in_the_next_serialization() {
        autoreleasepool(|_| {
            let raw = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required");
            let device = unsafe { super::super::Device::device_from_raw(raw.clone(), wgt::Features::PIPELINE_CACHE) };
            let source = alloc::format!(
                "#include <metal_stdlib>\nkernel void first(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i * {0}u + 11u; }}\nkernel void second(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i * {0}u + 13u; }}\nkernel void third(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i * {0}u + 17u; }}\n",
                std::process::id()
            );
            let library = raw.newLibraryWithSource_options_error(&NSString::from_str(&source), None).unwrap();
            let descriptors: Vec<_> = ["first", "second", "third"].into_iter().map(|name| {
                let descriptor = MTLComputePipelineDescriptor::new();
                descriptor.setComputeFunction(Some(&library.newFunctionWithName(&NSString::from_str(name)).unwrap()));
                descriptor
            }).collect();
            let create = |descriptor: &Retained<MTLComputePipelineDescriptor>| {
                let (raw, descriptor) = (raw.clone(), descriptor.clone());
                move |archives: Option<&NSArray<ProtocolObject<dyn MTLBinaryArchive>>>, options| {
                    descriptor.setBinaryArchives(archives);
                    raw.newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor, options, None)
                }
            };
            let cache = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: None }) }.unwrap();
            through_archive(Some(&cache), &device.archive_additions, compile_admission::service(), None, 0, create(&descriptors[0]), add(&descriptors[0])).unwrap();
            let one = unsafe { device.pipeline_cache_get_data(&cache) }.expect("first serialization");
            through_archive(Some(&cache), &device.archive_additions, compile_admission::service(), None, 0, create(&descriptors[1]), add(&descriptors[1])).unwrap();
            // The third meets a serialization in progress: it is created at once and recorded later.
            // It is counted only once it is queued, where the next serialization finds it.
            {
                let _serializing = cache.files.lock();
                assert_eq!(device.archive_additions(), 2);
                through_archive(Some(&cache), &device.archive_additions, compile_admission::service(), Some("third"), 4096, create(&descriptors[2]), add(&descriptors[2])).unwrap();
                assert_eq!((cache.deferred.lock().len(), device.archive_additions()), (1, 3));
                // The deferred recording keeps what it is admitted with when it is drained.
                let deferred = cache.deferred.lock();
                assert_eq!((deferred[0].0.label(), deferred[0].0.source_bytes()), (Some("third"), 4096));
            }
            // The caller's protocol: read the count, then serialize. Everything counted is written.
            let counted = device.archive_additions();
            let three = unsafe { device.pipeline_cache_get_data(&cache) }.expect("second serialization");
            assert_eq!(counted, 3);
            assert!(cache.deferred.lock().is_empty());
            assert!(three.len() > one.len(), "the second serialization holds no more than the first: {} and {} bytes", three.len(), one.len());
            let reloaded = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: Some(&three) }) }.unwrap();
            let archives = NSArray::from_retained_slice(&[reloaded.archive.clone()]);
            for (name, descriptor) in ["first", "second", "third"].into_iter().zip(&descriptors) {
                assert!(create(descriptor)(Some(&archives), MTLPipelineOption::FailOnBinaryArchiveMiss).is_ok(),
                    "the second serialization lost the pipeline {name}");
            }
        });
    }

    /// A pipeline compiled through an empty archive is stored; the serialized
    /// archive, reloaded as a new cache, serves it without compiling, and data
    /// Metal cannot open is a named rejection rather than device loss.
    #[test]
    fn binary_archive_round_trip_serves_pipelines_without_compiling() {
        autoreleasepool(|_| {
            let raw = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required");
            let device = unsafe { super::super::Device::device_from_raw(raw.clone(), wgt::Features::PIPELINE_CACHE) };
            let source = alloc::format!(
                "#include <metal_stdlib>\nkernel void archived(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i * {}u + 7u; }}\n",
                std::process::id()
            );
            let library = raw.newLibraryWithSource_options_error(&NSString::from_str(&source), None).unwrap();
            let function = library.newFunctionWithName(&NSString::from_str("archived")).unwrap();
            let descriptor = MTLComputePipelineDescriptor::new();
            descriptor.setComputeFunction(Some(&function));
            let create = |archives: Option<&NSArray<ProtocolObject<dyn MTLBinaryArchive>>>, options| {
                descriptor.setBinaryArchives(archives);
                raw.newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor, options, None)
            };
            let empty = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: None }) }.unwrap();
            let only_empty = NSArray::from_retained_slice(&[empty.archive.clone()]);
            // The probe really requires an archive entry, whatever Apple's shared cache holds.
            assert!(create(Some(&only_empty), MTLPipelineOption::FailOnBinaryArchiveMiss).is_err());
            assert_eq!(device.archive_additions(), 0);
            through_archive(Some(&empty), &device.archive_additions, compile_admission::service(), None, 0, &create, add(&descriptor)).unwrap();
            assert!(create(Some(&only_empty), MTLPipelineOption::FailOnBinaryArchiveMiss).is_ok());
            // The miss was recorded as an addition.
            assert_eq!(device.archive_additions(), 1);

            let data = unsafe { device.pipeline_cache_get_data(&empty) }.expect("serialized archive");
            let again = unsafe { device.pipeline_cache_get_data(&empty) }.expect("an archive serializes repeatedly");
            assert_eq!(again.len(), data.len());
            assert!(!data.is_empty());
            let reloaded = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: Some(&data) }) }.unwrap();
            let staged = reloaded.files.lock()[0].clone();
            assert!(staged.exists());
            let archives = NSArray::from_retained_slice(&[reloaded.archive.clone()]);
            assert!(create(Some(&archives), MTLPipelineOption::FailOnBinaryArchiveMiss).is_ok());
            // A hit neither compiles nor adds to the archive.
            through_archive(Some(&reloaded), &device.archive_additions, compile_admission::service(), None, 0, &create, |_| panic!("an archive hit must not compile or add")).unwrap();
            assert_eq!(device.archive_additions(), 1);
            let saved = unsafe { device.pipeline_cache_get_data(&reloaded) }.expect("a reloaded archive serializes");
            assert_eq!(saved.len(), data.len());
            let files = reloaded.files.lock().clone();
            drop(reloaded);
            assert!(files.iter().all(|file| !file.exists()), "dropping the cache removes its files");

            let rejected = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: Some(b"not a metal archive") }) };
            match rejected {
                Err(crate::PipelineCacheError::Rejected(reason)) => assert!(reason.contains("binary archive"), "{reason}"),
                other => panic!("corrupt archive data must be a named rejection, got {other:?}"),
            }
            // The OS build and GPU name key the archive; the key is stable within a process.
            assert_eq!(device.pipeline_cache_validation_key(), device.pipeline_cache_validation_key());
        });
    }

    /// A process that ends without destructors (vork's successful one-shot
    /// `_exit`) releases live caches' backing files first; afterwards nothing new
    /// is written, while the in-memory archive still serves pipelines. Runs in a
    /// child process because the release is process-wide and final.
    #[test]
    fn exit_release_removes_backing_files_and_stops_new_ones() {
        const CHILD: &str = "WGPU_METAL_ARCHIVE_EXIT_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "metal::device::binary_archive_tests::exit_release_removes_backing_files_and_stops_new_ones", "--nocapture"])
                .env(CHILD, "1").status().unwrap();
            assert!(status.success(), "archive exit child failed");
            return;
        }
        autoreleasepool(|_| {
            let raw = objc2_metal::MTLCreateSystemDefaultDevice().expect("Metal GPU required");
            let device = unsafe { super::super::Device::device_from_raw(raw.clone(), wgt::Features::PIPELINE_CACHE) };
            let source = alloc::format!(
                "#include <metal_stdlib>\nkernel void exiting(device uint* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {{ out[i] = i * {}u + 3u; }}\n",
                std::process::id()
            );
            let library = raw.newLibraryWithSource_options_error(&NSString::from_str(&source), None).unwrap();
            let function = library.newFunctionWithName(&NSString::from_str("exiting")).unwrap();
            let descriptor = MTLComputePipelineDescriptor::new();
            descriptor.setComputeFunction(Some(&function));
            let create = |archives: Option<&NSArray<ProtocolObject<dyn MTLBinaryArchive>>>, options| {
                descriptor.setBinaryArchives(archives);
                raw.newComputePipelineStateWithDescriptor_options_reflection_error(&descriptor, options, None)
            };
            let empty = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: None }) }.unwrap();
            through_archive(Some(&empty), &device.archive_additions, compile_admission::service(), None, 0, &create, add(&descriptor)).unwrap();
            let data = unsafe { device.pipeline_cache_get_data(&empty) }.expect("serialized archive");
            let warm = unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: Some(&data) }) }.unwrap();
            let files: Vec<_> = [empty.files.lock().clone(), warm.files.lock().clone()].concat();
            assert_eq!(files.len(), 2);
            assert!(files.iter().all(|file| file.exists()));

            assert_eq!(super::super::release_archive_files_for_exit(), 2);
            assert!(files.iter().all(|file| !file.exists()), "exit release left backing files");
            // The released archive still serves the stored pipeline without compiling.
            through_archive(Some(&warm), &device.archive_additions, compile_admission::service(), None, 0, &create, |_| panic!("a released archive must still hit")).unwrap();
            // No new backing file appears after the release.
            assert!(unsafe { device.pipeline_cache_get_data(&warm) }.is_none());
            assert!(warm.files.lock().is_empty());
            assert!(matches!(
                unsafe { device.create_pipeline_cache(&crate::PipelineCacheDescriptor { label: None, data: Some(&data) }) },
                Err(crate::PipelineCacheError::Rejected(_))
            ));
        });
    }
}
