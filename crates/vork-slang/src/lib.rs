//! Unstable, internal: all Rust APIs in this crate are free to change.
//!
//! Slang → WGSL program compiler, shared by build scripts and runtime install.
//! Persistent artifacts belong to the renderer's content-addressed store; this
//! crate is pure (source in, WGSL + layout out) and never caches artifacts.
//!
//! A program is compiled once, independently of every other program, into a
//! WGSL fragment whose identifiers carry a per-program tag, so any set of
//! programs can be concatenated into one scene shader. The kernel side of the
//! boundary is described by [`Abi`]; see docs/PROGRAMS.md.
#[cfg(all(test, unix))]
#[path = "../../test_support.rs"]
mod test_support;

mod image_native;
mod identity;
mod filesystem;
pub use identity::{ClosedInputIdentity, InputIdentity, PreparedModules, PreparedProgram};
pub mod metal_source;
pub mod spirv;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

/// `Command::output` through [`spawn_compiler`].
fn compiler_output(command: &mut Command) -> std::io::Result<std::process::Output> {
    use std::process::Stdio;
    command.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    spawn_compiler(command)?.wait_with_output()
}

/// Compiler subprocesses must not outlive a one-shot render that exits while
/// unused variants are still compiling. Outputs are private workspace files;
/// only the parent publishes completed artifacts into the persistent cache.
///
/// Children are created with `posix_spawn` (vfork + exec, no code run in the
/// child): a `pre_exec` hook would make std fall back to `fork()`, copying the
/// page tables of a multi-GB renderer for every one of thousands of compiles.
/// Instead every compiler joins the process group of a guard process whose
/// stdin is a pipe only this process holds. The kernel closes that pipe however
/// this process ends (return, `exit`, `_exit`, a signal, SIGKILL); the guard's
/// `read` then returns at EOF and it kills its whole group. The guard ignores
/// SIGHUP, SIGINT and SIGTERM, so only a deliberate SIGKILL of it can leave its
/// compilers unwatched; the next spawn then kills that group and starts a new
/// guard (spawning under the lock keeps compilers out of a retired group).
#[cfg(unix)]
fn spawn_compiler(command: &mut Command) -> std::io::Result<std::process::Child> {
    use std::{os::unix::process::CommandExt, process::{Child, Stdio}};
    static GUARD: Mutex<Option<Child>> = Mutex::new(None);
    let mut guard = GUARD.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(dead) = guard.take_if(|child| !matches!(child.try_wait(), Ok(None))) {
        // SAFETY: plain kill(2) of the group this process created; the reaped
        // leader's pid stays the group id while any member is alive.
        unsafe { libc::kill(-(dead.id() as libc::pid_t), libc::SIGKILL) };
    }
    let leader = match &mut *guard {
        Some(child) => child.id(),
        None => guard.insert(Command::new("/bin/sh")
            .args(["-c", "trap '' HUP INT TERM; read line; kill -9 0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .process_group(0)
            .spawn()?).id(),
    };
    command.process_group(leader as libc::pid_t).spawn()
}

#[cfg(not(unix))]
fn spawn_compiler(command: &mut Command) -> std::io::Result<std::process::Child> {
    command.spawn()
}

mod effects;
mod input_values;
mod trace;
pub use trace::QueryFilter;
pub use effects::EffectProbe;
mod enums;
pub use enums::InputEnum;
mod coordinates;
mod attributes;
mod declarations;
mod hair;
pub use hair::{HAIR_VEC4S, hair_fields};
mod openpbr;
mod native_blend;
mod native_value;
/// Outline generated material functions with a value-only native boundary.
pub use native_value::outline as outline_native_material_functions;
pub mod progress;
pub mod registry;
pub mod renderer_registry;
pub use openpbr::openpbr_fields;
mod compute;
pub use compute::{ComputeProgram,compute_used_bindings};
mod composite;
mod display;
mod propagation;
pub use propagation::PropagationProgram;
mod camera;
pub use camera::CameraProgram;
mod environment;
mod medium;
pub use composite::CompositeProgram;
pub use display::DisplayProgram;
pub use environment::EnvironmentProgram;
mod volume;
pub use medium::{MEDIUM_FP_FLAGS, MEDIUM_VEC4S, MediumProgram};

/// Changes to reflection/bridge metadata invalidate the shared program artifact key.
pub const ABI_VERSION: u32 = 43;
/// Program-visible ABI version, `VORK_ABI` in `import vork`. Bump it when a
/// public declaration a program can use changes incompatibly.
pub const PROGRAM_ABI: u32 = 1;
/// Bundled Slang sources use the same stage compiler as user programs.
pub use vork_scene::libraries::{modules as library_modules, module_source};
/// Kernel-internal Slang modules: imported by bundled programs and kernel shaders,
/// not part of any public library.
pub const KERNEL_MODULES: &[(&str, &str)] = &[("vork_sampling", include_str!("vork_sampling.slang"))];
/// Names beginning `vork_` are compiler plumbing (handle slots, packed input
/// words, stage wrappers). Authored source and the modules it imports must not
/// name them, nor reach them through Slang's reserved `__` intrinsics (such as
/// `__intrinsic_asm`) or `##` token pasting.
fn reject_reserved(source: &str, file: &str) -> Result<()> {
    let bytes = source.as_bytes();
    let line = |at: usize| source[..at].matches('\n').count() + 1;
    let mut i = 0;
    while i < bytes.len() {
        let rest = &source[i..];
        if rest.starts_with("//") {
            i += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            i += rest[2..].find("*/").map_or(rest.len(), |end| end + 4);
        } else if rest.starts_with("##") {
            anyhow::bail!("{file}:{}: ## token pasting is not available in authored programs", line(i));
        } else if bytes[i] == b'"' {
            // Skip the literal, honouring backslash escapes.
            i += 1;
            while i < bytes.len() && bytes[i] != b'"' {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            i = (i + 1).min(bytes.len());
        } else if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(rest.len());
            let name = &rest[..end];
            ensure!(name != "DisplayContext", "display(DisplayContext, Inputs) was removed; use ViewResult view(ViewContext, Inputs); see vork help upgrade");
            // Modules every workspace supplies (write_modules: vork_sampling and the
            // OpenPBR constants) are importable by name; a module name is not plumbing.
            let module = name == "vork_openpbr_constants" || KERNEL_MODULES.iter().any(|(kernel, _)| *kernel == name);
            ensure!((!name.starts_with("vork_") || module) && !name.starts_with("__"),
                "{file}:{}: {name}: names beginning vork_ or __ are reserved for the compiler", line(i));
            i += end;
        } else {
            i += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    Ok(())
}
/// Authored modules a compile read from outside its workspace obey the same rule.
fn reject_reserved_dependencies(work: &Workspace, dependencies: &[PathBuf]) -> Result<()> {
    for module in dependencies.iter().filter(|d| !d.starts_with(&work.0)) {
        reject_reserved(&read_compiler_text(module)?, &module.display().to_string())?;
    }
    Ok(())
}
pub const VERSION: &str = "2026.21";
// WGSL and native code must agree on implicit locals and declared field defaults.
const INITIALIZATION: &str = "-zero-initialize";
/// Kernel functions a compiled program may call; implemented by the scene shader.
pub const MATERIAL_WORD: &str = "vork_material_word";
pub const SAMPLE_IMAGE: &str = "vork_sample_image";
const WAVELENGTH_EFFECT: &str = "vork_effect_wavelength";
/// Image input with no image bound; the kernel's sampler returns its fallback.
pub const NO_IMAGE: u32 = u32::MAX;

pub struct Compiler {
    executable: PathBuf,
    expected: Option<InputIdentity>,
    captured: Option<std::sync::Arc<Mutex<Option<InputIdentity>>>>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Type {
    /// Opaque typed vertex attribute handle; width is float arity.
    Attribute(u32),
    Float,
    Int,
    Uint,
    Bool,
    Vector(u32),
    /// `vork::Image`: one u32 image-table slot.
    Image,
    Geometry,
    FloatGrid,
    VectorGrid,
    Aov,
}
impl Type {
    pub fn words(self) -> usize {
        match self {
            Type::Vector(n) => n as usize,
            _ => 1,
        }
    }
    fn slang(self) -> String {
        match self {
            Type::Float => "float".into(),
            Type::Int => "int".into(),
            Type::Uint => "uint".into(),
            Type::Bool => "bool".into(),
            Type::Vector(n) => format!("float{n}"),
            Type::Geometry => "Geometry".into(),
            Type::Image => "Image".into(),
            Type::FloatGrid => "FloatGrid".into(),
            Type::VectorGrid => "VectorGrid".into(),
            Type::Aov => "Aov".into(),
            Type::Attribute(n) => format!("Attribute<{}>", if n==1 {"float".into()} else {format!("float{n}")}),
        }
    }
    fn wgsl(self) -> String {
        match self {
            Type::Float => "f32".into(),
            Type::Int => "i32".into(),
            Type::Uint | Type::Geometry | Type::Image | Type::FloatGrid | Type::VectorGrid | Type::Aov | Type::Attribute(_) => "u32".into(),
            Type::Bool => "bool".into(),
            Type::Vector(n) => format!("vec{n}<f32>"),
        }
    }
}

/// One member of a kernel struct crossing the boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Field {
    pub name: String,
    pub ty: Type,
    /// Slang initializer expression; None means zero.
    pub default: Option<String>,
    /// Fixed array length (`float2 uv[2]`); None for a single value.
    pub array: Option<u32>,
    /// Closure vectors defaulting to the shading frame (normals, tangents):
    /// left zero by the program, they keep the kernel's value.
    #[serde(default)]
    pub frame: bool,
    /// Kernel plumbing carried in the struct but not readable by programs.
    #[serde(default)]
    pub internal: bool,
}
impl Field {
    pub fn new(name: &str, ty: Type, default: Option<&str>) -> Self {
        Self {
            name: name.into(),
            ty,
            default: default.map(Into::into),
            array: None,
            frame: false,
            internal: false,
        }
    }
    pub fn array(name: &str, ty: Type, len: u32) -> Self {
        Self {
            array: Some(len),
            ..Self::new(name, ty, None)
        }
    }
    fn slang(&self) -> String {
        let len = self.array.map(|n| format!("[{n}]")).unwrap_or_default();
        format!("{} {}{len}", self.ty.slang(), self.name)
    }
    fn wgsl(&self) -> String {
        match self.array {
            Some(n) => format!("{}:array<{},{n}>", self.name, self.ty.wgsl()),
            None => format!("{}:{}", self.name, self.ty.wgsl()),
        }
    }
}

/// Kernel structs visible to programs. Their WGSL twins (same member names and
/// types) are defined by the scene shader; values cross as packed vec4 arrays.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Abi {
    /// Number of named AOVs declared by the installed scene.
    pub aov_count: usize,
    pub context: Vec<Field>,
    pub closure: Vec<Field>,
}

impl Abi {
    /// api-v2-spec §2 ShadingContext (`attr<T>()` is resolved at install, not here).
    pub fn spec_context() -> Vec<Field> {
        let v3 = Type::Vector(3);
        let mut fields = vec![
            Field::new("POSITION", v3, None),
            Field::new("NORMAL", v3, None),
            Field::new("true_normal", v3, None),
            Field::new("incoming", v3, None),
            Field::new("TANGENT", v3, None),
            Field::new("COLOR_0", Type::Vector(4), Some("float4(1.)")),
            Field::new("TEXCOORD_0", Type::Vector(2), None),
            Field::new("TEXCOORD_1", Type::Vector(2), None),
            Field::array("duv_dx", Type::Vector(2), 2),
            Field::array("duv_dy", Type::Vector(2), 2),
            Field::new("time", Type::Float, None),
            Field::new("wavelength", Type::Float, Some("0.")),
            Field::new("object_id", Type::Uint, None),
            Field::new("instance_index", Type::Uint, None),
            Field::new("primitive_id", Type::Uint, None),
            Field::new("front_facing", Type::Bool, None),
            Field::new("tangent_sign", Type::Float, None),
            Field::new("sample_spread", v3, None),
            Field::new("has_tangent", Type::Bool, None),
            Field::new("position_dx", v3, None),
            Field::new("position_dy", v3, None),
        ];
        // Coordinate helpers take the context; the renderer's object table index stays private.
        fields.push(Field { internal: true, ..Field::new("object_index", Type::Uint, None) });
        for name in ["rest_position", "unperturbed_normal"] {
            fields.push(Field::new(name, v3, None));
            fields.push(Field::new(&format!("{name}_dx"), v3, None));
            fields.push(Field::new(&format!("{name}_dy"), v3, None));
        }
        fields.push(Field::new("incoming_dx", v3, None));
        fields.push(Field::new("incoming_dy", v3, None));
        fields
    }
    /// Public declarations from the exact module supplied to shader programs.
    pub fn public_declarations(&self) -> String { declarations::public(&self.module()) }

    /// Length of the packed `array<vec4<f32>,K>` a program returns.
    pub fn closure_vec4s(&self) -> usize {
        vec4s(&self.closure)
    }
    /// Length of the packed context array a program receives.
    pub fn context_vec4s(&self) -> usize {
        vec4s(&self.context)
    }
    /// WGSL twins of ShadingContext and Closure, for validating programs alone.
    pub fn structs_wgsl(&self) -> String {
        let mut s = String::new();
        for (name, fields) in [
            ("ShadingContext", &self.context),
            ("Closure", &self.closure),
        ] {
            let members: Vec<_> = fields.iter().map(Field::wgsl).collect();
            writeln!(s, "struct {name} {{ {} }}", members.join(", ")).unwrap();
        }
        s.push_str(&volume::transport_wgsl());
        s
    }
    /// Full compiler/kernel identity, shared by surface and medium artifact keys.
    pub fn cache_identity(&self) -> Result<String> {
        Ok(format!(
            "{VERSION}\n{ABI_VERSION}\n{INITIALIZATION}\n{}\n{}",
            serde_json::to_string(self)?,
            self.module()
                + &openpbr::with_layer_effects(&self.module(), &self.closure)
                + &openpbr::with_replay(&self.module(), &self.closure)
                + &library_modules()
                    .map(|module| module.source)
                    .chain(KERNEL_MODULES.iter().map(|(_, source)| *source))
                    .collect::<String>()
                + include_str!("trace_types.slang")
                + &renderer_registry::generate(true)
        ))
    }
    /// The Slang `Closure`: every OpenPBR field on every build. Fields absent from `closure` (hair and test ABIs) never
    /// cross into the kernel.
    fn declared_closure(&self) -> Vec<Field> {
        openpbr_fields()
            .into_iter()
            .map(|p| self.closure.iter().find(|f| f.name == p.name).cloned().unwrap_or(p))
            .collect()
    }
    /// The `vork` Slang module every program and library imports.
    pub fn module(&self) -> String {
        self.module_with(
            "",
            false,
            &format!("__target_switch {{ case wgsl: __intrinsic_asm \"{SAMPLE_IMAGE}($0, $1, $2, $3)\"; }}"),
            // Native medium transport (resolve) links its own word reader.
            &format!(
                "#ifdef VORK_NATIVE_MEDIUM\nreturn native_material_word(index);\n#else\n\
                 __target_switch {{ case wgsl: __intrinsic_asm \"{MATERIAL_WORD}($0)\"; }}\n#endif"
            ),
        )
    }
    /// The `vork` module for programs linked into the renderer's native colour
    /// passes: images and input words read the scene material table directly.
    pub fn native_module(&self) -> String {
        let module = self.module_with(
            "import material;\nimport frame;\n#ifdef VORK_CAMERA_ENVIRONMENT\ninternal extern float4 camera_environment_image(uint address,float2 uv);\n#endif\n",
            true,
            &format!("if (slot == {NO_IMAGE}u) {{ return float4(1.); }}\n#ifdef VORK_CAMERA_ENVIRONMENT\n    if ((slot & 0x80000000u) != 0u) return camera_environment_image(slot & 0x7fffffffu, uv);\n#endif\n    return material_image(slot, uv, ddx, ddy);"),
            "return material_words[index];",
        );
        openpbr::with_replay(&module, &self.closure)
    }
    /// Every module a native program compile needs besides the renderer's own:
    /// (name, source), with `vork` in its native form.
    pub fn native_modules(&self) -> Vec<(String, String)> {
        let mut modules = vec![("vork".to_owned(), self.native_module()), ("program_trace".into(), include_str!("trace_types.slang").into())];
        for module in library_modules() {
            modules.push((module.name.to_owned(), module.source.to_owned()));
        }
        modules.extend(KERNEL_MODULES.iter().map(|(name, source)| ((*name).to_owned(), (*source).to_owned())));
        modules.push(("openpbr_core".into(), module_source("std/openpbr_core").unwrap().to_owned()));
        modules.push(("vork_openpbr_constants".into(), renderer_registry::generate(true)));
        modules
    }
    fn module_with(&self, imports: &str, native: bool, sample: &str, word: &str) -> String {
        let aov_slots = self.aov_count.max(1);
        let aov_count = self.aov_count;
        let mut s = format!("module vork;\n{imports}public static const uint VORK_ABI = {PROGRAM_ABI}u;\n");
        let closure = self.declared_closure();
        for (name, fields) in [
            ("ShadingContext", &self.context[..]),
            ("Closure", &closure[..]),
        ] {
            writeln!(s, "public struct {name} {{").unwrap();
            for f in fields {
                let init = f
                    .default
                    .as_deref()
                    .map(|d| format!(" = {d}"))
                    .unwrap_or_default();
                let visibility = if f.internal { "internal" } else { "public" };
                writeln!(s, "    {visibility} {}{init};", f.slang()).unwrap();
            }
            s.push_str("};\n");
        }
        // Stage wrappers unpack the renderer's packed context here, where its
        // internal fields are writable.
        writeln!(s, "public ShadingContext vork_shading_context(float4 a[{}]) {{\n    ShadingContext ctx;", self.context_vec4s()).unwrap();
        for (i, f, at) in slots(&self.context) {
            let value = from_f32_slang(f.ty, &format!("a[{}][{}]", i / 4, i % 4));
            writeln!(s, "    ctx.{}{at} = {value};", f.name).unwrap();
        }
        s.push_str("    return ctx;\n}\n");
        s.push_str("[__AttributeUsage(_AttributeTargets.Var)]\npublic struct InputRangeAttribute { float min; float max; bool min_closed; bool max_closed; };\n");
        s.push_str(&coordinates::module(native));
        s.push_str(&trace::module(native));
        s.push_str(if native {""} else {
            "internal void vork_blend_overflow(uint count, uint capacity) { __target_switch { case wgsl: __intrinsic_asm \"vork_blend_overflow($0, $1)\"; } }\n"
        });
        // Blender's Shader to RGB: what the optional `lit` stage receives.
        s.push_str("public struct Lighting { public float3 color; public float alpha; public float3 direct; };\n");
        s.push_str("public struct MediumContext { public float3 position; public float time; public float3 world_position; public float wavelength = 0.; };\n");
        s.push_str(include_str!("camera_context.slang"));
        s.push_str(include_str!("propagation_context.slang"));
        s.push_str("public struct EnvironmentContext { public float3 origin; public float3 direction; public float time; public bool camera; };\n");
        // Display stage: unassociated scene-linear Rec.709 after the compositor.
        s.push_str("public enum WorkingColorSpace { LinearRec709 = 0, LinearRec2020 = 1, ACEScg = 2, Srgb = 3, DisplayP3 = 4 }; public enum ReferenceDomain { Scene = 0, Display = 1 }; public enum DisplayTarget { Srgb = 0, DisplayP3 = 1, Rec2100Pq = 2, Rec2100Hlg = 3 }; public struct ViewContext { public float3 color; public WorkingColorSpace color_space; public ReferenceDomain reference_domain; public DisplayTarget display; public float peak_luminance; public float reference_white; public uint2 pixel; public uint seed; public float time; };\n");
        // A view's result is a value and its form: absolute XYZ D65 in cd/m², or code values for one display.
        // The engine takes codes for its destination as they are and converts XYZ to the destination itself.
        s.push_str("public struct ViewResult { public float3 value; public bool codes; public DisplayTarget display; };\npublic ViewResult view_from_xyz(float3 xyz) { ViewResult r; r.value = xyz; r.codes = false; r.display = DisplayTarget::Srgb; return r; }\npublic ViewResult view_from_codes(float3 codes, DisplayTarget display) { ViewResult r; r.value = codes; r.codes = true; r.display = display; return r; }\n");
        s.push_str(include_str!("composite_context.slang"));
        s.push_str("public struct ComputeContext { public uint3 id; public uint frame; public float time; public float dt; public uint seed; };\n");
        s.push_str(&hair::module());
        s.push_str(&openpbr::module(&self.closure));
        s.push_str(&volume::module());
        write!(
            s,
            "public struct Aov {{ internal uint slot = 0xffffffffu; }};\n\
             public struct AovOutputs {{ internal float4 values[{aov_slots}]; }};\n\
             public void write_aov(inout AovOutputs outputs, Aov target, float value) {{if(target.slot<{aov_count}u) outputs.values[target.slot]=float4(value,value,value,1.);}}\n\
             public void write_aov(inout AovOutputs outputs, Aov target, float3 value) {{if(target.slot<{aov_count}u) outputs.values[target.slot]=float4(value,1.);}}\n\
             public float4 vork_aov_value(AovOutputs outputs, uint index) {{ return outputs.values[index]; }}\n\
             public struct Geometry {{ internal uint slot = 0xffffffffu; }};\n\
             public struct FloatGrid {{ internal uint slot = 0u; }};\n\
             public struct VectorGrid {{ internal uint slot = 0u; }};\n\
             public struct Image {{ internal uint slot = {NO_IMAGE}; }};\n\
             public Image vork_image(uint slot) {{ Image h; h.slot = slot; return h; }}\n\
             public Aov vork_aov(uint slot) {{ Aov h; h.slot = slot; return h; }}\n\
             public Geometry vork_geometry(uint slot) {{ Geometry h; h.slot = slot; return h; }}\n\
             public uint vork_slot(Geometry h) {{ return h.slot; }}\n\
             public FloatGrid vork_float_grid(uint slot) {{ FloatGrid h; h.slot = slot; return h; }}\n\
             public VectorGrid vork_vector_grid(uint slot) {{ VectorGrid h; h.slot = slot; return h; }}\n\
             public uint vork_slot(Image h) {{ return h.slot; }}\n\
             public uint vork_slot(Aov h) {{ return h.slot; }}\n\
             public uint vork_slot(FloatGrid h) {{ return h.slot; }}\n\
             public uint vork_slot(VectorGrid h) {{ return h.slot; }}\n\
             float4 sample_slot(uint slot, float2 uv, float2 ddx, float2 ddy) {{\n    \
             {sample}\n}}\n\
             // Gradients come from the context's first UV set.\n\
             public float4 sample(Image image, float2 uv, ShadingContext ctx) {{\n    \
             return sample_slot(image.slot, uv, ctx.duv_dx[0], ctx.duv_dy[0]);\n}}\n\
             #ifdef VORK_NATIVE_MEDIUM\ninternal extern uint native_material_word(uint index);\n#endif\n\
             public uint vork_input_word(uint index) {{\n    \
             {word}\n}}\n"
        )
        .unwrap();
        writeln!(s,"public Image vork_image_input(uint base,uint slot) {{return vork_image({});}}",if native {"base+slot"} else {"slot"}).unwrap();
        s.push_str(&attributes::module(native));
        s.push_str("public float4 sample(Image image, float2 uv, DisplacementContext ctx) { return sample_slot(image.slot, uv, float2(0), float2(0)); }\n");
        s.push_str("public float4 sample(Image image, float2 uv, EnvironmentContext ctx) { return sample_slot(image.slot, uv, float2(0), float2(0)); }\n");
        s.push_str("public float4 sample(Image image, float2 uv, ViewContext ctx) { return sample_slot(image.slot, uv, float2(0), float2(0)); }\n");
        // One texel exactly, for a bundled view whose data is a lattice; the display stage clamps the coordinate.
        s.push_str("float4 view_load_slot(uint slot, int2 texel) { __target_switch { case wgsl: __intrinsic_asm \"vork_view_load($0, $1)\"; } }\npublic float4 vork_view_texel(Image image, int2 texel) { return view_load_slot(image.slot, texel); }\n");
        s.push_str("float4 compositor_load_slot(uint slot, int2 pixel) { __target_switch { case wgsl: __intrinsic_asm \"vork_compositor_load($0, $1)\"; } }\npublic float4 compositor_load(Image image, int2 pixel) { return compositor_load_slot(image.slot,pixel); }\npublic float4 compositor_sample(Image image, float2 uv) { return sample_slot(image.slot, uv, float2(0), float2(0)); }\n");
        s.push_str("float2 compositor_size_slot(uint slot) { __target_switch { case wgsl: __intrinsic_asm \"vork_compositor_size($0)\"; } }\npublic float2 compositor_size(Image image) { return compositor_size_slot(image.slot); }\npublic float2 compositor_size() { return compositor_size_slot(0u); }\n");
        s.push_str("public float4 compositor_sample(float2 uv) { return sample_slot(0u, uv, float2(0), float2(0)); }\n");
        s
    }
    /// WGSL pack/unpack helpers the scene shader includes once, after its own
    /// ShadingContext and Closure definitions. Unpacking overwrites `base`
    /// (the kernel's default Closure) with the program's fields.
    pub fn kernel_wgsl(&self) -> String {
        let (c, k) = (vec4s(&self.context), vec4s(&self.closure));
        let mut s = format!(
            "fn vork_pack_context(c:ShadingContext)->array<vec4<f32>,{c}> {{\n    var a:array<vec4<f32>,{c}>;\n"
        );
        for (i, f, at) in slots(&self.context) {
            let value = to_f32_wgsl(f.ty, &format!("c.{}{at}", f.name));
            writeln!(s, "    a[{}][{}]={value};", i / 4, i % 4).unwrap();
        }
        s.push_str("    return a;\n}\n");
        writeln!(
            s,
            "fn vork_unpack_closure(a:array<vec4<f32>,{k}>,base:Closure)->Closure {{\n    var c=base;"
        )
        .unwrap();
        let slots = slots(&self.closure);
        for f in &self.closure {
            let mine: Vec<_> = slots
                .iter()
                .filter(|(_, g, _)| std::ptr::eq(*g, f))
                .collect();
            if f.frame {
                let parts: Vec<_> = mine
                    .iter()
                    .map(|(i, _, _)| format!("a[{}][{}]", i / 4, i % 4))
                    .collect();
                let v = format!("{}({})", f.ty.wgsl(), parts.join(","));
                writeln!(
                    s,
                    "    if any({v}!={}()) {{c.{}={v};}}",
                    f.ty.wgsl(),
                    f.name
                )
                .unwrap();
                continue;
            }
            for (i, f, at) in mine {
                let value = from_f32_wgsl(f.ty, &format!("a[{}][{}]", i / 4, i % 4));
                writeln!(s, "    c.{}{at}={value};", f.name).unwrap();
            }
        }
        s.push_str("    return c;\n}\n");
        s
    }
    fn wrapper(&self, inputs: &InputLayout, tag: &str, lobes: bool) -> String {
        format!(
            "[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}\n{}",
            self.surface_function("export", inputs, tag, lobes)
        )
    }
    /// `vork_surface_<tag>(context vec4s, input word base, out closure vec4s)`:
    /// the packed boundary between a program module and the renderer, which
    /// cannot import both `vork` and its own Closure.
    fn aov_function(&self,linkage:&str,inputs:&InputLayout,tag:&str)->String {
        let aov_slots = self.aov_count.max(1);
        let c=self.context_vec4s();
        let mut s=format!("{linkage} void vork_aov_{tag}(float4 a[{c}],uint base,out float4 r[{aov_slots}]) {{\nShadingContext ctx=vork_shading_context(a);\n");
        s.push_str(&inputs.slang_decode());
        s.push_str(&format!("AovOutputs outputs={{}};\naov(ctx,inputs,outputs);\nfor(uint i=0u;i<{aov_slots}u;i++) r[i]=vork_aov_value(outputs,i);\n}}\n"));
        s
    }
    /// `vork_lit_<tag>(context vec4s, input word base, lighting vec4s, out closure vec4s)`:
    /// the optional `lit` stage over the same context and inputs as `surface`.
    fn lit_function(&self, linkage: &str, inputs: &InputLayout, tag: &str) -> String {
        let (c, k) = (vec4s(&self.context), vec4s(&self.closure));
        let mut s = format!("{linkage} void vork_lit_{tag}(float4 a[{c}], uint base, float4 light[2], out float4 r[{k}]) {{\n    ShadingContext ctx = vork_shading_context(a);\n");
        s.push_str(&inputs.slang_decode());
        s.push_str("    Lighting lighting;\n    lighting.color = light[0].xyz; lighting.alpha = light[0].w; lighting.direct = light[1].xyz;\n    Closure c = lit(ctx, inputs, lighting);\n");
        for (i, f, at) in slots(&self.closure) {
            let value = to_f32_slang(f.ty, &format!("c.{}{at}", f.name));
            writeln!(s, "    r[{}][{}] = {value};", i / 4, i % 4).unwrap();
        }
        s.push_str("}\n");
        s
    }
    fn displacement_function(&self, linkage: &str, inputs: &InputLayout, tag: &str) -> String {
        let mut source = format!("float3 vork_displacement_result<int N>(vector<float,N> v) {{ static_assert(N == 3, \"displace must return float3\"); return float3(v[0],v[1],v[2]); }}\n{linkage} float3 vork_displace_{tag}(float3 position, float3 normal, float time, uint base) {{\nDisplacementContext ctx; ctx.POSITION=position; ctx.NORMAL=normal; ctx.time=time;\n");
        source.push_str(&inputs.slang_decode());
        source.push_str("return vork_displacement_result(displace(ctx,inputs));\n}\n");
        source
    }
    /// With `lobes`, the program returns a Blend: the function also takes the
    /// requested lobe and pick (vork_lobe) and returns (weight, lobe count).
    fn surface_function(&self, linkage: &str, inputs: &InputLayout, tag: &str, lobes: bool) -> String {
        let (c, k) = (vec4s(&self.context), vec4s(&self.closure));
        let lobe_args = if lobes { "uint lobe, float u, " } else { "" };
        let lobe_out = if lobes { ", out float2 lobes" } else { "" };
        let mut s = format!(
            "{linkage} void vork_surface_{tag}(float4 a[{c}], uint base, {lobe_args}out float4 r[{k}]{lobe_out}) {{\n    \
             ShadingContext ctx = vork_shading_context(a);\n"
        );
        s.push_str(&inputs.slang_decode());
        s.push_str(if lobes {
            "    float weight; uint count;\n    Closure c = vork_lobe(surface(ctx, inputs), lobe, u, weight, count);\n    lobes = float2(weight, float(count));\n"
        } else {
            "    Closure c = surface(ctx, inputs);\n"
        });
        for (i, f, at) in slots(&self.closure) {
            let value = to_f32_slang(f.ty, &format!("c.{}{at}", f.name));
            writeln!(s, "    r[{}][{}] = {value};", i / 4, i % 4).unwrap();
        }
        s.push_str("}\n");
        s
    }
}

impl Abi {
    /// Slang statements packing context fields into `float4 a[C]` (zeroed by the
    /// caller); `value(field, element)` is the field's Slang expression, or None
    /// to leave it zero.
    pub fn pack_context_slang(&self, value: impl Fn(&str, usize) -> Option<String>) -> String {
        let mut s = String::new();
        for (i, f, at) in slots(&self.context) {
            let element = at
                .strip_prefix('[')
                .filter(|_| f.array.is_some())
                .and_then(|a| a.split(']').next())
                .map_or(0, |e| e.parse().unwrap());
            let Some(v) = value(&f.name, element) else { continue };
            // The component accessor after any array index.
            let component = match f.array {
                Some(_) => at.splitn(2, ']').nth(1).unwrap_or(""),
                None => &at,
            };
            let value = to_f32_slang(f.ty, &format!("({v}){component}"));
            writeln!(s, "    a[{}][{}] = {value};", i / 4, i % 4).unwrap();
        }
        s
    }
    /// Slang statements writing a program's packed closure `r` over `target`, a
    /// renderer Closure holding the defaults: frame fields only when nonzero, as
    /// kernel_wgsl's vork_unpack_closure.
    pub fn unpack_closure_slang(&self, target: &str) -> String {
        let mut s = String::new();
        let slots = slots(&self.closure);
        for f in &self.closure {
            let mine: Vec<_> = slots
                .iter()
                .filter(|(_, g, _)| std::ptr::eq(*g, f))
                .collect();
            if f.frame {
                let parts: Vec<_> = mine
                    .iter()
                    .map(|(i, _, _)| format!("r[{}][{}]", i / 4, i % 4))
                    .collect();
                let v = format!("{}({})", f.ty.slang(), parts.join(","));
                writeln!(s, "    if (any({v} != {}(0))) {{ {target}.{} = {v}; }}", f.ty.slang(), f.name).unwrap();
                continue;
            }
            for (i, f, at) in mine {
                let value = from_f32_slang(f.ty, &format!("r[{}][{}]", i / 4, i % 4));
                writeln!(s, "    {target}.{}{at} = {value};", f.name).unwrap();
            }
        }
        s
    }
}

/// Flat f32 slot index, field, and accessor suffix (`[k][j]`) for each scalar.
fn slots(fields: &[Field]) -> Vec<(usize, &Field, String)> {
    let mut out = Vec::new();
    for f in fields {
        let elements: Vec<String> = match f.array {
            Some(n) => (0..n).map(|k| format!("[{k}]")).collect(),
            None => vec![String::new()],
        };
        for e in elements {
            match f.ty {
                Type::Vector(n) => {
                    (0..n).for_each(|j| out.push((out.len(), f, format!("{e}[{j}]"))))
                }
                _ => out.push((out.len(), f, e)),
            }
        }
    }
    out
}
fn vec4s(fields: &[Field]) -> usize {
    slots(fields).len().div_ceil(4).max(1)
}
fn to_f32_slang(ty: Type, v: &str) -> String {
    match ty {
        Type::Float | Type::Vector(_) => v.into(),
        Type::Int | Type::Uint => format!("asfloat({v})"),
        Type::Geometry | Type::Image | Type::FloatGrid | Type::VectorGrid | Type::Aov => format!("asfloat(vork_slot({v}))"),
        Type::Attribute(_) => format!("asfloat(vork_slot({v}))"),
        Type::Bool => format!("({v} ? 1.0 : 0.0)"),
    }
}
fn from_f32_slang(ty: Type, v: &str) -> String {
    match ty {
        Type::Float | Type::Vector(_) => v.into(),
        Type::Int => format!("asint({v})"),
        Type::Uint => format!("asuint({v})"),
        Type::Geometry | Type::Image | Type::Aov | Type::FloatGrid | Type::VectorGrid | Type::Attribute(_) => handle_slang(ty, &format!("asuint({v})")),
        Type::Bool => format!("{v} != 0.0"),
    }
}
/// An opaque resource handle from its u32 slot, through the reserved constructors.
fn handle_slang(ty: Type, slot: &str) -> String {
    match ty {
        Type::Geometry => format!("vork_geometry({slot})"),
        Type::Image => format!("vork_image({slot})"),
        Type::Aov => format!("vork_aov({slot})"),
        Type::FloatGrid => format!("vork_float_grid({slot})"),
        Type::VectorGrid => format!("vork_vector_grid({slot})"),
        Type::Attribute(n) => format!("vork_attribute_handle<{}>({slot})", if n == 1 { "float".into() } else { format!("float{n}") }),
        _ => panic!("{ty:?} is not a resource handle"),
    }
}
fn to_f32_wgsl(ty: Type, v: &str) -> String {
    match ty {
        Type::Float | Type::Vector(_) => v.into(),
        Type::Int | Type::Uint | Type::Geometry | Type::Image | Type::FloatGrid | Type::VectorGrid | Type::Aov | Type::Attribute(_) => format!("bitcast<f32>({v})"),
        Type::Bool => format!("select(0.,1.,{v})"),
    }
}
fn from_f32_wgsl(ty: Type, v: &str) -> String {
    match ty {
        Type::Float | Type::Vector(_) => v.into(),
        Type::Bool => format!("{v}!=0."),
        _ => format!("bitcast<{}>({v})", ty.wgsl()),
    }
}

/// Conservative effects of an evaluated program closure. These are compiler
/// facts consumed by scene installation, independent of input names; inputs
/// are unknown, or an instance's material words (`EffectProbe::capabilities`).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ProgramCapabilities {
    /// `surface` returns a curves-only fibre closure.
    pub hair: bool,
    /// A fibre closure that may select physical (Chiang) scattering. Unbound,
    /// every hair program that can select it may; installation reads the
    /// per-instance `EffectProbe::capabilities` value.
    pub physical_hair: bool,
    pub opacity: bool,
    pub transmission: bool,
    pub subsurface: bool,
    /// The closure can scatter through a solid interior (thin sheets cannot).
    pub spatial_subsurface: bool,
    pub anisotropy: bool,
    pub coat_anisotropy: bool,
    pub film: bool,
    pub coat: bool,
    pub fuzz: bool,
    pub tint: bool,
    /// Any returned closure can have a non-unit dielectric specular weight.
    pub weighted_ior: bool,
    /// `surface` returns a Blend: independently shaded lobes.
    #[serde(default)]
    pub lobes: bool,
    /// Proven upper bound on independently shaded lobes; None means unbounded.
    #[serde(default)]
    pub blend_lobes: Option<u32>,
    /// The program declares the `lit` stage (Blender's Shader to RGB).
    #[serde(default)]
    pub lit: bool,
    /// The `lit` stage's closure may reflect light; otherwise it only emits.
    #[serde(default)]
    pub lit_reflects: bool,
    /// The material supplies an object-local displacement offset.
    pub displacement: bool,
    #[serde(default)]
    pub emission: bool,
    pub dependencies: ProgramDependencies,
}

/// Runtime inputs of a compiled material. False means proven absent; unknown
/// resource effects retain their dependency. These are facts, not another cache.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct ProgramDependencies {
    /// Actual sample-time dependence, including resource addresses/coordinates.
    /// Reading a separately tracked camera, transform or image is not a clock read.
    pub time: bool,
    pub camera: bool,
    pub object_transforms: bool,
    pub attributes: bool,
    pub images: bool,
    /// A reachable geometric query reads the evaluated scene snapshot.
    #[serde(default)]
    pub trace: bool,
}

impl ProgramDependencies {
    pub fn include(&mut self, other: Self) {
        self.time |= other.time;
        self.camera |= other.camera;
        self.object_transforms |= other.object_transforms;
        self.attributes |= other.attributes;
        self.images |= other.images;
        self.trace |= other.trace;
    }
}

/// A compiled `surface` program, ready to concatenate into the scene shader.
#[derive(Debug, Serialize, Deserialize)]
pub struct Program {
    pub capabilities: ProgramCapabilities,
    /// Attribute inputs reachable from displacement, independent of surface reads.
    pub displacement_attributes: Vec<String>,
    pub displacement_images: Vec<String>,
    pub displacement_uses_normal: bool,
    pub displacement_uses_time: bool,
    /// Self-contained WGSL; every identifier is tagged, no bindings, no entries.
    pub wgsl: String,
    /// `fn(ctx: array<vec4<f32>,C>, base: u32, r: ptr<function, array<vec4<f32>,K>>)`.
    pub surface: String,
    /// Optional named AOV write bridge, returning the scene-sized float4 array.
    pub aov: Option<String>,
    /// The program as a Slang module for native colour passes: its source and a
    /// public `vork_surface_<tag>` with the same packed signature as `surface`.
    pub slang: String,
    /// Directories its relative imports resolve from.
    pub include: Vec<PathBuf>,
    pub inputs: InputLayout,
    /// Every Slang file read besides the program source (imports); callers
    /// caching a Program must revalidate these.
    pub dependencies: Vec<PathBuf>,
    /// Install-local bytes from the identity snapshot, refreshed even on a
    /// proof-cache hit. Native linking must not race ahead of those proofs.
    #[serde(skip)]
    pub install_identity: Option<InputIdentity>,
    pub warnings: String,
    /// Authored stages that this closure cannot execute, independent of compiler diagnostics.
    pub dropped_stages: Vec<String>,
    /// The closure-effect probe (`Compiler::surface_effects`) as WGSL.
    pub effects: String,
}

impl Program {
    /// Refresh diagnostic locations when reusing a compiler-token-identical proof.
    /// The owned wrapper follows the last marker, even if source comments quote it.
    pub fn set_native_source(&mut self, source: &str, origin: &Path) -> Result<()> {
        let generated = self.slang.rsplit_once("\n#line 1 \"<vork program wrapper>\"\n")
            .context("program is missing its generated native wrapper")?.1;
        self.slang = program_text(source, origin) + generated;
        self.include = include_paths(origin);
        Ok(())
    }
    pub fn needs_spectral(&self, _abi: &Abi, words: &[Option<u32>]) -> Result<bool> {
        Ok(self.effect_probe()?.needs_spectral(words))
    }
    /// The parsed effect probe, to evaluate per material instance.
    pub fn effect_probe(&self) -> Result<EffectProbe> {
        EffectProbe::parse(&self.effects, self.capabilities.lit)
    }
}

/// The pinned Slang compiler is missing, cannot run, or has the wrong version.
#[derive(Debug)]
pub struct CompilerUnavailable(pub String);
impl std::fmt::Display for CompilerUnavailable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CompilerUnavailable {}


/// Where admissions are kept between processes: the identities ([`Compiler::toolchain`]) of
/// toolchains that reported the pinned version when run. The renderer keeps them in its
/// disk store beside the receipts.
pub trait AdmissionLedger {
    /// The one place this ledger's store keeps `toolchain`'s admission, whoever asks.
    fn entry(&self, toolchain: blake3::Hash) -> Option<PathBuf>;
    fn admitted(&self, entry: &Path, toolchain: blake3::Hash) -> bool;
    fn record(&self, entry: &Path, toolchain: blake3::Hash);
}

/// What the environment tells the dynamic loader when the compiler starts.
struct Loader {
    /// Path lists it searches for libraries, by variable.
    searched: Vec<(String, std::ffi::OsString)>,
    /// Anything else addressed to it: injected libraries, audit modules, switches. These
    /// are not modelled, so a toolchain started under one has no identity.
    other: Vec<(String, std::ffi::OsString)>,
    /// The directory the compiler is started in.
    started_in: PathBuf,
}
impl Loader {
    /// Every variable the platform's loader reads is one of its search lists or `other`:
    /// a variable this code does not know cannot be silently ignored.
    fn environment() -> Self {
        let (prefix, lists): (&str, &[&str]) = if cfg!(target_os = "macos") {
            ("DYLD_", &["DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH", "DYLD_VERSIONED_LIBRARY_PATH"])
        } else if cfg!(unix) { ("LD_", &["LD_LIBRARY_PATH"]) } else { ("", &[]) };
        let mut loader = Self { searched: Vec::new(), other: Vec::new(), started_in: std::env::current_dir().unwrap_or_default() };
        if prefix.is_empty() { return loader; }
        let mut variables: Vec<_> = std::env::vars_os()
            .filter_map(|(name, value)| Some((name.to_str()?.to_owned(), value)))
            .filter(|(name, value)| name.starts_with(prefix) && !value.is_empty()).collect();
        variables.sort();
        for (name, value) in variables {
            if lists.contains(&name.as_str()) { loader.searched.push((name, value)); } else { loader.other.push((name, value)); }
        }
        loader
    }
}

/// This process's verdict on one selected executable that passed.
struct Checked {
    /// The toolchain vouched for: the files that were run, or that a ledger held.
    toolchain: Option<blake3::Hash>,
    /// The ledger entries known to hold it.
    published: std::collections::BTreeSet<PathBuf>,
}
impl Checked {
    /// Make sure `ledger`'s store holds the admission; once per store per process.
    fn publish(&mut self, ledger: Option<&dyn AdmissionLedger>) {
        let (Some(ledger), Some(toolchain)) = (ledger, self.toolchain) else { return };
        let Some(entry) = ledger.entry(toolchain) else { return };
        if self.published.contains(&entry) { return; }
        if !ledger.admitted(&entry, toolchain) { ledger.record(&entry, toolchain); }
        self.published.insert(entry);
    }
}

/// Evidence that the selected toolchain is the pinned compiler: it reported [`VERSION`]
/// when run, in this process or as these same files in an earlier one. Only
/// [`Compiler::admit`] makes one.
pub struct Admitted<'a> {
    pub(crate) compiler: &'a Compiler,
}
impl Admitted<'_> {
    /// The version the admitted toolchain reported.
    pub fn version(&self) -> &'static str {
        VERSION
    }
}

fn compiler_start_advice(macos: bool) -> &'static str {
    if macos {
        "check the loader diagnostic and macOS requirement; if the compiler requires a newer macOS, update macOS or use a compatible vork build. VORK_SLANGC must select the pinned, OS-compatible toolchain"
    } else {
        "check the executable and its runtime libraries; install the pinned toolchain or set VORK_SLANGC"
    }
}

impl Compiler {
    /// Explicit override, installed companion toolchain, then developer PATH.
    pub fn discover() -> Self {
        let executable = std::env::var_os("VORK_SLANGC")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe()
                    .ok()?
                    .canonicalize().ok()?
                    .parent()
                    .map(|p| {
                        p.join("slang/bin")
                            .join(format!("slangc{}", std::env::consts::EXE_SUFFIX))
                    })
                    .filter(|p| p.is_file())
            })
            .unwrap_or_else(|| PathBuf::from("slangc"));
        Self::new(executable)
    }
    pub fn new(executable: impl Into<PathBuf>) -> Self {
        let executable=executable.into();
        // The executable directory is an implicit Slang module search path.
        // An exact-input receipt must not survive selection of another compiler
        // through PATH, even when both report the pinned version.
        let name=if cfg!(windows) && executable.extension().is_none() {executable.with_extension("exe")} else {executable.clone()};
        let resolved=if executable.components().count()==1 && !executable.is_absolute() {
            std::env::var_os("PATH").and_then(|paths| std::env::split_paths(&paths)
                .map(|directory|directory.join(&name))
                .find(|path|path.is_file()))
        } else {Some(executable.clone())};
        let resolved=resolved.map(|path|std::env::current_dir().map_or(path.clone(),|cwd|cwd.join(path)));
        let executable=resolved.and_then(|path|if cfg!(windows) {Some(path)} else {path.canonicalize().ok()}).unwrap_or(executable);
        Self {
            executable,
            expected: None,
            captured: None,
        }
    }
    /// The pinned version, once this process has checked the selected toolchain by running it.
    pub fn version(&self) -> Result<String> {
        self.checked(None)
    }

    /// Admit the selected toolchain as a value. A cache key that names the compiler is built
    /// from this, so a decision read from a cache and a fresh compile refuse the same compilers.
    ///
    /// The toolchain is admitted by what it is ([`Self::toolchain`]): one whose identity the
    /// ledger already holds passed the version check as these same files and is not run again;
    /// any other is run, and recorded when it reports the pinned version. A replaced, edited
    /// or missing compiler has another identity (or none) and takes the version check.
    pub fn admit(&self, ledger: &dyn AdmissionLedger) -> Result<Admitted<'_>> {
        self.checked(Some(ledger))?;
        Ok(Admitted { compiler: self })
    }

    fn checked(&self, ledger: Option<&dyn AdmissionLedger>) -> Result<String> {
        // Compiler instances are short-lived and used concurrently during install.
        // Keep both success and failure for each selected executable for this process.
        static CHECKS: OnceLock<Mutex<BTreeMap<PathBuf, Result<Checked, String>>>> = OnceLock::new();
        let mut checks = CHECKS.get_or_init(Mutex::default).lock().unwrap();
        let check = checks
            .entry(self.executable.clone())
            .or_insert_with(|| self.check(ledger, &Loader::environment()).map_err(|e| format!("{e:#}")));
        match check {
            Ok(checked) => {
                // A check made with another ledger, or none, is published to this one.
                checked.publish(ledger);
                Ok(VERSION.to_owned())
            }
            Err(reason) => Err(anyhow::Error::new(CompilerUnavailable(reason.clone()))),
        }
    }

    /// One check of the selected toolchain, outside the process memo.
    fn check(&self, ledger: Option<&dyn AdmissionLedger>, loader: &Loader) -> Result<Checked> {
        let before = self.toolchain_under(loader).ok();
        // An admission is taken from the ledger only for files that are the same when
        // looked at again; files changing under the look take the run below.
        if let (Some(ledger), Some(toolchain)) = (ledger, before)
            && let Some(entry) = ledger.entry(toolchain)
            && ledger.admitted(&entry, toolchain)
            && self.toolchain_under(loader).ok() == before {
            return Ok(Checked { toolchain: before, published: [entry].into() });
        }
        self.probe_version()?;
        // The verdict vouches for an identity only when the same files were there before
        // and after the run. Otherwise (they could not all be observed, or the directories
        // changed meanwhile, which a shared `bin` does for unrelated reasons) the toolchain
        // passed for this process on the strength of the run alone: nothing is recorded and
        // the next process runs it again.
        let stable = before.filter(|toolchain| self.toolchain_under(loader).ok() == Some(*toolchain));
        let mut checked = Checked { toolchain: stable, published: Default::default() };
        checked.publish(ledger);
        Ok(checked)
    }

    /// What the selected toolchain is, for the pinned version: the executable's path, size,
    /// permissions, modification time and content, and Slang's own files in its directory
    /// and in the `lib` directory next to it. Slang's files are the ones named `slang*` or
    /// `libslang*` (its tools, the compiler library that carries the version, the libraries
    /// and standard-module directory it loads at run time), each by name, size, permissions
    /// and modification time. Other files beside the compiler, in a shared `bin` say, are
    /// not part of it. The directories the environment tells the loader to search
    /// (`LD_LIBRARY_PATH`, with glibc's `glibc-hwcaps` subdirectories; the `DYLD_*_LIBRARY_PATH` lists on macOS) are
    /// observed the same way, in order, since a Slang library there can be the one loaded.
    /// A search list the loader would interpret rather than read as directories, or any
    /// other variable addressed to the loader (an injected library, an audit module), leaves
    /// no identity. On Windows the loader takes a library beside the executable first. An error when the executable cannot be read or
    /// one of Slang's files cannot be observed: such a toolchain has no identity and is
    /// checked by running it.
    pub fn toolchain(&self) -> Result<blake3::Hash> {
        self.toolchain_under(&Loader::environment())
    }

    /// [`Self::toolchain`] under a given loader environment.
    fn toolchain_under(&self, loader: &Loader) -> Result<blake3::Hash> {
        fn part(hash: &mut blake3::Hasher, bytes: &[u8]) {
            hash.update(&(bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        }
        fn state(hash: &mut blake3::Hasher, meta: &fs::Metadata) {
            hash.update(&meta.len().to_le_bytes());
            // Whether it may be read and run is part of what it is: the same bytes without
            // their execute bits cannot be the compiler.
            // Its owner and group decide which of those bits apply to this process.
            #[cfg(unix)]
            {
                use std::os::unix::fs::{MetadataExt, PermissionsExt};
                hash.update(&meta.permissions().mode().to_le_bytes());
                hash.update(&meta.uid().to_le_bytes());
                hash.update(&meta.gid().to_le_bytes());
            }
            #[cfg(not(unix))]
            hash.update(&[u8::from(meta.permissions().readonly())]);
            let modified = meta.modified().ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(u128::MAX, |since| since.as_nanos());
            hash.update(&modified.to_le_bytes());
        }
        fn slang_owned(name: &std::ffi::OsStr) -> bool {
            let name = name.to_string_lossy().to_ascii_lowercase();
            name.strip_prefix("lib").unwrap_or(&name).starts_with("slang")
        }
        fn children(directory: &Path) -> Result<Vec<PathBuf>> {
            let mut paths = fs::read_dir(directory)
                .and_then(|entries| entries.map(|entry| entry.map(|entry| entry.path())).collect::<std::io::Result<Vec<_>>>())
                .with_context(|| format!("list the Slang toolchain directory {}", directory.display()))?;
            paths.sort();
            Ok(paths)
        }
        /// One of Slang's files, or everything under one of its directories.
        fn observe(hash: &mut blake3::Hasher, path: &Path) -> Result<()> {
            part(hash, path.file_name().unwrap_or_default().as_encoded_bytes());
            // Through links: a library is usually a link to its versioned file.
            match fs::metadata(path) {
                Ok(meta) if meta.is_dir() => {
                    let paths = children(path)?;
                    hash.update(&[2]);
                    hash.update(&(paths.len() as u64).to_le_bytes());
                    for path in paths { observe(hash, &path)?; }
                }
                Ok(meta) => { hash.update(&[1]); state(hash, &meta); }
                // A link to nothing is a complete observation; anything else is not.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => { hash.update(&[0]); }
                Err(error) => return Err(error).with_context(|| format!("examine the Slang toolchain file {}", path.display())),
            }
            Ok(())
        }
        let mut hash = blake3::Hasher::new();
        part(&mut hash, b"slang-toolchain-v5");
        part(&mut hash, VERSION.as_bytes());
        part(&mut hash, self.executable.as_os_str().as_encoded_bytes());
        let file = fs::File::open(&self.executable)
            .with_context(|| format!("read Slang at {}", self.executable.display()))?;
        state(&mut hash, &file.metadata()?);
        let mut content = blake3::Hasher::new();
        content.update_reader(file)?;
        hash.update(content.finalize().as_bytes());
        /// Slang's files directly in one directory the loader may take them from.
        fn directory(hash: &mut blake3::Hasher, directory: &Path) -> Result<()> {
            part(hash, directory.as_os_str().as_encoded_bytes());
            // Only a directory that is not there holds nothing; one that cannot be listed
            // leaves the toolchain without an identity.
            if matches!(fs::metadata(directory), Err(error) if error.kind() == std::io::ErrorKind::NotFound) {
                hash.update(&u64::MAX.to_le_bytes());
                return Ok(());
            }
            let owned: Vec<_> = children(directory)?.into_iter()
                .filter(|path| path.file_name().is_some_and(slang_owned)).collect();
            hash.update(&(owned.len() as u64).to_le_bytes());
            for path in owned { observe(hash, &path)?; }
            // glibc prefers a library in a hardware-capability subdirectory of a searched one.
            let capabilities = directory.join("glibc-hwcaps");
            match fs::metadata(&capabilities) {
                Ok(meta) if meta.is_dir() => under(hash, &capabilities)?,
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error).with_context(|| format!("examine {}", capabilities.display())),
            }
            Ok(())
        }
        /// Slang's files anywhere under a directory.
        fn under(hash: &mut blake3::Hasher, directory: &Path) -> Result<()> {
            for path in children(directory)? {
                if fs::metadata(&path).is_ok_and(|meta| meta.is_dir()) {
                    part(hash, path.file_name().unwrap_or_default().as_encoded_bytes());
                    under(hash, &path)?;
                } else if path.file_name().is_some_and(slang_owned) {
                    observe(hash, &path)?;
                }
            }
            Ok(())
        }
        let beside = self.executable.parent().filter(|p| !p.as_os_str().is_empty());
        let libraries = beside.and_then(Path::parent).map(|p| p.join("lib"));
        for path in beside.map(Path::to_path_buf).into_iter().chain(libraries) {
            directory(&mut hash, &path)?;
        }
        // The loader's own environment is followed only where it names directories. A list
        // it would interpret (an expansion like `$ORIGIN`, a `;` separator) or a library it
        // is told to inject cannot be observed faithfully: no identity, so the compiler is run.
        for (name, value) in &loader.other {
            ensure!(value.is_empty(), "{name} changes how the compiler's libraries load; its toolchain has no identity");
        }
        for (name, value) in &loader.searched {
            part(&mut hash, name.as_bytes());
            part(&mut hash, value.as_encoded_bytes());
            if value.is_empty() { continue; }
            ensure!(!value.to_string_lossy().contains(['$', ';']), "{name} is a loader expression; the compiler's toolchain has no identity");
            for path in std::env::split_paths(value) {
                // An empty or relative entry is the directory the compiler is started in:
                // this process's, for the version check. (A compile starts it in a fresh
                // workspace that holds only the modules written there.)
                directory(&mut hash, &loader.started_in.join(path))?;
            }
        }
        Ok(hash.finalize())
    }

    fn probe_version(&self) -> Result<String> {
        let output = compiler_output(Command::new(&self.executable).arg("-version"))
            .with_context(|| format!("cannot start Slang at {}; {}", self.executable.display(), compiler_start_advice(cfg!(target_os = "macos"))))?;
        ensure!(output.status.success(), "Slang at {} could not run ({}); {}\n{}{}",
            self.executable.display(), output.status, compiler_start_advice(cfg!(target_os = "macos")),
            String::from_utf8_lossy(&output.stderr), String::from_utf8_lossy(&output.stdout));
        // slangc prints its version on stderr.
        let version = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        ensure!(
            version == VERSION,
            "Slang {VERSION} required, found {version:?}; select the pinned toolchain with VORK_SLANGC"
        );
        Ok(version)
    }

    /// Compile renderer shader modules to one native shader module. `modules` are
    /// `(name, source)` pairs written as `<name>.slang` into a fresh workspace; `main`
    /// names the module declaring `entries`, `(entry, stage)`. `args` carry the
    /// target and its flags; the output file's bytes are returned.
    pub fn compile_modules(
        &self,
        modules: &[(&str, &str)],
        main: &str,
        entries: &[(&str, &str)],
        args: &[&str],
    ) -> Result<Vec<u8>> {
        self.version()?;
        let work = Workspace::new()?;
        for (name, source) in modules {
            let path = work.0.join(format!("{name}.slang"));
            fs::create_dir_all(path.parent().unwrap())?;
            fs::write(path, source)?;
        }
        let output = work.0.join("out");
        let _progress = progress::Job::start(false).named(main);
        let mut command = Command::new(&self.executable);
        command
            .arg(work.0.join(format!("{main}.slang")))
            .arg(INITIALIZATION)
            .current_dir(&work.0)
            .arg("-I")
            .arg(&work.0)
            .args(args);
        for (entry, stage) in entries {
            command.args(["-entry", entry, "-stage", stage]);
        }
        let metal = args.windows(2).any(|pair| pair == ["-target", "metal"]);
        // Metal's system shader cache keys compiled libraries by source text.
        // `#line` directives would embed this per-process workspace path and
        // miss that cache in every new process.
        if metal && !args.contains(&"-line-directive-mode") {
            command.args(["-line-directive-mode", "none"]);
        }
        let depfile = work.0.join("out.d");
        command.arg("-depfile").arg(&depfile).arg("-o").arg(&output);
        let (result, snapshot) = if self.expected.is_some() || self.captured.is_some() {
            let (output, snapshot) = identity::capture_command(&work, &mut command, &depfile)?;
            (output, Some(snapshot))
        } else { (compiler_output(&mut command).context("run Slang compiler")?, None) };
        ensure!(
            result.status.success(),
            "slangc {}: {}",
            result.status,
            String::from_utf8_lossy(&result.stderr)
        );
        // Native-only branches (e.g. VORK_NATIVE_MEDIUM) may import authored modules the WGSL probe never saw.
        let dependencies = snapshot.as_ref().map(|s| s.dependencies())
            .unwrap_or(depfile_paths(&read_compiler_text(&depfile)?));
        reject_reserved_dependencies(&work, &dependencies)?;
        if let Some(snapshot) = &snapshot {
            identity::check_imports(self, &work, snapshot)?;
        }
        let mut code=fs::read(output)?;
        if metal {metal_source::validate(std::str::from_utf8(&code)?)?;}
        // Slang emits MSL source; its fp-mode flag cannot configure the later
        // Metal compiler invocation in a passthrough API. Carry the precise
        // contract as standard source directives, independent of shader names.
        if metal
            && args.windows(2).any(|pair|matches!(pair[0],"-fp-mode"|"-floating-point-mode") && pair[1]=="precise") {
            let mut source=b"#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n".to_vec();
            source.append(&mut code);code=source;
        }
        Ok(code)
    }

    /// Compile one program. `source` is the module text, `origin` the path used
    /// in diagnostics (its directory resolves relative imports). Bundled modules
    /// are embedded and installed in every workspace. `tag` makes identifiers unique
    /// within a scene shader; use a content hash so the result is cacheable.
    pub fn compile_surface(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
    ) -> Result<Program> {
        let mut ignored_stages = Vec::new();
        let mut is_hair = false;
        let mut capabilities = ProgramCapabilities::default();
        let capacity = std::cell::Cell::new(openpbr::BLEND_LOBES);
        let mut effects = String::new();
        let compiled = self.compile_with_capacity(abi, source, origin, tag, Some(&capacity), false, false, false, |inputs, declarations| {
            ensure!(declarations.contains("compute") || inputs.fields.iter().all(|field|field.ty!=Type::Geometry),"Geometry inputs require a compute program; bind collider geometry on the geometry producer");
            (capabilities, effects) = self.surface_effects(abi, source, origin, declarations.contains("lit"), inputs)?;
            let needed = capabilities.blend_lobes.unwrap_or(openpbr::BLEND_LOBES).max(1);
            capacity.set(needed.max(openpbr::BLEND_LOBES));
            capabilities.displacement = declarations.contains("displace");
            is_hair = capabilities.hair;
            let mut wrapper = if is_hair {
                let hair_abi = Abi { aov_count: abi.aov_count,
                    context: abi.context.clone(),
                    closure: hair_fields(),
                };
                hair_abi.wrapper(inputs, tag, false).replace(
                    "Closure c = surface(ctx, inputs);",
                    "Hair c = surface(ctx, inputs);",
                )
            } else {
                // The effect slice also decides the wrapper: Closure or Blend.
                abi.wrapper(inputs, tag, capabilities.lobes)
            };
            if capabilities.lobes && capabilities.blend_lobes.is_none() {
                wrapper = checked_blend_wrapper(wrapper, capacity.get());
            }
            if is_hair && capabilities.lit { ignored_stages.push("lit".into()); capabilities.lit=false; }
            if declarations.contains("aov") { wrapper.push_str(&abi.aov_function("export", inputs, tag)); }
            if is_hair && capabilities.displacement { ignored_stages.push("displace".into()); capabilities.displacement=false; }
            if capabilities.displacement { wrapper.push_str(&abi.displacement_function("export", inputs, tag)); }
            if capabilities.lit { wrapper.push_str(&abi.lit_function("export", inputs, tag)); }
            Ok(wrapper)
        })?;
        let has_aov=compiled.declarations.contains("aov");
        let inputs = compiled.inputs;
        let pass = compiled.output;
        let wgsl = pass.wgsl;
        let surface = format!("vork_surface_{tag}_0_{tag}");
        ensure!(
            wgsl.contains(&format!("fn {surface}(")),
            "Slang did not emit {surface}"
        );
        let native_abi = if is_hair { Abi { aov_count: abi.aov_count, context: abi.context.clone(), closure: hair_fields() } } else { abi.clone() };
        let function = native_abi.surface_function("public", &inputs, tag, capabilities.lobes);
        let function = if capabilities.lobes {
            // Path compilation needs an outline boundary to limit optimizer
            // growth. Raster must inline replay: across this function boundary
            // NVIDIA hangs on divergent reused blends, and Metal faults in
            // control fragments and hangs in material evaluation. The Metal
            // mechanism is unproven: removing this attribute flips both, and
            // API validation reports nothing.
            let wrapper = function.replace(
                "Closure c = vork_lobe(surface(ctx, inputs), lobe, u, weight, count);",
                &format!("vork_begin_blend(0xffffffffu);\n    let selected=vork_lobe(surface(ctx, inputs), lobe, u, weight, count);\n    vork_begin_blend(selected.leaf);\n    [loop] while(!vork_blend_done()) {{surface(ctx, inputs);vork_continue_blend();}}\n    weight=vork_blend_weight();\n    Closure c=vork_replay_closure(selected.geometry_opacity);"),
            );
            outline_native_material_functions(&format!("#ifdef VORK_PATH_SLANG_OUTLINE\n[noinline]\n#endif\n{wrapper}"))
        } else { function };
        let function = if is_hair { function.replace("Closure c = surface(ctx, inputs);", "Hair c = surface(ctx, inputs);") } else { function };
        let mut slang = program_text(source, origin) + &function;
        if has_aov { slang.push_str(&abi.aov_function("public", &inputs, tag)); }
        if capabilities.displacement { slang.push_str(&abi.displacement_function("public", &inputs, tag)); }
        if capabilities.lit { slang.push_str(&abi.lit_function("public", &inputs, tag)); }
        let resources = if capabilities.displacement { self.displacement_resources(abi, source, origin, &inputs)? } else {Vec::new()};
        let displacement_attributes: Vec<String>=resources.iter().filter(|name|inputs.fields.iter().any(|f|&f.name==*name && matches!(f.ty,Type::Attribute(_)))).cloned().collect();
        let displacement_images: Vec<String>=resources.iter().filter(|name|inputs.fields.iter().any(|f|&f.name==*name && f.ty==Type::Image)).cloned().collect();
        let displacement_uses_normal=capabilities.displacement && effects::displacement_depends_on(&wgsl,&format!("vork_displace_{tag}_0_{tag}"),1)?;
        let displacement_uses_time=capabilities.displacement && effects::displacement_depends_on(&wgsl,&format!("vork_displace_{tag}_0_{tag}"),2)?;
        for (present, entry) in [
            (capabilities.displacement, format!("vork_displace_{tag}_0_{tag}")),
            (has_aov, format!("vork_aov_{tag}_0_{tag}")),
            (capabilities.lit, format!("vork_lit_{tag}_0_{tag}")),
        ] {
            ensure!(!present || !effects::calls_trace(&wgsl, &entry, abi)?, "trace is supported only in surface; displacement, lit and AOV queries are not qualified");
        }
        capabilities.dependencies = effects::program_dependencies(&wgsl, &surface, abi)?;
        if capabilities.lit {
            capabilities.dependencies.include(effects::program_dependencies(&wgsl, &format!("vork_lit_{tag}_0_{tag}"), abi)?);
        }
        if capabilities.dependencies.trace {
            effects::require_bounded_query_caller(&wgsl, &surface, abi)?;
            if capabilities.lit {
                effects::require_bounded_query_caller(&wgsl, &format!("vork_lit_{tag}_0_{tag}"), abi)?;
            }
        }
        capabilities.dependencies.time |= displacement_uses_time;
        capabilities.dependencies.attributes |= !displacement_attributes.is_empty();
        capabilities.dependencies.images |= !displacement_images.is_empty();
        Ok(Program {
            displacement_uses_time,
            displacement_attributes,
            displacement_images,
            displacement_uses_normal,
            capabilities,
            wgsl,
            surface,
            aov: has_aov.then(|| format!("vork_aov_{tag}_0_{tag}")),
            slang,
            include: include_paths(origin),
            inputs,
            dependencies: pass.dependencies,
            install_identity: None,
            warnings: pass.diagnostics,
            dropped_stages: ignored_stages,
            effects,
        })
    }

    /// The closure-effect slice of `surface` with unknown context and inputs.
    /// A Blend's effects are those of any lobe; effect 3 is whether it is one.
    fn displacement_resources(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        inputs: &InputLayout,
    ) -> Result<Vec<String>> {
        let fields: Vec<_> = inputs
            .fields
            .iter()
            .filter(|f| matches!(f.ty, Type::Attribute(_) | Type::Image))
            .collect();
        if fields.is_empty() {
            return Ok(Vec::new());
        }
        let work = Workspace::new()?;
        write_modules(&work, abi, source)?;
        let mut includes = vec![work.0.clone()];
        includes.extend(include_paths(origin));
        let mut probe = program_text(source, origin)
            + "ConstantBuffer<Inputs> vork_probe_inputs; ConstantBuffer<DisplacementContext> vork_probe_context; RWStructuredBuffer<float3> vork_probe_result; [shader(\"compute\")][numthreads(1,1,1)] void vork_entry() { Inputs i=vork_probe_inputs;\n";
        for (slot, field) in fields.iter().enumerate() {
            match field.ty {
                Type::Attribute(width) => {
                    let ty = if width == 1 {
                        "float".into()
                    } else {
                        format!("float{width}")
                    };
                    writeln!(
                        probe,
                        "i.{}=vork_attribute_handle<{ty}>({}u);",
                        field.name,
                        slot * 5
                    )?;
                }
                Type::Image => {
                    writeln!(probe, "i.{}=vork_image({}u);", field.name, slot)?;
                }
                _ => unreachable!(),
            }
        }
        probe.push_str("vork_probe_result[0]=displace(vork_probe_context,i); }\n");
        let output = self.run(&work, &probe, &includes, false)?;
        let reads = effects::displacement_reads(&output.wgsl, fields.len())?;
        Ok(fields
            .into_iter()
            .enumerate()
            .filter(|(i, _)| reads[*i])
            .map(|(_, f)| f.name.clone())
            .collect())
    }

    /// Effect 4 is whether a declared `lit` stage's closure can reflect light.
    /// Returns the capabilities with every word unknown, and the probe WGSL.
    fn surface_effects(&self, abi: &Abi, source: &str, origin: &Path, lit: bool, inputs: &InputLayout) -> Result<(ProgramCapabilities, String)> {
        let work = Workspace::new()?;
        write_modules(&work, abi, source)?;
        fs::write(work.0.join("vork.slang"), openpbr::with_layer_effects(&abi.module(), &abi.closure))?;
        let mut includes = vec![work.0.clone()]; includes.extend(include_paths(origin));
        let field = |name: &str| if abi.closure.iter().any(|f| f.name == name) { format!("c.{name}") } else { "0.".into() };
        let opacity = if abi.closure.iter().any(|f| f.name == "geometry_opacity") { "1.-c.geometry_opacity" } else { "0." };
        let square = |name: &str| if abi.closure.iter().any(|f| f.name == name) { format!("l.{name}*l.{name}") } else { "0.".into() };
        let reflects = ["base_weight", "specular_weight", "coat_weight", "fuzz_weight", "transmission_weight", "subsurface_weight"].map(square).join("+");
        let surface_dispersion=registry::EFFECT_SURFACE_DISPERSION;
        let lit_dispersion=registry::EFFECT_LIT_DISPERSION;
        let surface_wavelength=registry::EFFECT_SURFACE_WAVELENGTH;
        let lit_wavelength=registry::EFFECT_LIT_WAVELENGTH;
        let hair_wavelength=slots(&hair_fields()).into_iter().map(|(_,f,at)|format!("{WAVELENGTH_EFFECT}({})",to_f32_slang(f.ty,&format!("h.{}{at}",f.name)))).collect::<Vec<_>>().join("+");
        let surface_coat_anisotropy=registry::EFFECT_SURFACE_COAT_ANISOTROPY;
        let lit_coat_anisotropy=registry::EFFECT_LIT_COAT_ANISOTROPY;
        let surface_weighted_ior=registry::EFFECT_SURFACE_WEIGHTED_IOR;
        let lit_weighted_ior=registry::EFFECT_LIT_WEIGHTED_IOR;
        let lit_probe = if lit { format!("Closure l=lit(vork_context,inputs,vork_light); vork_effects[4]={reflects}; vork_effects[{lit_weighted_ior}]=vork_weighted_ior_of(l); let lr=vork_response_of(l); vork_effects[9]=lr.x; vork_effects[10]=lr.y; vork_effects[{lit_coat_anisotropy}]=lr.w; vork_effects[{lit_dispersion}]=lr.z; vork_effects[{lit_wavelength}]=vork_wavelength_of(l); let lit_layers=vork_layers_of(l); vork_effects[14]=lit_layers.x; vork_effects[15]=lit_layers.y; vork_effects[16]=lit_layers.z;") } else { String::new() };
        let spatial = openpbr::spatial_subsurface(&abi.closure);
        let tint = if abi.closure.iter().any(|f| f.name == "specular_color") { "(c.specular_color.x-1.)*(c.specular_color.x-1.)+(c.specular_color.y-1.)*(c.specular_color.y-1.)+(c.specular_color.z-1.)*(c.specular_color.z-1.)" } else { "0." };
        let probe = format!("{}\nfloat vork_weighted_ior_of(Closure c) {{return vork_closure_weighted_ior(c);}}\nfloat vork_weighted_ior_of(Blend s) {{return vork_blend_weighted_ior(s);}}\nfloat vork_weighted_ior_of(Hair h) {{return 1.;}}\nfloat vork_emission_of(Closure c) {{return vork_closure_emission(c);}}\nfloat vork_emission_of(Blend s) {{return vork_blend_emission(s);}}\nfloat vork_emission_of(Hair h) {{return 0.;}}\nfloat vork_wavelength_of(Closure c) {{return vork_closure_wavelength(c);}}\nfloat vork_wavelength_of(Blend s) {{return vork_blend_wavelength(s);}}\nfloat vork_wavelength_of(Hair h) {{return {hair_wavelength};}}\nfloat3 vork_layers_of(Closure c) {{return float3({coat},{fuzz},{tint});}}\nfloat3 vork_layers_of(Blend s) {{return vork_blend_layers(s);}}\nfloat3 vork_layers_of(Hair h) {{return float3(0.);}}\nfloat4 vork_response_of(Closure c) {{return float4({anisotropy},{film},{dispersion},{coat_anisotropy});}}\nfloat4 vork_response_of(Blend s) {{return vork_blend_response(s);}}\nfloat4 vork_response_of(Hair h) {{return float4(0.);}}\nfloat vork_spatial_of(Closure c) {{return {spatial};}}\nfloat vork_spatial_of(Blend s) {{return vork_blend_effects(s).w;}}\nfloat vork_spatial_of(Hair h) {{return 0.;}}\nfloat4 vork_effects_of(Closure c) {{ return float4({},{opacity},{},0.); }}\n\
            float4 vork_effects_of(Blend s) {{ let e=vork_blend_effects(s); return float4(e.x,e.y,e.z,1.); }}\n\
            float4 vork_effects_of(Hair h) {{ return float4(0.); }}\n\
            float vork_count_of(Closure c) {{return 1.;}} float vork_count_of(Hair h) {{return 1.;}} float vork_count_of(Blend s) {{return float(vork_blend_count(s));}}\n\
            float vork_is_hair(Closure c) {{ return 0.; }} float vork_is_hair(Blend s) {{ return 0.; }} float vork_is_hair(Hair h) {{ return 1.; }}\n\
            float vork_physical_of(Closure c) {{ return 0.; }} float vork_physical_of(Blend s) {{ return 0.; }} float vork_physical_of(Hair h) {{ return h.scattering ? 1. : 0.; }}\n\
            ConstantBuffer<ShadingContext> vork_context; ConstantBuffer<Lighting> vork_light; RWStructuredBuffer<float> vork_effects; [shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{ uint base=0u;\n{decode} let vork_s=surface(vork_context,inputs); vork_effects[{surface_weighted_ior}]=vork_weighted_ior_of(vork_s); vork_effects[{lit_weighted_ior}]=0.; vork_effects[{emission_effect}]=vork_emission_of(vork_s); vork_effects[17]=vork_physical_of(vork_s); vork_effects[18]=vork_spatial_of(vork_s); let response=vork_response_of(vork_s); vork_effects[7]=response.x; vork_effects[8]=response.y; vork_effects[{surface_coat_anisotropy}]=response.w; vork_effects[{lit_coat_anisotropy}]=0.; vork_effects[{surface_wavelength}]=vork_wavelength_of(vork_s); vork_effects[{lit_wavelength}]=0.; vork_effects[{surface_dispersion}]=response.z; vork_effects[{lit_dispersion}]=0.; vork_effects[9]=0.;vork_effects[10]=0.; let ll=vork_layers_of(vork_s); vork_effects[11]=ll.x;vork_effects[12]=ll.y;vork_effects[13]=ll.z;vork_effects[14]=0.;vork_effects[15]=0.;vork_effects[16]=0.; let e=vork_effects_of(vork_s); vork_effects[0]=e.x;vork_effects[1]=e.y;vork_effects[2]=e.z;vork_effects[3]=e.w; vork_effects[5]=vork_is_hair(vork_s); vork_effects[6]=vork_count_of(vork_s); {lit_probe} }}",
            program_text(source,origin), field("subsurface_weight"), field("transmission_weight"), anisotropy=format!("({})+({})*({})",field("specular_roughness_anisotropy"),field("coat_weight"),field("coat_roughness_anisotropy")), coat_anisotropy=format!("({})*({})",field("coat_weight"),field("coat_roughness_anisotropy")), film=field("thin_film_weight"), dispersion=format!("({})*({})",field("transmission_weight"),field("transmission_dispersion_scale")), coat=field("coat_weight"), fuzz=field("fuzz_weight"), decode=inputs.slang_decode(), emission_effect=registry::EFFECT_SURFACE_EMISSION);
        let output = self.run(&work, &probe, &includes, false)?;
        Ok((EffectProbe::parse(&output.wgsl, lit)?.capabilities(&[]), output.wgsl))
    }

    /// Reflect authorable Inputs without compiling a stage or choosing a backend.
    pub fn reflect_inputs(&self, abi: &Abi, source: &str, origin: &Path) -> Result<InputLayout> {
        self.version()?;
        let work = Workspace::new()?;
        write_modules(&work, abi, source)?;
        let mut includes = vec![work.0.clone()];
        includes.extend(include_paths(origin));
        let probe = format!("{}\nConstantBuffer<Inputs> vork_inputs;\n[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}", program_text(source, origin));
        let output = self.run(&work, &probe, &includes, true)?;
        let mut inputs = InputLayout::reflect(&output.reflection)?;
        enums::reflect(&work.0, &mut inputs)?;
        Ok(inputs)
    }

    /// Discover module-level stage declarations using the compiler AST, including imports.
    /// Dependencies let callers cache the result with the same invalidation as compilation.
    pub fn stages(&self, abi: &Abi, source: &str, origin: &Path) -> Result<(BTreeSet<String>, Vec<PathBuf>)> {
        self.version()?;
        let work = Workspace::new()?;
        write_modules(&work, abi, source)?;
        let mut includes = vec![work.0.clone()];
        includes.extend(include_paths(origin));
        let probe = format!("{}\n[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}", program_text(source, origin));
        let mut output = self.run(&work, &probe, &includes, true)?;
        let program_file = work.0.join("program.slang");
        output.dependencies.retain(|d| d != &program_file && !d.starts_with(&work.0));
        Ok((output.declarations, output.dependencies))
    }

    fn compile(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
        precise: bool,
        wrapper: impl FnOnce(&InputLayout, &BTreeSet<String>) -> Result<String>,
    ) -> Result<Compiled> {
        self.compile_with_capacity(abi, source, origin, tag, None, false, false, precise, wrapper)
    }

    fn compile_with_capacity(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
        capacity: Option<&std::cell::Cell<u32>>,
        allow_workgroup: bool,
        optimize: bool,
        precise: bool,
        wrapper: impl FnOnce(&InputLayout, &BTreeSet<String>) -> Result<String>,
    ) -> Result<Compiled> {
        ensure!(
            !tag.is_empty() && tag.bytes().all(|b| b.is_ascii_alphanumeric()),
            "program tag must be alphanumeric"
        );
        self.version()?;
        let work = Workspace::new()?;
        write_modules(&work, abi, source)?;
        let mut includes = vec![work.0.clone()];
        includes.extend(include_paths(origin));
        let program = program_text(source, origin);
        let entry = "[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {}\n";
        let probe = format!("{program}ConstantBuffer<Inputs> vork_inputs;\n{entry}");
        // The probe's compiler AST decides optional stages before generating wrappers.
        // A declared stage with a bad signature must fail, never silently fall back.
        let probe_output = self.run(&work, &probe, &includes, true)?;
        let mut inputs = InputLayout::reflect(&probe_output.reflection)?;
        enums::reflect(&work.0, &mut inputs)?;
        let wrapper = wrapper(&inputs, &probe_output.declarations)?;
        if let Some(capacity) = capacity {
            fs::write(work.0.join("vork.slang"), openpbr::with_capacity(&abi.module(), &abi.closure, capacity.get()))?;
        }
        // Medium interval certificates describe the ordered expression graph.
        // Preserve that graph both here and when native consumers link it.
        let mut output = self.run_optimized(&work, &format!("{program}{wrapper}"), &includes, false, optimize, precise)?;
        output.wgsl = isolate(&output.wgsl, tag, allow_workgroup)?;
        validate(&output.wgsl, abi)
            .with_context(|| format!("{}: Slang emitted invalid WGSL", origin.display()))?;
        let program_file = work.0.join("program.slang");
        output
            .dependencies
            .retain(|d| d != &program_file && !d.starts_with(&work.0));
        Ok(Compiled {
            native: format!("{program}{wrapper}"),
            inputs,
            output,
            declarations: probe_output.declarations,
        })
    }

    fn run(
        &self,
        work: &Workspace,
        source: &str,
        includes: &[PathBuf],
        probe: bool,
    ) -> Result<Output> {
        self.run_optimized(work, source, includes, probe, false, false)
    }
    fn run_optimized(
        &self,
        work: &Workspace,
        source: &str,
        includes: &[PathBuf],
        probe: bool,
        optimize: bool,
        precise: bool,
    ) -> Result<Output> {
        let (input, wgsl, reflection, depfile) = (
            work.0.join("program.slang"),
            work.0.join("program.wgsl"),
            work.0.join("reflection.json"),
            work.0.join("program.d"),
        );
        fs::write(&input, source)?;
        let _progress = progress::Job::start(false).named("program");
        let mut command = Command::new(&self.executable);
        command.arg(&input).arg(INITIALIZATION).current_dir(&work.0);
        // Image programs execute directly as WGSL; give their numeric loops
        // the same speed optimization level as the native renderer shaders.
        if optimize {command.arg("-O2");}
        if precise {command.args(MEDIUM_FP_FLAGS);}
        if probe {
            command.arg("-dump-ast");
        }
        for include in includes {
            command.arg("-I").arg(if include.is_absolute() {
                include.clone()
            } else {
                std::env::current_dir()?.join(include)
            });
        }
        command
            .args([
                "-target",
                "wgsl",
                "-entry",
                "vork_entry",
                "-reflection-json",
            ])
            .arg(&reflection)
            .arg("-depfile")
            .arg(&depfile)
            .arg("-o")
            .arg(&wgsl);
        let (output, snapshot) = if self.expected.is_some() || self.captured.is_some() {
            let (output, snapshot) = identity::capture_command(work, &mut command, &depfile)?;
            (output, Some(snapshot))
        } else { (compiler_output(&mut command).context("run Slang compiler")?, None) };
        let diagnostics = String::from_utf8_lossy(&output.stderr).into_owned();
        ensure!(
            output.status.success(),
            "slangc {}: {diagnostics}",
            output.status
        );
        let dependencies = snapshot.as_ref().map(|s| s.dependencies())
            .unwrap_or(depfile_paths(&read_compiler_text(&depfile)?));
        reject_reserved_dependencies(work, &dependencies)?;
        if let Some(snapshot) = &snapshot {
            identity::check_imports(self, work, snapshot)?;
        }
        Ok(Output {
            wgsl: read_compiler_text(wgsl)?,
            reflection: serde_json::from_slice(&fs::read(reflection)?)?,
            dependencies,
            diagnostics,
            declarations: if probe {
                probe_declarations(&work.0)?
            } else {
                BTreeSet::new()
            },
        })
    }
}

// Slang emits host-native line endings in WGSL, AST dumps and depfiles.
fn read_compiler_text(path: impl AsRef<Path>) -> Result<String> {
    Ok(fs::read_to_string(path)?.replace("\r\n", "\n"))
}

#[cfg(test)]
mod reserved_tests {
    use super::*;
    #[test]
    fn library_source_access_is_unchanged_by_comments() {
        for module in library_modules() {
            assert_eq!(
                reject_reserved(module.source, module.name).is_ok(),
                reject_reserved(&format!("// edited\n{}", module.source), module.name).is_ok(),
                "{} changes compiler access after a comment", module.name
            );
        }
    }
    #[test]
    fn authored_source_cannot_name_compiler_plumbing() {
        for source in ["uint s=vork_slot(inputs.image);", "Image i=vork_image(3u);", "float x=vork_input_word(0u);"] {
            let error = reject_reserved(source, "program").unwrap_err().to_string();
            assert!(error.contains("reserved"), "{error}");
        }
        // Comments, strings, other casing and longer identifiers are not uses.
        reject_reserved("// vork_slot\n/* vork_image */ uint my_vork_slot=VORK_ABI; let s=\"vork_x\"; let é=1;", "program").unwrap();
        // Escapes neither end a string early nor hide what follows it.
        reject_reserved(r#"#define Q "\"vork_slot""#, "program").unwrap();
        assert!(reject_reserved("#define Q \"\\\"\"\nuint x=vork_slot(i);", "program").is_err());
        // Compiler intrinsics and token pasting cannot reach the plumbing either.
        for source in ["__target_switch { case wgsl: __intrinsic_asm \"vork_material_word($0)\"; }", "#define J(a,b) a##b\nJ(vork,_image)(3u);"] {
            assert!(reject_reserved(source, "program").is_err(), "{source}");
        }
        let error = reject_reserved("uint a;\nuint b=vork_slot(i);", "p.slang").unwrap_err().to_string();
        assert!(error.starts_with("p.slang:2:"), "{error}");
        // Kernel modules supplied to every program are importable.
        reject_reserved("// Header.\nimport vork_sampling;\nfloat2 p=sample_2d(uint2(0u),0u,4u,0u);", "program").unwrap();
        assert!(reject_reserved("import vork_sampling;\nuint w=vork_input_word(0u);", "program").is_err());
        reject_reserved("import vork_openpbr_constants;", "program").unwrap();
        assert!(reject_reserved("import vork_private;", "program").is_err());
        // Source provenance never grants compiler access, even for shipped code.
        let internal = module_source("std/grid").unwrap();
        assert!(reject_reserved(internal, "library").is_err());
        assert!(reject_reserved(&format!("// edited\n{internal}"), "library").is_err());
        let compositor = "import vork; float4 composite(CompositeContext ctx) { return compositor_sample(ctx.uv); }";
        reject_reserved(compositor, "program").unwrap();
        reject_reserved(&format!("// edited\n{compositor}"), "program").unwrap();
    }
}

fn write_modules(work: &Workspace, abi: &Abi, source: &str) -> Result<()> {
    reject_reserved(source, "program")?;
    fs::write(work.0.join("vork.slang"), abi.module())?;
    for module in library_modules() {
        let path = work.0.join(format!("{}.slang", module.name));
        fs::create_dir_all(path.parent().unwrap())?;
        fs::write(path, module.source)?;
    }
    for (name, source) in KERNEL_MODULES {
        fs::write(work.0.join(format!("{name}.slang")), source)?;
    }
    fs::write(
        work.0.join("openpbr_core.slang"),
        module_source("std/openpbr_core").unwrap(),
    )?;
    fs::write(
        work.0.join("vork_openpbr_constants.slang"),
        &renderer_registry::generate(true),
    )?;
    Ok(())
}

/// A program's module text: the `vork` import, its source, then generated code.
fn program_text(source: &str, origin: &Path) -> String {
    format!(
        "import vork;\n#line 1 \"{}\"\n{source}\n#line 1 \"<vork program wrapper>\"\n",
        origin
            .display()
            .to_string()
            .replace('\\', "/")
            .replace('"', "\\\"")
    )
}

fn include_paths(origin: &Path) -> Vec<PathBuf> {
    origin
        .parent()
        .map(|p| if p.as_os_str().is_empty() { Path::new(".") } else { p })
        .map(Path::to_path_buf)
        .into_iter()
        .collect()
}

struct Compiled {
    native: String,
    inputs: InputLayout,
    output: Output,
    declarations: BTreeSet<String>,
}
struct Output {
    wgsl: String,
    reflection: Value,
    dependencies: Vec<PathBuf>,
    diagnostics: String,
    declarations: BTreeSet<String>,
}

// Slang 2026.21's AST dump is generated by the successful reflection probe.
// Read declaration records, not diagnostics or source text (comments/macros/imports
// cannot manufacture a stage). Only module-level functions are stage candidates.
fn probe_declarations(work: &Path) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for file in fs::read_dir(work)? {
        let path = file?.path();
        if path.extension().is_none_or(|ext| ext != "slang-ast") {
            continue;
        }
        let ast = read_compiler_text(path)?;
        names.extend(ast_declarations(&ast)?);
    }
    // This function is injected into every probe. A missing name means the AST
    // format changed, rather than that an optional user stage was absent.
    ensure!(
        names.contains("vork_entry"),
        "Slang AST probe did not recognize vork_entry; cannot discover optional stages"
    );
    Ok(names)
}

fn ast_declarations(ast: &str) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    for record in ast.split("\nFuncDecl:").skip(1) {
        let header = record.split("\n}").next().unwrap_or(record);
        let parent = header
            .lines()
            .find_map(|line| line.trim_start().strip_prefix("parentDecl : "))
            .context("Slang AST function record has no recognized parentDecl")?;
        if !parent.starts_with("ModuleDecl:") {
            continue;
        }
        let name = header
            .lines()
            .find_map(|line| {
                line.trim_start()
                    .strip_prefix("nameAndLoc : NameLoc{\"")
                    .and_then(|line| line.split_once('"').map(|(name, _)| name))
            })
            .filter(|name| !name.is_empty())
            .context("Slang AST module function has no recognized nameAndLoc")?;
        names.insert(name.to_owned());
    }
    Ok(names)
}

#[cfg(test)]
mod ast_tests {
    use super::*;

    #[test]
    fn aov_abi_storage_and_cache_identity_follow_the_declared_count() -> Result<()> {
        let mut identities = std::collections::BTreeSet::new();
        for count in [0, 1, 3, 8, 32] {
            let abi = Abi { aov_count: count, context: Abi::spec_context(), closure: openpbr_fields() };
            for module in [abi.module(), abi.native_module()] {
                assert!(module.contains(&format!("float4 values[{}]", count.max(1))));
                assert!(module.contains(&format!("target.slot<{count}u")));
            }
            assert!(identities.insert(abi.cache_identity()?));
        }
        Ok(())
    }

    #[test]
    fn compiler_text_normalizes_crlf_before_parsing() -> Result<()> {
        let work = Workspace::new()?;
        let path = work.0.join("program.wgsl");
        fs::write(&path, "@compute\r\n@workgroup_size(1, 1, 1)\r\nfn vork_entry()\r\n{\r\n    return;\r\n}\r\n")?;
        let text = read_compiler_text(&path)?;
        assert!(!text.contains('\r'));
        assert_eq!(isolate(&text, "test", false)?, "");
        Ok(())
    }

    #[test]
    fn depfiles_decode_slang_makefile_escapes() {
        let text = r"C\:\\tmp\\out.wgsl: C\:\\tmp\\vork-slang-123\\std\\math.slang C\:\\My\ Scene\\shader\#\[1\]$$.slang";
        assert_eq!(depfile_paths(text), vec![
            PathBuf::from(r"C:\tmp\vork-slang-123\std\math.slang"),
            PathBuf::from(r"C:\My Scene\shader#[1]$.slang"),
        ]);
        assert_eq!(depfile_paths("out: /tmp/a\\ b.slang \\\n /tmp/c.slang\n"), vec![
            PathBuf::from("/tmp/a b.slang"), PathBuf::from("/tmp/c.slang"),
        ]);
    }

    #[cfg(unix)]
    #[test]
    fn compiler_start_failure_preserves_loader_diagnostic() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        let work = Workspace::new()?;
        let executable = work.0.join("cannot-run");
        fs::write(&executable, "#!/bin/sh\necho 'dyld: library built for macOS 26.0; symbol not found' >&2\nexit 1\n")?;
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))?;
        let error = Compiler::new(&executable).version().unwrap_err();
        assert!(error.downcast_ref::<CompilerUnavailable>().is_some());
        let message = error.to_string();
        assert!(message.contains("could not run"), "{message}");
        assert!(message.contains("built for macOS 26.0"), "{message}");
        assert!(!message.contains("required, found"), "{message}");
        assert!(compiler_start_advice(true).contains("update macOS or use a compatible vork build"));
        assert!(!compiler_start_advice(true).contains("reinstall"));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn version_is_checked_once_across_instances_and_threads() -> Result<()> {
        let work = Workspace::new()?;
        for (name, version) in [("pinned", VERSION), ("wrong", "wrong-version")] {
            let executable = work.0.join(name);
            let count = work.0.join(format!("{name}.count"));
            crate::test_support::write_executable(&executable, format!("#!/bin/sh\necho check >> '{}'\necho '{version}' >&2\n", count.display()))?;
            std::thread::scope(|scope| {
                for _ in 0..8 {
                    let executable = &executable;
                    scope.spawn(move || {
                        let compiler = Compiler::new(executable);
                        for _ in 0..2 {
                            let result = compiler.version();
                            if version == VERSION {
                                assert_eq!(result.unwrap(), VERSION);
                            } else {
                                let error = result.unwrap_err();
                                assert!(error.downcast_ref::<CompilerUnavailable>().is_some());
                                assert!(error.to_string().contains("wrong-version"));
                            }
                        }
                    });
                }
            });
            assert_eq!(fs::read_to_string(count)?, "check\n");
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn a_toolchain_is_run_once_and_again_only_when_its_files_change() -> Result<()> {
        // audit-3d 9: admission is by the toolchain's identity, so a warm process does not
        // run the compiler and a changed one is still checked.
        struct Ledger(&'static str, Mutex<Vec<PathBuf>>);
        impl AdmissionLedger for Ledger {
            fn entry(&self, toolchain: blake3::Hash) -> Option<PathBuf> { Some(PathBuf::from(self.0).join(toolchain.to_hex().as_str())) }
            fn admitted(&self, entry: &Path, _: blake3::Hash) -> bool { self.1.lock().unwrap().iter().any(|held| held == entry) }
            fn record(&self, entry: &Path, _: blake3::Hash) { self.1.lock().unwrap().push(entry.to_path_buf()) }
        }
        use std::os::unix::fs::PermissionsExt;
        let work = Workspace::new()?;
        let bin = work.0.join("toolchain/bin");
        let lib = work.0.join("toolchain/lib");
        fs::create_dir_all(&bin)?;
        fs::create_dir_all(&lib)?;
        let executable = bin.join("slangc");
        let count = work.0.join("count");
        let script = |version: &str| format!("#!/bin/sh\necho check >> '{}'\necho '{version}' >&2\n", count.display());
        let runs = || fs::read_to_string(&count).map_or(0, |text| text.lines().count());
        crate::test_support::write_executable(&executable, script(VERSION))?;
        fs::write(lib.join("libslang.so"), "one")?;
        let compiler = Compiler::new(&executable);
        let ledger = Ledger("store", Mutex::default());
        // The test's own loader environment, not the runner's.
        let loader = |searched: &str, injected: &str| Loader {
            searched: vec![("LD_LIBRARY_PATH".to_owned(), searched.into())],
            other: vec![("LD_PRELOAD".to_owned(), injected.into())],
            started_in: work.0.join("started-in"),
        };
        let bare = loader("", "");
        let held = |ledger: &Ledger| ledger.1.lock().unwrap().len();
        // Each `check` stands for a new process: the per-process memo is above it.
        compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger)), (1, 1), "first seen: run and recorded");
        compiler.check(Some(&ledger), &bare)?;
        let mut checked = compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger)), (1, 1), "the same files are not run again");
        // The same process meeting another store publishes the admission there, once.
        let other = Ledger("other-store", Mutex::default());
        checked.publish(Some(&other));
        checked.publish(Some(&other));
        checked.publish(Some(&ledger));
        assert_eq!((runs(), held(&other), held(&ledger)), (1, 1, 1));
        compiler.check(Some(&other), &bare)?;
        assert_eq!(runs(), 1, "and a process that asks that store first does not run it");
        // Files that are not Slang's, installed beside the compiler or its libraries, are
        // not the toolchain: nothing is run.
        fs::write(bin.join("another-tool"), "unrelated")?;
        fs::write(lib.join("libother.so"), "unrelated")?;
        compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger)), (1, 1));
        // A file under Slang's standard-module directory is.
        fs::create_dir_all(lib.join("slang-standard-module-test/slang"))?;
        compiler.check(Some(&ledger), &bare)?;
        fs::write(lib.join("slang-standard-module-test/slang/core.slang-module"), "module")?;
        compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger)), (3, 3));
        fs::remove_dir_all(lib.join("slang-standard-module-test"))?;
        compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger)), (3, 3), "back to the files first admitted");
        // A directory the environment tells the loader to search is part of the toolchain
        // when it holds a Slang library, and by its place in the list either way.
        let elsewhere = work.0.join("other-sdk");
        fs::create_dir_all(&elsewhere)?;
        let listed = |value: &str| compiler.toolchain_under(&loader(value, ""));
        let there = elsewhere.to_str().context("workspace path")?;
        let plain = compiler.toolchain_under(&bare)?;
        let searched = listed(there)?;
        assert_ne!(plain, searched);
        fs::write(elsewhere.join("unrelated.so"), "x")?;
        assert_eq!(listed(there)?, searched, "a file that is not Slang's there changes nothing");
        fs::write(elsewhere.join("libslang-compiler.so"), "another sdk")?;
        let shadowed = listed(there)?;
        assert_ne!(shadowed, searched, "a Slang library on the loader's path is another toolchain");
        fs::write(elsewhere.join("libslang-compiler.so"), "another sdk, upgraded")?;
        assert_ne!(listed(there)?, shadowed, "and replacing it there is seen");
        // glibc takes a library from a hardware-capability subdirectory of a searched one first.
        fs::create_dir_all(elsewhere.join("glibc-hwcaps/x86-64-v3"))?;
        let capable = listed(there)?;
        fs::write(elsewhere.join("glibc-hwcaps/x86-64-v3/libslang-compiler.so"), "for this processor")?;
        assert_ne!(listed(there)?, capable);
        // An empty or relative entry is the directory the compiler starts in.
        fs::create_dir_all(work.0.join("started-in/relative"))?;
        let trailing = listed("/nonexistent:")?;
        fs::write(work.0.join("started-in/libslang-rt.so"), "in the working directory")?;
        assert_ne!(listed("/nonexistent:")?, trailing);
        let relative = listed("relative")?;
        fs::write(work.0.join("started-in/relative/libslang-rt.so"), "relative")?;
        assert_ne!(listed("relative")?, relative);
        // What the loader would interpret, or inject, cannot be observed: no identity.
        for value in ["$ORIGIN/../alternate-lib", "/opt/a;/opt/b"] {
            assert!(listed(value).is_err(), "{value}");
        }
        assert!(compiler.toolchain_under(&loader("", "/opt/other-sdk/libslang-compiler.so")).is_err());
        let injected = loader("", "/opt/other-sdk/libslang-compiler.so");
        let before = runs();
        assert_eq!(compiler.check(Some(&ledger), &injected)?.toolchain, None);
        compiler.check(Some(&ledger), &injected)?;
        assert_eq!((runs(), held(&ledger)), (before + 2, 3), "an injected library: run every time, nothing recorded");
        // A library replaced beside it: another toolchain, checked by running it.
        fs::write(lib.join("libslang.so"), "another build")?;
        compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger)), (6, 4));
        // The same bytes without their execute bits are not the admitted compiler.
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o644))?;
        assert!(compiler.check(Some(&ledger), &bare).is_err(), "a compiler that cannot be run was admitted");
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755))?;
        assert_eq!((runs(), held(&ledger)), (6, 4));
        // The executable replaced by one that reports another version: run, refused, not recorded.
        crate::test_support::write_executable(&executable, script("1900.0"))?;
        for _ in 0..2 {
            let error = compiler.check(Some(&ledger), &bare).err().context("a wrong version was admitted")?.to_string();
            assert!(error.contains("required, found \"1900.0\""), "{error}");
        }
        assert_eq!((runs(), held(&ledger)), (8, 4), "a refused toolchain is run every time");
        // Without a ledger the check runs the compiler and keeps the identity to publish later.
        crate::test_support::write_executable(&executable, script(VERSION))?;
        let mut checked = compiler.check(None, &bare)?;
        assert_eq!((runs(), checked.toolchain), (9, Some(compiler.toolchain_under(&bare)?)));
        checked.publish(Some(&ledger));
        assert_eq!(held(&ledger), 5);
        // A library directory that cannot be listed leaves no identity: the compiler is run
        // each time and nothing is recorded, however its libraries then change.
        // (A user who may list any directory, such as root, cannot be denied: nothing to check.)
        fs::set_permissions(&lib, fs::Permissions::from_mode(0o300))?;
        if fs::read_dir(&lib).is_err() {
            assert!(compiler.toolchain_under(&bare).is_err());
            let unlisted = compiler.check(Some(&ledger), &bare)?;
            compiler.check(Some(&ledger), &bare)?;
            assert_eq!((runs(), held(&ledger), unlisted.toolchain), (11, 5, None));
        }
        fs::set_permissions(&lib, fs::Permissions::from_mode(0o755))?;
        let (ran, kept) = (runs(), held(&ledger));
        // Files that change during the check (here the stand-in rewrites its own library;
        // in a shared directory, anything) leave nothing to vouch for: no record, run again.
        let changing = format!("#!/bin/sh\necho check >> '{}'\necho more >> '{}'\necho '{VERSION}' >&2\n", count.display(), lib.join("libslang.so").display());
        crate::test_support::write_executable(&executable, changing)?;
        let unstable = compiler.check(Some(&ledger), &bare)?;
        compiler.check(Some(&ledger), &bare)?;
        assert_eq!((runs(), held(&ledger), unstable.toolchain), (ran + 2, kept, None));
        // A missing executable has no identity and takes the launch diagnostic.
        fs::remove_file(&executable)?;
        let error = compiler.check(Some(&ledger), &bare).err().context("a missing compiler was admitted")?.to_string();
        assert!(error.contains("cannot start Slang"), "{error}");
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn metal_source_is_identical_across_workspaces() -> Result<()> {
        let compiler = Compiler::discover();
        let modules = [
            ("helper", "public float twice(float x) { return x*2.; }"),
            ("main", "import helper; RWStructuredBuffer<float> output; [shader(\"compute\")][numthreads(1,1,1)] void main() { output[0]=twice(output[1]); }"),
        ];
        let compile = || compiler.compile_modules(&modules, "main", &[("main", "compute")], &["-target", "metal", "-O2"]);
        let (first, second) = (compile()?, compile()?);
        assert!(!String::from_utf8_lossy(&first).contains("#line"));
        assert_eq!(first, second, "Metal source must not depend on the per-call workspace path");
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn mesh_stage_discovery_handles_imported_medium_and_combined_surface() -> Result<()> {
        let compiler = Compiler::discover();
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let root=std::env::temp_dir().join(format!("vork-medium-discovery-{}",std::process::id()));
        std::fs::create_dir_all(&root)?;
        std::fs::write(root.join("authored_medium.slang"),include_str!("../../vork-scene/tests/fixtures/medium.slang"))?;
        let origin_path=root.join("stage-discovery.slang");
        let origin=origin_path.as_path();
        for (source, surface, medium) in [
            ("struct Inputs {}; Closure surface(ShadingContext c, Inputs i) {return Closure();} // MediumCoefficients medium(...) {}", true, false),
            ("import authored_medium;", false, true),
            ("import authored_medium; Closure surface(ShadingContext c, Inputs i) {return Closure();}", true, true),
        ] {
            let (stages, _) = compiler.stages(&abi, source, origin)?;
            assert_eq!(stages.contains("surface"), surface);
            assert_eq!(stages.contains("medium"), medium);
            if surface { compiler.compile_surface(&abi, source, origin, "stagesurface")?; }
            if medium { compiler.compile_medium(&abi, source, origin, "stagemedium")?; }
        }
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn reused_native_source_refreshes_locations_without_changing_proofs() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let compiler = Compiler::discover();
        let source = "/*\n#line 1 \"<vork program wrapper>\"\n*/\nstruct Inputs {}; Closure surface(ShadingContext c, Inputs i) {return unlit(float3(1.));}";
        let mut program = compiler.compile_surface(&abi, source, Path::new("first.slang"), "relocate")?;
        let original_native = program.slang.clone();
        program.set_native_source(source, Path::new("first.slang"))?;
        assert_eq!(program.slang, original_native, "an unchanged source must keep identical native code");
        let wgsl = program.wgsl.clone();
        let moved = format!("// moved\n// twice\n{source}");
        program.set_native_source(&moved, Path::new("author/second.slang"))?;
        assert_eq!(program.wgsl, wgsl, "source locations must not change the reused proof");
        assert_eq!(program.include, vec![PathBuf::from("author")]);
        assert!(program.slang.starts_with(&format!("import vork;\n#line 1 \"author/second.slang\"\n{moved}\n")));
        assert!(!program.slang.contains("first.slang"));
        // Compile the refreshed native text too: quoting the owned marker in
        // an authored comment must not duplicate or truncate its declarations.
        let native = format!("{}\n[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}", program.slang);
        let work = Workspace::new()?;
        write_modules(&work, &abi, "")?;
        compiler.run(&work, &native, &[work.0.clone()], true)?;
        Ok(())
    }

    #[test]
    fn private_constant_tables_are_safe_but_writes_are_refused() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        let table = "var<private> table: array<f32,2> = array<f32,2>(1.,2.);";
        validate(&format!("{table} fn lookup(i:u32)->f32 {{return table[i];}}"), &abi)?;
        for body in ["table[0]=3.;", "write();"] {
            let wgsl = format!("{table} fn write() {{table[1]=4.;}} fn use_table() {{{body}}}");
            assert!(validate(&wgsl, &abi).unwrap_err().to_string().contains("read-only"));
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn static_const_array_compiles_with_dynamic_index() -> Result<()> {
        let abi = Abi { aov_count: 0, context: Abi::spec_context(), closure: openpbr_fields() };
        Compiler::discover().compile_surface(&abi,
            "struct Inputs {}; static const float3 colors[2]={float3(1,0,0),float3(0,1,0)}; Closure surface(ShadingContext c, Inputs i) {return unlit(colors[uint(c.time)%2]);}",
            Path::new("author/palette.slang"), "palette")?;
        Ok(())
    }

    #[test]
    fn optional_stage_probe_rejects_unrecognized_function_records() {
        let record = "\nFuncDecl:\n{\n parentDecl : ModuleDecl: module\n nameAndLoc : NameLoc{\"medium\"}\n}";
        assert!(ast_declarations(record).unwrap().contains("medium"));
        assert!(ast_declarations(&record.replace("nameAndLoc", "renamed")).is_err());
        assert!(ast_declarations(&record.replace("parentDecl", "renamed")).is_err());
        assert!(
            ast_declarations(&record.replace("ModuleDecl:", "StructDecl:"))
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn optional_stage_probe_requires_its_injected_entry() -> Result<()> {
        let work = std::env::temp_dir().join(format!("vork-ast-format-test-{}", std::process::id()));
        fs::create_dir_all(&work)?;
        let result = probe_declarations(&work);
        fs::remove_dir(&work)?;
        assert!(result.unwrap_err().to_string().contains("vork_entry"));
        Ok(())
    }
}

/// Makefile syntax `out: a b\ c`; spaces inside paths are backslash-escaped.
fn depfile_paths(text: &str) -> Vec<PathBuf> {
    let deps = text.split_once(": ").map_or("", |(_, d)| d);
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = deps.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\\' if chars.peek().is_some_and(|c| matches!(c, ' ' | ':' | '#' | '[' | ']' | '\\')) => {
                current.push(chars.next().unwrap());
            }
            '$' if chars.peek() == Some(&'$') => {
                chars.next();
                current.push('$');
            }
            '\\' if chars.peek() == Some(&'\n') => {
                chars.next();
            }
            c if c.is_whitespace() => {
                if !current.is_empty() {
                    out.push(PathBuf::from(std::mem::take(&mut current)));
                }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() {
        out.push(PathBuf::from(current));
    }
    out
}

/// Remove the probe entry point and tag every Slang-generated identifier
/// (Slang suffixes all of them with `_<n>`), so programs never collide with
/// each other or with kernel names. Kernel intrinsics carry no numeric suffix.
fn isolate(wgsl: &str, tag: &str, allow_workgroup: bool) -> Result<String> {
    ensure!(
        !wgsl.contains("@binding")
            && (allow_workgroup || !wgsl.contains("var<workgroup>")),
        "programs may not declare global resources or variables"
    );
    let entry = "@compute\n@workgroup_size(1, 1, 1)\nfn vork_entry()\n{\n    return;\n}\n";
    let body = wgsl.replacen(entry, "", 1);
    ensure!(
        !body.contains("vork_entry"),
        "unexpected Slang entry layout"
    );
    let mut out = String::with_capacity(body.len() + body.len() / 8);
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let word = &body[start..i];
            out.push_str(word);
            let digits = |n: &str| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit());
            // `name_<n>` and Slang temporaries `_S<n>`.
            let numbered = word.rsplit_once('_').is_some_and(|(_, n)| digits(n))
                || word.strip_prefix("_S").is_some_and(digits);
            if numbered && start.checked_sub(1).is_none_or(|p| bytes[p] != b'@') {
                write!(out, "_{tag}").unwrap();
            }
        } else if c.is_ascii_digit() {
            // Numeric literals (e.g. 1.5e-3f) are not identifiers.
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'.') {
                i += 1;
            }
            out.push_str(&body[start..i]);
        } else {
            out.push(c as char);
            i += 1;
        }
    }
    Ok(out)
}

/// Check a program in isolation, with stub kernel functions and structs, so a
/// Slang/naga disagreement surfaces at install with the program's name.
fn validated_module(wgsl: &str, abi: &Abi) -> Result<naga::Module> {
    let mut module = abi.structs_wgsl();
    module.push_str(&coordinates::validation_stubs());
    module.push_str("fn vork_blend_overflow(count:u32,capacity:u32) {}\n");
    module.push_str("fn vork_attribute(slot:u32,differential:u32)->vec4<f32>{return vec4(f32(slot+differential));}\n");
    module.push_str("fn vork_compositor_context(index:u32)->vec4<f32>{return vec4(f32(index));}\n");
    module.push_str("fn vork_compositor_size(slot:u32)->vec2<f32>{return vec2(1.); }\n");
    module.push_str("fn vork_compositor_load(slot:u32,p:vec2<i32>)->vec4<f32>{return vec4(vec2<f32>(p),f32(slot),1.); }\n");
    module.push_str("fn vork_view_load(slot:u32,p:vec2<i32>)->vec4<f32>{return vec4(vec2<f32>(p),f32(slot),1.); }\n");
    write!(
        module,
        "fn {MATERIAL_WORD}(i:u32)->u32 {{ return i; }}\n\
         fn {SAMPLE_IMAGE}(slot:u32, uv:vec2<f32>, ddx:vec2<f32>, ddy:vec2<f32>)->vec4<f32> {{ return vec4(uv+ddx+ddy, f32(slot), 1.); }}\n{}{wgsl}",
        abi.kernel_wgsl()
    )
    .unwrap();
    let parsed = naga::front::wgsl::parse_str(&module)
        .map_err(|e| anyhow::anyhow!(e.emit_to_string(&module)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        // Half packing is ordinary f32 arithmetic, also used by compositor
        // working images; it does not enable native f16 shader types.
        naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32,
    )
    .validate(&parsed)
    .map_err(|e| anyhow::anyhow!("{e:?}"))?;
    for (global, variable) in parsed.global_variables.iter() {
        if variable.space != naga::AddressSpace::Private { continue; }
        let writes = naga::valid::GlobalUse::WRITE | naga::valid::GlobalUse::ATOMIC;
        ensure!(variable.init.is_some()
            && parsed.functions.iter().all(|(f, _)| !info[f][global].intersects(writes))
            && (0..parsed.entry_points.len()).all(|e| !info.get_entry_point(e)[global].intersects(writes)),
            "program global {:?} must be initialized and read-only; use Inputs or compute buffers for mutable state; see vork help programs",
            variable.name);
    }
    Ok(parsed)
}

fn validate(wgsl: &str, abi: &Abi) -> Result<()> {
    validated_module(wgsl, abi).map(drop)
}

struct Workspace(PathBuf);
impl Workspace {
    fn new() -> Result<Self> {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        loop {
            let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("vork-slang-{}-{serial}", std::process::id()));
            match fs::create_dir(&path) {
                // Match the compiler's canonical dependency spelling, including
                // Windows short TEMP names, without its unsupported verbatim prefix.
                Ok(()) => {
                    let path = path.canonicalize()?;
                    #[cfg(windows)]
                    let path = PathBuf::from(filesystem::compiler_path(&path));
                    return Ok(Self(path));
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }
}
impl Drop for Workspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputField {
    #[serde(default)]
    pub enumeration: Option<InputEnum>,
    #[serde(default)]
    pub range: Option<InputRange>,
    pub name: String,
    pub ty: Type,
    /// Word offset from the material's base, after the supplied-input masks.
    pub offset: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputRange {
    pub min: f64,
    pub max: f64,
    pub min_closed: bool,
    pub max_closed: bool,
}
/// Material words: ceil(n/32) supplied-input mask words, then each input's
/// words in declaration order. Unsupplied inputs keep Slang initializers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InputLayout {
    pub fields: Vec<InputField>,
    pub words: usize,
}

impl InputLayout {
    pub fn slang_decode(&self) -> String {self.slang_decode_images(&BTreeMap::new())}
    fn slang_decode_images(&self, images:&BTreeMap<String,u32>) -> String {
        let inputs = self;
        let mut s = String::new();
        // Unsupplied inputs keep the author's Slang initializers.
        s.push_str("    Inputs inputs = {};\n");
        for m in 0..inputs.fields.len().div_ceil(32) {
            writeln!(s, "    uint supplied{m} = vork_input_word(base + {m});").unwrap();
        }
        for (n, f) in inputs.fields.iter().enumerate() {
            if let Some(slot)=images.get(&f.name) {
                writeln!(s,"    inputs.{} = {};",f.name,handle_slang(f.ty,&format!("{slot}u"))).unwrap();
                continue;
            }
            let mask = format!("supplied{}", n / 32);
            writeln!(s, "    if (({mask} & {}u) != 0) {{", 1u32 << (n % 32)).unwrap();
            let word = |j: usize| format!("vork_input_word(base + {})", f.offset + j);
            if let Some(enumeration) = &f.enumeration {
                writeln!(s, "        switch ({}) {{", word(0)).unwrap();
                for (i, case) in enumeration.cases.iter().enumerate() {
                    writeln!(s, "        case {i}u: inputs.{} = {}::{case}; break;", f.name, enumeration.ty).unwrap();
                }
                s.push_str("        }\n    }\n");
                continue;
            }
            let value = match f.ty {
                Type::Float => format!("asfloat({})", word(0)),
                Type::Int => format!("asint({})", word(0)),
                Type::Uint => word(0),
                Type::Bool => format!("{} != 0", word(0)),
                Type::Geometry => format!("vork_geometry({})",word(0)),
                Type::Image => format!("vork_image_input(base, {})", word(0)),
                Type::FloatGrid => format!("vork_float_grid(base + {})",word(0)),
                Type::VectorGrid => format!("vork_vector_grid(base + {})",word(0)),
                Type::Aov => format!("vork_aov({})",word(0)),
                Type::Attribute(n) => format!("vork_attribute_handle<{}>(base + {})", if n==1 {"float".into()} else {format!("float{n}")},word(0)),
                Type::Vector(w) => {
                    let parts: Vec<_> = (0..w as usize)
                        .map(|j| format!("asfloat({})", word(j)))
                        .collect();
                    format!("float{w}({})", parts.join(", "))
                }
            };
            writeln!(s, "        inputs.{} = {value};\n    }}", f.name).unwrap();
        }

        s
    }

    /// Packed words with animated numeric inputs unknown and marked supplied.
    /// Static words (including resource descriptors) retain their existing proofs.
    pub fn varying_words(&self, packed: &[u32], dynamic: &BTreeSet<String>) -> Vec<Option<u32>> {
        let mut words: Vec<_> = packed.iter().copied().map(Some).collect();
        for (n, field) in self.fields.iter().enumerate() {
            if dynamic.contains(&field.name) {
                words[n / 32] = words[n / 32].map(|mask| mask | 1 << (n % 32));
                words[field.offset..field.offset + field.ty.words()].fill(None);
            }
        }
        words
    }

    /// An instance's material words for `EffectProbe::capabilities`: `pack`ed
    /// values, with resource handles (images, grids, AOVs, attributes) and
    /// `dynamic` (animated) inputs unknown. An animated input reads as
    /// supplied: its unknown value covers its initializer too.
    pub fn effect_words(&self, values: &BTreeMap<String, Value>, dynamic: &BTreeSet<String>) -> Result<Vec<Option<u32>>> {
        let mut words = self.varying_words(&self.pack(values, |_, _| Ok(0))?, dynamic);
        for f in &self.fields {
            let animated = dynamic.contains(&f.name);
            if animated || matches!(f.ty, Type::Geometry | Type::Image | Type::FloatGrid | Type::VectorGrid | Type::Aov | Type::Attribute(_)) {
                words[f.offset..f.offset + f.ty.words()].fill(None);
            }
        }
        Ok(words)
    }

    /// From the reflection of the probe's `ConstantBuffer<Inputs> vork_inputs`.
    pub fn reflect(reflection: &Value) -> Result<Self> {
        let ty = &reflection["parameters"]
            .as_array()
            .context("Slang parameters missing")?
            .iter()
            .find(|p| p["name"] == "vork_inputs")
            .context("program must declare struct Inputs")?["type"]["elementType"];
        ensure!(ty["kind"] == "struct", "Inputs must be a struct");
        let reflected = ty["fields"].as_array().context("input fields missing")?;
        let mut offset = reflected.len().div_ceil(32);
        let mut fields = Vec::new();
        for f in reflected {
            let name = f["name"].as_str().context("input name missing")?.to_owned();
            let t = &f["type"];
            let ty = match (t["kind"].as_str(), t["scalarType"].as_str()) {
                (Some("scalar"), Some("float32")) => Type::Float,
                (Some("scalar"), Some("int32")) => Type::Int,
                (Some("scalar"), Some("uint32")) => Type::Uint,
                (Some("scalar"), Some("bool")) => Type::Bool,
                (Some("vector"), _) if t["elementType"]["scalarType"] == "float32" => {
                    let n = t["elementCount"].as_u64().context("vector width missing")? as u32;
                    ensure!(
                        (2..=4).contains(&n),
                        "input {name}: unsupported vector width"
                    );
                    Type::Vector(n)
                }
                (Some("struct"), _) if t["name"] == "Geometry" => Type::Geometry,
                (Some("struct"), _) if t["name"] == "Image" => Type::Image,
                (Some("struct"), _) if t["name"] == "FloatGrid" => Type::FloatGrid,
                (Some("struct"), _) if t["name"] == "VectorGrid" => Type::VectorGrid,
                (Some("struct"), _) if t["name"] == "Aov" => Type::Aov,
                (Some("struct"), _) if t["name"] == "Attribute" => {
                    let witness=&t["fields"][1]["type"];
                    let width=if witness["kind"]=="scalar" && witness["scalarType"]=="float32" {1} else {
                        ensure!(witness["kind"]=="vector" && witness["elementType"]["scalarType"]=="float32", "input {name}: Attribute supports float through float4");
                        witness["elementCount"].as_u64().context("attribute width missing")? as u32
                    };
                    ensure!((1..=4).contains(&width),"input {name}: Attribute supports float through float4");
                    Type::Attribute(width)
                },
                _ => bail!(
                    "input {name}: unsupported type {t}; inputs are float, int, uint, bool, float2-4, Attribute<T> or Image"
                ),
            };
            let range = f["userAttribs"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|attribute| attribute["name"] == "InputRange")
                .map(|attribute| -> Result<InputRange> {
                    let args = attribute["arguments"]
                        .as_array()
                        .context("InputRange requires four arguments")?;
                    ensure!(
                        args.len() == 4,
                        "InputRange requires min, max, min_closed and max_closed"
                    );
                    Ok(InputRange {
                        min: args[0].as_f64().context("InputRange min")?,
                        max: args[1].as_f64().context("InputRange max")?,
                        min_closed: args[2]
                            .as_bool()
                            .or_else(|| args[2].as_i64().map(|x| x != 0))
                            .context("InputRange min_closed")?,
                        max_closed: args[3]
                            .as_bool()
                            .or_else(|| args[3].as_i64().map(|x| x != 0))
                            .context("InputRange max_closed")?,
                    })
                })
                .transpose()?;
            fields.push(InputField {
                enumeration: None,
                name,
                ty,
                offset,
                range,
            });
            offset += ty.words();
        }
        Ok(Self {
            fields,
            words: offset,
        })
    }

    /// Pack supplied JSON values. `image` resolves an image input value
    /// (a tagged resource binding) to its table slot.
    pub fn pack(
        &self,
        values: &BTreeMap<String, Value>,
        mut image: impl FnMut(&str, &Value) -> Result<u32>,
    ) -> Result<Vec<u32>> {
        let mut words = vec![0u32; self.words];
        vork_scene::diagnostics::joined(
            values
                .keys()
                .filter(|name| !self.fields.iter().any(|f| &f.name == *name))
                .map(|name| {
                    anyhow::anyhow!(
                        "{}",
                        vork_scene::diagnostics::unknown(
                            "program input",
                            name,
                            self.fields.iter().map(|f| f.name.as_str())
                        )
                    )
                })
                .collect(),
        )?;
        for (name, value) in values {
            let (n, field) = self
                .fields
                .iter()
                .enumerate()
                .find(|(_, f)| &f.name == name)
                .expect("unknown inputs reported above");
            if let Some(enumeration) = &field.enumeration {
                let token = value.as_str().with_context(|| format!("input {name}: expected enum token"))?;
                let tokens: Vec<_> = enumeration.cases.iter().map(|case| case.to_ascii_lowercase()).collect();
                let index = tokens.iter().position(|case| case == token).with_context(|| format!("input {name}: {}",
                    vork_scene::diagnostics::unknown("token", token, tokens.iter().map(String::as_str))))?;
                words[n / 32] |= 1 << (n % 32);
                words[field.offset] = index as u32;
                continue;
            }
            if let Some(range) = &field.range {
                let scalar = |value: &Value| -> Result<()> {
                    let number = value.as_f64().context("ranged input must be numeric")?;
                    ensure!(
                        number.is_finite()
                            && if range.min_closed {
                                number >= range.min
                            } else {
                                number > range.min
                            }
                            && if range.max_closed {
                                number <= range.max
                            } else {
                                number < range.max
                            },
                        "input {name}: received {value}; expected {}{}, {}{} (InputRange)",
                        if range.min_closed { "[" } else { "(" }, range.min, range.max,
                        if range.max_closed { "]" } else { ")" }
                    );
                    Ok(())
                };
                if let Some(values) = value.as_array() {
                    for value in values {
                        scalar(value)?;
                    }
                } else {
                    scalar(value)?;
                }
            }
            words[n / 32] |= 1 << (n % 32);
            let out = &mut words[field.offset..field.offset + field.ty.words()];
            match field.ty {
                Type::Float => out[0] = float(value, name)?.to_bits(),
                Type::Vector(w) => {
                    let a = value
                        .as_array()
                        .filter(|a| a.len() == w as usize)
                        .with_context(|| format!("input {name}: expected {w} numbers"))?;
                    for (dst, v) in out.iter_mut().zip(a) {
                        *dst = float(v, name)?.to_bits();
                    }
                }
                Type::Bool => {
                    out[0] = value
                        .as_bool()
                        .with_context(|| format!("input {name}: expected bool"))?
                        as u32
                }
                Type::Int => {
                    out[0] = i32::try_from(
                        value
                            .as_i64()
                            .or_else(|| {
                                value
                                    .as_f64()
                                    .filter(|v| {
                                        v.fract() == 0.
                                            && *v >= i64::MIN as f64
                                            && *v < i64::MAX as f64
                                    })
                                    .map(|v| v as i64)
                            })
                            .with_context(|| format!("input {name}: expected integer"))?,
                    )
                    .with_context(|| format!("input {name}: outside int32"))?
                        as u32
                }
                Type::Uint => {
                    out[0] = u32::try_from(
                        value
                            .as_u64()
                            .or_else(|| {
                                value
                                    .as_f64()
                                    .filter(|v| v.fract() == 0. && *v >= 0. && *v < u64::MAX as f64)
                                    .map(|v| v as u64)
                            })
                            .with_context(|| format!("input {name}: expected unsigned integer"))?,
                    )
                    .with_context(|| format!("input {name}: outside uint32"))?
                }
                Type::Geometry | Type::Image | Type::FloatGrid | Type::VectorGrid | Type::Aov | Type::Attribute(_) => out[0] = image(name, value)?,
            }
        }
        for field in &self.fields {
            ensure!(!matches!(field.ty,Type::Attribute(_)) || values.contains_key(&field.name),"input {}: Attribute requires an attribute binding",field.name);
            ensure!(field.ty!=Type::Geometry || values.contains_key(&field.name),"input {}: Geometry requires {{geometry: mesh-node-name}}",field.name);
        }
        Ok(words)
    }
}
fn float(value: &Value, name: &str) -> Result<f32> {
    let f = value
        .as_f64()
        .with_context(|| format!("input {name}: expected number"))? as f32;
    ensure!(f.is_finite(), "input {name}: outside finite float32 range");
    Ok(f)
}

fn checked_blend_wrapper(wrapper: String, capacity: u32) -> String {
    wrapper.replace("vork_lobe(surface(ctx, inputs),", &format!("vork_checked_lobe(surface(ctx, inputs), {capacity}u,"))
}

#[cfg(all(test, target_os = "linux"))]
mod lifecycle_tests {
    use std::{io::Write, process::{Command, Stdio}, time::{Duration, Instant}};
    const CHILD: &str = "VORK_TEST_COMPILER_PARENT";

    fn long_compile() -> std::process::Child {
        super::spawn_compiler(Command::new("/bin/sleep").arg("60").stdout(Stdio::null())).unwrap()
    }

    /// Field `n` of /proc/<pid>/stat after the command name (0 = state, 2 = pgrp).
    fn stat_field(pid: u32, n: usize) -> Option<String> {
        let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        stat[stat.rfind(')')? + 2..].split_whitespace().nth(n).map(str::to_owned)
    }

    fn assert_dies(pid: u32, case: &str) {
        let deadline = Instant::now() + Duration::from_secs(5);
        // A dead orphan can briefly remain a zombie until init reaps it.
        while stat_field(pid, 0).is_some_and(|state| state != "Z") {
            if Instant::now() >= deadline {
                // SAFETY: this is the private compiler child created by this test.
                unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL); }
                panic!("compiler outlived its parent ({case})");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Re-run only `test` (its path below the crate) in a fresh test process
    /// with CHILD set; `None` when already in that process.
    fn in_child(test: &str) -> Option<Command> {
        std::env::var_os(CHILD).is_none().then(|| {
            let mut command = Command::new(std::env::current_exe().unwrap());
            // Exercise the same libtest output mode as the CI scheduler, even
            // when the outer test runner uses its default parallelism.
            command.args(["--exact", test, "--nocapture", "--test-threads=1"]).env(CHILD, "1").stdout(Stdio::piped());
            command
        })
    }

    /// Read the child's stdout up to `prefix`, returning the rest of that line.
    /// A missing line means the child ran nothing (for example a renamed test).
    fn read_marker(child: &mut std::process::Child, prefix: &str) -> String {
        let mut stdout = std::io::BufReader::new(child.stdout.take().unwrap());
        let prefix = prefix.to_owned();
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut line = String::new();
            while std::io::BufRead::read_line(&mut stdout, &mut line).is_ok_and(|n| n > 0) {
                // Single-threaded libtest writes `test name ... ` without a
                // newline before nocapture output. The marker can follow it.
                if let Some((_, rest)) = line.trim().split_once(&prefix) {
                    let _ = send.send(rest.to_owned());
                    // Keep draining so later output cannot hit a closed pipe.
                    let _ = std::io::copy(&mut stdout, &mut std::io::sink());
                    return;
                }
                line.clear();
            }
        });
        match receive.recv_timeout(Duration::from_secs(10)) {
            Ok(marker) => marker,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("child marker unavailable: {error}");
            }
        }
    }

    /// Run as a renderer that starts a long compile, then either exits without
    /// dropping it (the CLI's fast exit) or is SIGKILLed.
    fn assert_compiler_dies_with_parent(test: &str, sigkill: bool) {
        let Some(mut command) = in_child(test) else {
            let child = long_compile();
            println!("compiler-pid:{}", child.id());
            std::io::stdout().flush().unwrap();
            if sigkill { loop { std::thread::sleep(Duration::from_secs(1)) } }
            std::process::exit(0);
        };
        let mut parent = command.spawn().unwrap();
        let pid: u32 = read_marker(&mut parent, "compiler-pid:").parse().unwrap();
        if sigkill { parent.kill().unwrap(); }
        let status = parent.wait().unwrap();
        assert!(sigkill || status.success());
        assert_dies(pid, if sigkill { "SIGKILL" } else { "exit" });
    }

    #[test]
    fn compiler_dies_when_parent_exits() {
        assert_compiler_dies_with_parent("lifecycle_tests::compiler_dies_when_parent_exits", false);
    }

    #[test]
    fn compiler_dies_when_parent_is_killed() {
        assert_compiler_dies_with_parent("lifecycle_tests::compiler_dies_when_parent_is_killed", true);
    }

    /// A guard killed from outside takes its compilers with it when replaced.
    /// Runs in its own process: killing the guard here would kill the compilers
    /// of tests running alongside in this one.
    #[test]
    fn replacing_a_killed_guard_kills_its_compilers() {
        if let Some(mut command) = in_child("lifecycle_tests::replacing_a_killed_guard_kills_its_compilers") {
            let mut child = command.spawn().unwrap();
            read_marker(&mut child, "replaced");
            assert!(child.wait().unwrap().success());
            return;
        }
        let first = long_compile();
        let group: libc::pid_t = stat_field(first.id(), 2).unwrap().parse().unwrap();
        // SAFETY: the guard is a private child of this test process.
        unsafe { libc::kill(group, libc::SIGKILL); }
        while stat_field(group as u32, 0).is_some_and(|state| state != "Z") {
            std::thread::sleep(Duration::from_millis(10));
        }
        let mut second = long_compile();
        assert_ne!(stat_field(second.id(), 2).unwrap(), group.to_string());
        assert_dies(first.id(), "guard replaced");
        second.kill().unwrap();
        second.wait().unwrap();
        println!("replaced");
    }
}

#[cfg(test)]
mod range_diagnostic_tests {
    use super::*;
    #[test]
    fn ranges_report_values_and_endpoint_kinds() {
        for (min_closed, max_closed) in [(true,true),(true,false),(false,true),(false,false)] {
            let mut layout = InputLayout { words: 4, fields: vec![InputField {
                name: "size".into(), ty: Type::Vector(3), offset: 1,
                enumeration: None,
                range: Some(InputRange { min: 0., max: 1., min_closed, max_closed }),
            }] };
            for value in [-1., 0., 0.5, 1., 6.] {
                let values = BTreeMap::from([("size".into(), serde_json::json!([0.5,value,0.5]))]);
                let result = layout.pack(&values, |_,_| unreachable!());
                let valid = (value > 0. || min_closed && value == 0.) && (value < 1. || max_closed && value == 1.);
                assert_eq!(result.is_ok(), valid);
                if let Err(error) = result {
                    let text = error.to_string();
                    assert!(text.contains(&format!("received {}", values["size"][1])), "{text}");
                    assert!(text.contains(&format!("{}0, 1{}", if min_closed {"["} else {"("}, if max_closed {"]"} else {")"})), "{text}");
                }
            }
            layout.fields[0].ty = Type::Float;
            let error = layout.pack(&BTreeMap::from([("size".into(), serde_json::json!(6))]), |_,_| unreachable!()).unwrap_err().to_string();
            assert!(error.contains("received 6"), "{error}");
        }
    }
}

#[cfg(test)]
mod texture_program_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang toolchain"]
    fn transformed_texture_normals_compile_with_geometric_derivatives() -> Result<()> {
        let source=r#"import vork; import std.texture;
struct Inputs { Image image; float strength=1.; };
Closure surface(ShadingContext ctx,Inputs p) {
    let t=float2x2(0.,2.,-3.,0.);
    let f=texture_footprint(mul(ctx.TEXCOORD_1,t)+float2(.1,.2),mul(ctx.duv_dx[1],t),mul(ctx.duv_dy[1],t),ctx);
    Closure c={};float2 variance;
    c.geometry_normal=texture_normal(p.image,f,ctx,float3(p.strength,p.strength,1.),true,variance);
    c.geometry_tangent=texture_tangent(p.image,f,ctx,float3(1.),true);
    c.base_color=texture_sample(p.image,f,ctx).rgb;
    c.specular_roughness=filtered_ggx_roughness(.3,variance);
    return c;
}"#;
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let program=Compiler::discover().compile_surface(&abi,source,Path::new("texture-test.slang"),"texturetest")?;
        assert!(program.inputs.fields.iter().any(|f|f.name=="image" && f.ty==Type::Image));
        Ok(())
    }
}
