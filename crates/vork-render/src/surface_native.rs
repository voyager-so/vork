//! Built-in surface entry families. Capabilities specialize in SPIR-V; only
//! resource interfaces (ray bindings and probes) select a shipped module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKernel { Evaluate, Indirect, Occupancy, Gi, Reflections, Probes, Directional }
#[allow(dead_code)] // Entry generation is shared with build_slang.rs.
impl SurfaceKernel {
    pub const ALL: [Self;7] = [Self::Evaluate,Self::Indirect,Self::Occupancy,Self::Gi,Self::Reflections,Self::Probes,Self::Directional];
    pub fn name(self) -> &'static str {
        match self {Self::Evaluate=>"evaluate",Self::Indirect=>"indirect",Self::Occupancy=>"occupancy",Self::Gi=>"gi",Self::Reflections=>"reflections",Self::Probes=>"probes",Self::Directional=>"directional"}
    }
    pub fn camera(self) -> bool {matches!(self,Self::Evaluate|Self::Indirect|Self::Occupancy)}
    pub fn source(self) -> String {
        let (module,threads,parameters,call)=match self {
            Self::Evaluate=>("decoupled","64,1,1","uint3 gid:SV_DispatchThreadID","evaluate_material<Key,NoPrograms,true,NoPrograms>(gid);"),
            Self::Indirect=>("decoupled","REG_CAMERA_INDIRECT_GROUP,1,1","uint3 gid:SV_DispatchThreadID","indirect_material<Key,NoPrograms>(gid);"),
            Self::Occupancy=>("decoupled","8,8,1","uint3 gid:SV_DispatchThreadID","prepare_refraction_occupancy<Key,NoPrograms>(gid);"),
            Self::Gi=>("gi_ray","REG_COMPUTE_TILE,REG_COMPUTE_TILE,1","uint3 gid:SV_DispatchThreadID","gi_pixel<Key,NoPrograms>(gid);"),
            Self::Reflections=>("reflections_hits","REG_COMPUTE_TILE*REG_COMPUTE_TILE,1,1","uint3 gid:SV_DispatchThreadID","reflections_pixel<Key,NoPrograms>(gid);"),
            Self::Probes=>("probe_update","REG_PROBE_GROUP,1,1","uint3 group:SV_GroupID,uint thread:SV_GroupIndex","probe_trace<Key,NoPrograms>(group,thread);"),
            Self::Directional=>("probe_update","REG_PROBE_GROUP,1,1","uint3 group:SV_GroupID,uint thread:SV_GroupIndex","probe_directional_trace<Key,NoPrograms>(group,thread);"),
        };
        format!("import vork_constants;import frame;import surface;import {module};\n[shader(\"compute\")][numthreads({threads})] void surface_main({parameters}) {{{call}}}\n")
    }
}

/// The prebuilt path engine: path transport over the material table with every
/// capability (`Unspecialized`) and no scene code, compiled once at build for
/// both backends. A family is what a capability cannot express, because it
/// changes the kernel's resources or stored state: adaptive sampling and the
/// inline dielectric boundary capacity. Each family is one driver pipeline for
/// every scene it draws, so setup can create it before any scene exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathEngine { pub adaptive: bool, pub boundaries: u32 }
#[allow(dead_code)] // Entry generation is shared with build_slang.rs.
impl PathEngine {
    /// Boundary capacities with a prebuilt family: no solid, or one at a time.
    pub const BOUNDARIES: u32 = 1;
    pub const ALL: [Self;4] = [
        Self {adaptive:false,boundaries:0}, Self {adaptive:false,boundaries:1},
        Self {adaptive:true,boundaries:0}, Self {adaptive:true,boundaries:1},
    ];
    pub const ENTRY: &'static str = "path_engine";
    /// The engine keeps every scene proof general: table containment, connection
    /// transmission, weighted IOR and glTF tint all stay, so no proof selects a
    /// family. What it leaves out is what no table-material triangle scene has:
    /// implicit fields, points, AOVs (and Gaussians, which no shipped entry has).
    pub const DEFINES: [&'static str;6] = ["VORK_NO_IMPLICIT","VORK_AOV_COUNT=0","VORK_PATH_SLANG_OUTLINE","VORK_NO_POINTS","VORK_LIGHT_PATHS","VORK_PROGRAM_COVERAGE"];
    pub fn name(self) -> String { format!("path_engine_{}_{}",u8::from(self.adaptive),self.boundaries) }
    pub fn source(self) -> String {
        format!("import vork_constants;import frame;import variant;import surface;import path;\n[shader(\"compute\")][numthreads(REG_COMPUTE_TILE,REG_COMPUTE_TILE,1)] void {}(uint3 gid:SV_DispatchThreadID) {{path_pixel<{},false,Unspecialized,NoPrograms,{},false>(gid);}}\n",
            Self::ENTRY,self.adaptive,self.boundaries)
    }
}
