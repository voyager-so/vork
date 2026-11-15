//! Numeric environment radiance stage, sharing reflection, diagnostics and cache ABI.
use super::*;
#[derive(Debug, Serialize, Deserialize)]
pub struct EnvironmentProgram {
    pub wgsl: String,
    pub entry: String,
    pub animated: bool,
    pub spatial: bool,
    pub slang: String,
    pub include: Vec<PathBuf>,
    pub inputs: InputLayout,
    pub dependencies: Vec<PathBuf>,
    pub warnings: String,
}
impl EnvironmentProgram {
    /// Prove camera radiance independent of the complete camera context. The
    /// caller must separately exclude changing inputs and certify image planes.
    pub fn constant_camera(&self, uniform_images: bool) -> Result<bool> {
        effects::environment_constant_camera(&self.wgsl, &self.entry, uniform_images)
    }
}
impl Compiler {
    pub fn compile_environment(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
    ) -> Result<EnvironmentProgram> {
        let compiled=self.compile(abi,source,origin,tag,false,|inputs,declarations| {
            ensure!(declarations.contains("environment"),"program requires float3 environment(EnvironmentContext ctx, Inputs inputs)");
            ensure!(inputs.fields.iter().all(|f|!matches!(f.ty,Type::Geometry | Type::Attribute(_)|Type::Aov|Type::FloatGrid|Type::VectorGrid)),"environment program inputs must be numeric or Image");
            let decode=inputs.slang_decode();
            Ok(format!("[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}\nexport void vork_environment_{tag}(float3 origin,float3 direction,float time,uint base,out float4 r[2]) {{\n{decode}\nr[0]=float4(environment(EnvironmentContext(origin,direction,time,false),inputs),1);\nr[1]=float4(environment(EnvironmentContext(origin,direction,time,true),inputs),1);\n}}\n"))
        })?;
        let entry = format!("vork_environment_{tag}_0_{tag}");
        ensure!(
            compiled.output.wgsl.contains(&format!("fn {entry}(")),
            "Slang did not emit {entry}"
        );
        let animated=effects::environment_uses_time(&compiled.output.wgsl,&entry)?;
        Ok(EnvironmentProgram {
            spatial: effects::environment_uses_origin(&compiled.output.wgsl, &entry)?,
            slang: program_text(source, origin),
            include: include_paths(origin),
            animated,
            wgsl: compiled.output.wgsl,
            entry,
            inputs: compiled.inputs,
            dependencies: compiled.output.dependencies,
            warnings: compiled.output.diagnostics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires pinned Slang"]
    fn compiled_environment_constants_preserve_procedural_dependencies() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let compiler=Compiler::discover();
        for (source,uniform_images,expected) in [
            (module_source("std/environment_image").unwrap(),true,true),
            (module_source("std/environment_image").unwrap(),false,false),
            ("struct Inputs {float gain=1.;}; float3 environment(EnvironmentContext c,Inputs i){return float3(i.gain);}",false,true),
            ("struct Inputs {Image image;}; float3 environment(EnvironmentContext c,Inputs i){return sample(i.image,c.direction.xy,c).rgb+float3(c.direction.x>0.?1.:0.);}",true,false),
            ("struct Inputs {}; float3 environment(EnvironmentContext c,Inputs i){return c.camera?float3(2.,4.,8.):c.direction;}",false,true),
            ("struct Inputs {}; float3 environment(EnvironmentContext c,Inputs i){return float3(c.time);}",true,false),
        ] {
            let program=compiler.compile_environment(&abi,source,Path::new("constant-environment.slang"),"constant")?;
            assert_eq!(program.constant_camera(uniform_images)?,expected,"{source}\n{}",program.wgsl);
        }
        Ok(())
    }
}
