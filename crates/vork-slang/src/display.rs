//! Display stage: scene-linear colour to display-encoded RGB, sharing reflection,
//! diagnostics and the cache ABI. The renderer evaluates it per output pixel.
use super::*;
#[derive(Debug, Serialize, Deserialize)]
pub struct DisplayProgram {
    pub wgsl: String,
    pub entry: String,
    pub inputs: InputLayout,
    pub dependencies: Vec<PathBuf>,
    pub warnings: String,
}
impl Compiler {
    /// The compiled entry returns the view's result in one vector: its value, and in `w` its form, zero for
    /// absolute XYZ D65 in cd/m² or one plus the `DisplayTarget` its code values are for.
    pub fn compile_display(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
    ) -> Result<DisplayProgram> {
        reject_reserved(source, &origin.display().to_string())?;
        let compiled=self.compile(abi,source,origin,tag,false,|inputs,declarations| {
            ensure!(declarations.contains("view"),"program requires ViewResult view(ViewContext ctx, Inputs inputs): view_from_xyz(XYZ D65 in cd/m²) or view_from_codes(codes, display)");
            ensure!(inputs.fields.iter().all(|f|!matches!(f.ty,Type::Geometry | Type::Attribute(_)|Type::Aov|Type::FloatGrid|Type::VectorGrid)),"display program inputs must be numeric or Image");
            let decode=inputs.slang_decode();
            Ok(format!("[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}\nexport void vork_display_{tag}(float3 color,uint2 pixel,uint seed,float time,uint color_space,uint domain,uint target,float peak,float white,uint base,out float4 r[1]) {{\n{decode}\nViewResult v=view(ViewContext(color,WorkingColorSpace(color_space),ReferenceDomain(domain),DisplayTarget(target),peak,white,pixel,seed,time),inputs);\nr[0]=float4(v.value,v.codes ? 1.+float(uint(v.display)) : 0.);\n}}\n"))
        })?;
        let entry = format!("vork_display_{tag}_0_{tag}");
        ensure!(
            compiled.output.wgsl.contains(&format!("fn {entry}(")),
            "Slang did not emit {entry}"
        );
        Ok(DisplayProgram {
            wgsl: compiled.output.wgsl,
            entry,
            inputs: compiled.inputs,
            dependencies: compiled.output.dependencies,
            warnings: compiled.output.diagnostics,
        })
    }
}
