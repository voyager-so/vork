//! Medium entries share the surface stage's ABI, reflection and compiler transport.
use super::*;
/// Three float4 bridge vectors carry absorption/anisotropy, scattering, and emission.
pub const MEDIUM_VEC4S: usize = 3;
/// The interval proof follows ordered f32 operations, never fast-math algebra.
pub const MEDIUM_FP_FLAGS: [&str;2] = ["-fp-mode","precise"];
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediumProgram {
    pub native: String,
    pub native_point: String,
    /// Compiler bridge identity; native instances may be renamed independently.
    coefficient_entry: String,
    work_source_stable: bool,
    pub native_includes: Vec<PathBuf>,
    pub wgsl: String,
    /// local position, time, word base, world position, and an output array of MEDIUM_VEC4S vectors.
    pub point: String,
    /// Proven independent of position by compiler dependency analysis.
    pub homogeneous: bool,
    #[serde(skip)]
    pub tail_slab: Option<(usize,f32)>,
    #[serde(skip)]
    pub emission_free: bool,
    #[serde(skip)]
    pub scattering_free: bool,
    #[serde(skip)]
    pub time_dependent: bool,
    #[serde(skip)]
    pub exponential: Option<([f32;3],[f32;3],[f32;3])>,
    pub inputs: InputLayout,
    pub dependencies: Vec<PathBuf>,
    pub warnings: String,
}
impl Compiler {
    pub fn compile_medium(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
    ) -> Result<MediumProgram> {
        let compiled=self.compile(abi,source,origin,tag,true,|inputs,declarations| {
            ensure!(declarations.contains("medium"),"Slang AST probe did not recognize the required medium stage");
            ensure!(inputs.fields.iter().all(|field|field.ty!=Type::Image),"medium image inputs require a volume sampling interface");
            let decode=inputs.slang_decode();
            let entry="[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {}\n";
            // Preserve authored physical coefficients; transport validates them and
            // refuses the frame instead of silently clamping invalid outputs.
            let point=format!("public MediumCoefficients vork_medium_{tag}_point(float3 p,float time,uint base,float3 world_position,float wavelength=0.) {{\n{decode}\nMediumCoefficients m=medium(MediumContext(p,time,world_position,wavelength),inputs);\nreturn m;}}\npublic MediumCoefficients vork_medium_{tag}_coefficients(float3 p,float time,uint base,float3 world_position,float wavelength=0.) {{return vork_medium_{tag}_point(p,time,base,world_position,wavelength);}}\nexport void vork_medium_{tag}(float3 p,float time,uint base,float3 world_position,out float4 r[{MEDIUM_VEC4S}]) {{\nMediumCoefficients m=vork_medium_{tag}_coefficients(p,time,base,world_position);\nr[0]=float4(m.absorption,m.anisotropy);r[1]=float4(m.scattering,0);r[2]=float4(m.emission,0);\n}}\n");
            Ok(format!("{entry}{point}"))
        })?;
        let point = format!("vork_medium_{tag}_0_{tag}");
        let output = compiled.output;
        let stable=|text:&str| !["#","%:","??=","__target_switch","VORK_","SLANG"].iter().any(|needle|text.contains(needle));
        let work_source_stable=stable(source) && output.dependencies.iter().all(|path| std::fs::read_to_string(path).is_ok_and(|text|stable(&text)));
        ensure!(output.wgsl.contains(&format!("fn {point}(")), "Slang did not emit {point}");
        let module = validated_module(&output.wgsl, abi)?;
        let homogeneous = effects::medium_homogeneous(&module, &format!("vork_medium_{tag}_point_0_{tag}"),None)?;
        let native_point = format!("vork_medium_{tag}");
        let native = compiled.native.replace("export void", "public void");
        let current = std::env::current_dir()?;
        Ok(MediumProgram {
            native,
            native_point,
            coefficient_entry: format!("vork_medium_{tag}_point_0_{tag}"),
            work_source_stable,
            native_includes: include_paths(origin)
                .into_iter()
                .map(|p| if p.is_absolute() { p } else { current.join(p) })
                .collect(),
            wgsl: output.wgsl,
            point,
            homogeneous,
            tail_slab: None,
            emission_free: false,
            scattering_free: false,
            time_dependent: true,
            exponential: None,
            inputs: compiled.inputs,
            dependencies: output.dependencies,
            warnings: output.diagnostics,
        })
    }
}

impl MediumProgram {
    /// Prove invariance with static words known and animated numeric words unknown,
    /// without baking values into a separate shader per instance.
    pub fn homogeneous_for_inputs(&self,abi:&Abi,words:&[Option<u32>]) -> Result<bool> {
        if self.homogeneous {return Ok(true);}
        let module=validated_module(&self.wgsl,abi)?;
        let entry=&self.coefficient_entry;
        effects::medium_homogeneous(&module,entry,Some(&words[..self.inputs.words]))
    }
}

impl MediumProgram {
    pub fn time_dependent_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<bool> {
        let module=validated_module(&self.wgsl,abi)?;
        effects::medium_time_dependent(&module,&self.coefficient_entry,&words[..self.inputs.words])
    }
    pub fn scattering_free_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<bool> {
        let module=validated_module(&self.wgsl,abi)?;
        effects::medium_scattering_free(&module,&self.coefficient_entry,&words[..self.inputs.words])
    }
    pub fn emission_free_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<bool> {
        let module=validated_module(&self.wgsl,abi)?;
        let entry=&self.coefficient_entry;
        effects::medium_emission_free(&module,entry,&words[..self.inputs.words])
    }
    /// Identity-space exponential profile for a world binding. Finite carriers
    /// must certify their actual coordinate domain separately.
    pub fn exponential_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<Option<([f32;3],[f32;3],[f32;3])>> {
        let module=validated_module(&self.wgsl,abi)?;
        let entry=&self.coefficient_entry;
        effects::medium_exponential(&module,entry,&words[..self.inputs.words])
    }
    /// Resource-free extinction profile for shipped surface connections.
    pub fn connection_profile_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<Option<([f32;3],[f32;3],[f32;3])>> {
        let module=validated_module(&self.wgsl,abi)?;
        effects::medium_connection_profile(&module,&self.coefficient_entry,words,true)
    }
    /// Compiler-proven eventually constant slab over the known input words.
    pub fn tail_slab_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<Option<(usize,f32)>> {
        let module=validated_module(&self.wgsl,abi)?;
        let entry=&self.coefficient_entry;
        effects::medium_tail_slab(&module,entry,&words[..self.inputs.words])
    }
}

#[cfg(test)]
mod tail_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang"]
    fn scattering_proof_keeps_unknown_and_animated_inputs() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        for expression in ["i.amount", "i.amount*c.position.y", "i.amount*c.time"] {
            let source=format!("struct Inputs {{float amount=0.;}}; MediumCoefficients medium(MediumContext c,Inputs i) {{MediumCoefficients m;m.absorption=float3(.1);m.emission=float3(.2);m.scattering=float3({expression});return m;}}");
            let p=Compiler::discover().compile_medium(&abi,&source,Path::new("scatter-proof.slang"),"scatterproof")?;
            let words=p.inputs.pack(&BTreeMap::new(),|_,_|unreachable!())?;
            let known=words.iter().copied().map(Some).collect::<Vec<_>>();
            assert!(p.scattering_free_for_inputs(&abi,&known)?,"{expression}");
            assert!(!p.scattering_free_for_inputs(&abi,&vec![None;words.len()])?,"{expression}");
            let mut inputs=BTreeMap::new();inputs.insert("amount".into(),serde_json::json!(0.2));
            let positive=p.inputs.pack(&inputs,|_,_|unreachable!())?;
            assert!(!p.scattering_free_for_inputs(&abi,&positive.into_iter().map(Some).collect::<Vec<_>>())?);
        }
        let fog=Compiler::discover().compile_medium(&abi,include_str!("../../vork-scene/tests/fixtures/medium.slang"),Path::new("fog-proof.slang"),"fogproof")?;
        for density in [0.1,1.,10.] {
            let inputs=serde_json::from_value(serde_json::json!({"density":density,"scattering":[0.,0.,0.]}))?;
            let words=fog.inputs.pack(&inputs,|_,_|unreachable!())?;
            assert!(fog.scattering_free_for_inputs(&abi,&words.into_iter().map(Some).collect::<Vec<_>>())?,"fog density {density}");
        }
        for expression in ["0.*exp(c.position.y)", "0.*(1./c.position.y)"] {
            let source=format!("struct Inputs {{}}; MediumCoefficients medium(MediumContext c,Inputs i) {{MediumCoefficients m;m.scattering=float3({expression});return m;}}");
            let p=Compiler::discover().compile_medium(&abi,&source,Path::new("scatter-nonfinite.slang"),"scatternonfinite")?;
            assert!(!p.scattering_free_for_inputs(&abi,&[])?,"{expression}");
        }
        for expression in ["max(c.time,0.)", "max(c.position.x,0.)", "max(c.wavelength,0.)"] {
            let source=format!("struct Inputs {{}}; MediumCoefficients medium(MediumContext c,Inputs i) {{MediumCoefficients m;m.scattering=float3({expression});return m;}}");
            let p=Compiler::discover().compile_medium(&abi,&source,Path::new("scatter-varying.slang"),"scattervarying")?;
            assert!(!p.scattering_free_for_inputs(&abi,&[])?,"{expression}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn medium_clock_dependence_follows_coefficients() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        for (emission,expected) in [("i.energy",false),("i.energy*(1.+sin(c.time))",true),("0.*c.time+i.energy",false)] {
            let source=format!("struct Inputs {{float energy=1.;}}; MediumCoefficients medium(MediumContext c,Inputs i) {{MediumCoefficients m;m.emission=float3({emission});return m;}}");
            let p=Compiler::discover().compile_medium(&abi,&source,Path::new("medium-clock.slang"),"clock")?;
            let words=p.inputs.pack(&BTreeMap::new(),|_,_|unreachable!())?;
            assert_eq!(p.time_dependent_for_inputs(&abi,&words.into_iter().map(Some).collect::<Vec<_>>())?,expected,"{emission}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn height_program_proves_exponential_profile() -> Result<()> {
        let abi=Abi { aov_count: 0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source=include_str!("../tests/fixtures/fog-height.slang");
        let compiler=Compiler::discover();
        let program=compiler.compile_medium(&abi,source,Path::new("height-fog.slang"),"heightfog")?;
        let inputs=serde_json::from_value(serde_json::json!({"falloff":0.5,"absorption":[0.,2.,4.],"scattering":[1.,0.,2.]}))?;
        let words=program.inputs.pack(&inputs,|_,_|unreachable!())?;
        let proof_words=words.iter().copied().map(Some).collect::<Vec<_>>();
        // The finite-domain bound service takes local coordinate bounds.
        let local=compiler.compile_medium(&abi,&source.replace("ctx.world_position.y","ctx.position.y"),Path::new("local-height.slang"),"localheight")?;
        let bound=local.extinction_majorant(&abi,&proof_words,[[-1.,1.];3],[0.,1.])?;
        let expected=6.*0.5f32.exp();
        assert!(bound>=expected && bound<expected*1.0001,"{bound} vs {expected}");
        assert_eq!(program.exponential_for_inputs(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>())?,Some(([0.,2.,4.],[1.,0.,2.],[0.,-0.5,0.])));
        Ok(())
    }
    #[test]
    fn authored_height_fixture_certifies_vacuum_channels() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source=include_str!("../../vork-scene/tests/fixtures/medium.slang");
        let program=Compiler::discover().compile_medium(&abi,source,Path::new("authored-height.slang"),"authoredheight")?;
        for mask in 1..8 {
            let absorption: [f32;3]=std::array::from_fn(|c|if mask&(1<<c)!=0 {1.} else {0.});
            let inputs=serde_json::from_value(serde_json::json!({
                "density":0.2,"height_falloff":0.4,"absorption":absorption,"scattering":[0,0,0]
            }))?;
            let words=program.inputs.pack(&inputs,|_,_|unreachable!())?;
            let known=words.iter().copied().map(Some).collect::<Vec<_>>();
            let expected=Some((absorption.map(|x|x*0.2),[0.;3],[0.,-0.4,0.]));
            assert_eq!(program.exponential_for_inputs(&abi,&known)?,expected,"vacuum channel mask {mask}");
        }
        for (density,coefficients) in [(0.2,[0.;3]),(0.,[1.;3])] {
            let inputs=serde_json::from_value(serde_json::json!({
                "density":density,"height_falloff":0.4,"absorption":coefficients,"scattering":coefficients
            }))?;
            let words=program.inputs.pack(&inputs,|_,_|unreachable!())?;
            assert!(program.homogeneous_for_inputs(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>())?);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned Slang"]
    fn imported_constant_volume_retains_homogeneous_tail() -> Result<()> {
        let abi=Abi { aov_count: 0,context:Abi::spec_context(),closure:openpbr_fields()};
        for intensity in [0.,1.] {
            let source=format!("import blender.volume; struct Inputs {{}}; MediumCoefficients medium(MediumContext ctx,Inputs inputs) {{MediumPoint m;m.density=.05;m.color=float3(.8);m.blackbody_intensity={intensity:.1};return principled_volume(m);}}");
            let program=Compiler::discover().compile_medium(&abi,&source,Path::new("constant-volume.slang"),"constantvolume")?;
            assert!(program.homogeneous_for_inputs(&abi,&[])?,"{}",program.wgsl);
        }
        let source="import blender.volume; struct Inputs {}; MediumCoefficients medium(MediumContext ctx,Inputs inputs) {MediumPoint m;m.density=.05*exp(-ctx.position.y);m.color=float3(.8);return principled_volume(m);}";
        let program=Compiler::discover().compile_medium(&abi,source,Path::new("varying-volume.slang"),"varyingvolume")?;
        assert!(!program.homogeneous_for_inputs(&abi,&[])?);
        assert!(program.emission_free_for_inputs(&abi,&[])?,"{}",program.wgsl);
        for declaration in ["MediumCoefficients m;","MediumCoefficients m={};"] {
            let source=format!("struct Inputs {{}}; MediumCoefficients medium(MediumContext ctx,Inputs inputs) {{{declaration}m.absorption=float3(exp(-ctx.position.y));return m;}}");
            let program=Compiler::discover().compile_medium(&abi,&source,Path::new("default-emission.slang"),"defaultemission")?;
            assert!(!program.homogeneous_for_inputs(&abi,&[])?);
            assert!(program.emission_free_for_inputs(&abi,&[])?,"{}",program.wgsl);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn clamped_height_tail_is_proven_not_probed() -> Result<()> {
        let abi=Abi { aov_count: 0,context:Abi::spec_context(),closure:openpbr_fields()};
        for (profile,expected) in [("exp(clamp(-i.h*c.position.y,-80.,80.))",true),
            ("max(exp(clamp(-i.h*c.position.y,-80.,80.))-exp(-80.),0.)",true),
            ("exp(-i.h*c.position.y)",false),
            ("exp(clamp(-i.h*c.position.y,-80.,80.))+sin(c.position.x)",false)] {
            let source=format!("struct Inputs {{float h=.04;}}; MediumCoefficients medium(MediumContext c,Inputs i) {{MediumCoefficients m={{}};m.absorption=float3({profile});m.emission=float3(1.);return m;}}");
            let program=Compiler::discover().compile_medium(&abi,&source,Path::new("tail-proof.slang"),"tailproof")?;
            let words=program.inputs.pack(&BTreeMap::new(),|_,_|unreachable!())?;
            assert_eq!(program.tail_slab_for_inputs(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>())?.is_some(),expected,"{profile}");
        }
        Ok(())
    }
}

impl MediumProgram {
    /// Conservative complete-expression extinction bound over a finite local
    /// space/time domain. Unknown input words (including animation) cannot be
    /// specialized to their initial values. Unsupported operations are errors.
    pub fn extinction_majorant(&self, abi:&Abi, words:&[Option<u32>], position:[[f32;2];3], times:[f32;2])->Result<f32> {
        let module=validated_module(&self.wgsl,abi)?;
        let entry=&self.coefficient_entry;
        effects::medium_majorant(&module,entry,words,position,times)
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::*;
    #[test]
    #[ignore="requires pinned Slang"]
    fn compiled_certificate_covers_inputs_helpers_time_and_unsampled_peak()->Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="struct Inputs {float amplitude=1000.;}; float peak(float x,float time) {return x>.01 && x<.02 ? 1.+time : 0.;} MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;m.absorption=float3(i.amplitude*peak(c.position.x,c.time));return m;}";
        let p=Compiler::discover().compile_medium(&abi,source,Path::new("certified-peak.slang"),"peak")?;
        let words=p.inputs.pack(&BTreeMap::new(),|_,_|unreachable!())?;
        let bound=p.extinction_majorant(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>(),[[-1.,1.];3],[0.,2.])?;
        assert!(bound>=3000. && bound<3000.01,"{bound}");
        Ok(())
    }
    #[test]
    #[ignore="requires pinned Slang"]
    fn world_position_retains_spatial_dependency()->Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let p=Compiler::discover().compile_medium(&abi,"struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;m.emission=c.world_position;return m;}",Path::new("world-coordinates.slang"),"worldcoordinates")?;
        assert!(!p.homogeneous_for_inputs(&abi,&[])?);
        assert!(p.exponential_for_inputs(&abi,&[])?.is_none());
        Ok(())
    }
}

#[cfg(test)]
mod principled_certificate_tests {
    use super::*;
    #[test]
    #[ignore="requires pinned Slang"]
    fn converted_volume_coefficients_are_certified()->Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="import blender.volume; struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumPoint m={};m.density=.8+.4*c.position.z;return principled_volume(m);}";
        let p=Compiler::discover().compile_medium(&abi,source,Path::new("converted-volume.slang"),"converted")?;
        let bound=p.extinction_majorant(&abi,&[],[[-0.5,0.5];3],[0.,1.])?;
        assert!(bound>=1. && bound<1.00001,"{bound}");
        Ok(())
    }
}

#[cfg(test)]
mod arithmetic_contract_tests {
    use super::*;
    #[test]
    #[ignore="requires pinned Slang; verifies the certificate's native arithmetic contract"]
    fn certified_operations_preserve_order_in_spirv_and_metal()->Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="struct Inputs {float a=16777216.;float b=-16777216.;float c=-1.;float d=-.5;float e=1.5;}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;float sum=(((i.a+i.b)+i.c)+i.d)+i.e;m.absorption=float3(sum+max(c.position.x,0.));return m;}";
        let p=Compiler::discover().compile_medium(&abi,source,Path::new("ordered-medium.slang"),"ordered")?;
        let words=p.inputs.pack(&BTreeMap::new(),|_,_|unreachable!())?;
        assert_eq!(p.extinction_majorant(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>(),[[-1.,0.],[-1.,1.],[-1.,1.]],[0.,1.])?,0.);
        // These runtime words must not be reassociated. All operations in the
        // declared graph are exact, but a different parenthesization can add .5.
        let mut modules=abi.native_modules();
        modules[0].1=abi.module();
        modules.push(("point_program".into(),p.native.clone()));
        modules.push(("ordered_test".into(),format!(r#"
import point_program;
[[vk::binding(0,0)]] StructuredBuffer<uint> words;
[[vk::binding(1,0)]] RWStructuredBuffer<float4> output;
export uint native_material_word(uint i) {{return words[i];}}
[shader("compute")][numthreads(1,1,1)] void main() {{
    float4 values[3];{}(float3(-.5,0.,0.),0.,0u,float3(-.5,0.,0.),values);output[0]=values[0];
}}
"#,p.native_point)));
        let refs=modules.iter().map(|(n,s)|(n.as_str(),s.as_str())).collect::<Vec<_>>();
        let compiler=Compiler::discover();
        let bytes=compiler.compile_modules(&refs,"ordered_test",&[("main","compute")],
            &["-target","spirv","-emit-spirv-directly","-DVORK_NATIVE_MEDIUM","-fp-mode","precise"])?;
        let words=bytes.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let mut protected=std::collections::BTreeSet::new();let mut arithmetic=Vec::new();let mut at=5;
        while at<words.len() {
            let count=(words[at]>>16) as usize;let op=words[at]&0xffff;
            ensure!(count>0 && at+count<=words.len(),"invalid SPIR-V instruction");
            if op==71 && count>=3 && words[at+2]==42 {protected.insert(words[at+1]);}
            if matches!(op,129|131|133|136) {arithmetic.push(words[at+2]);}
            at+=count;
        }
        assert!(arithmetic.len()>=4,"runtime arithmetic was not exercised");
        assert!(arithmetic.iter().all(|id|protected.contains(id)),"certificate arithmetic lacks NoContraction: {arithmetic:?}, protected {protected:?}");
        let msl=compiler.compile_modules(&refs,"ordered_test",&[("main","compute")],
            &["-target","metal","-DVORK_NATIVE_MEDIUM","-fp-mode","precise"])?;
        assert!(msl.starts_with(b"#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n"));
        Ok(())
    }
}

#[cfg(test)]
mod identity_and_grid_tests {
    use super::*;
    #[test]
    #[ignore="requires pinned Slang"]
    fn coefficient_analysis_survives_native_specialization() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let mut p=Compiler::discover().compile_medium(&abi,"struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;m.absorption=float3(exp(-c.position.y));return m;}",Path::new("renamed-medium.slang"),"renamed")?;
        p.native_point.push_str("_exponential_000000000000000000000000000000000000000000000000000000000000000000000000");
        assert!(!p.homogeneous_for_inputs(&abi,&[])?);
        assert!(p.emission_free_for_inputs(&abi,&[])?);
        assert!(p.exponential_for_inputs(&abi,&[])?.is_some());
        assert!(p.tail_slab_for_inputs(&abi,&[])?.is_none());
        let bound=p.extinction_majorant(&abi,&[],[[-1.,1.];3],[0.,1.])?;
        assert!(bound>=1f32.exp() && bound<1f32.exp()*1.00001);
        Ok(())
    }
    #[test]
    #[ignore="requires pinned Slang"]
    fn sparse_grid_analysis_remains_conservative() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="import std.grid; struct Inputs {FloatGrid density;VectorGrid color;}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;float d=max(sample_grid(i.density,c.position),0.);float3 a=max(sample_grid(i.color,c.position),0.);m.scattering=d*a;m.absorption=d*max(1.-a,0.);m.emission=.1*a;return m;}";
        let p=Compiler::discover().compile_medium(&abi,source,Path::new("sparse-medium.slang"),"sparse")?;
        // Grid payloads live beyond the reflected numeric input words.
        let words=vec![Some(128);p.inputs.words];
        assert!(!p.homogeneous_for_inputs(&abi,&words)?);
        assert!(p.exponential_for_inputs(&abi,&words)?.is_none());
        assert!(p.tail_slab_for_inputs(&abi,&words)?.is_none());
        assert!(!p.emission_free_for_inputs(&abi,&words)?);
        Ok(())
    }
}

#[cfg(test)]
mod dense_certificate_tests {
    use super::*;
    #[test]
    #[ignore="requires pinned Slang"]
    fn dense_interpolation_encloses_complete_nonlinear_expression()->Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="import std.grid; struct Inputs {FloatGrid density;}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;float d=sample_grid(i.density,c.position);m.absorption=float3(d*(1.-d));return m;}";
        let p=Compiler::discover().compile_medium(&abi,source,Path::new("dense-proof.slang"),"denseproof")?;
        let mut words=p.inputs.pack(&BTreeMap::from([("density".into(),serde_json::json!({"grid":"d"}))]),|_,_|Ok(p.inputs.words as u32))?;
        words.extend([u32::MAX,2,2,2,0f32.to_bits(),1f32.to_bits()]);words.extend([0;8]);
        let proof=words.iter().copied().map(Some).collect::<Vec<_>>();
        assert!(!p.homogeneous_for_inputs(&abi,&proof)?);
        let bound=p.extinction_majorant(&abi,&proof,[[-1.,1.];3],[0.,1.])?;
        assert!(bound>=0.25 && bound<1.001,"{bound}");
        // Replacing a grid by its maximum would incorrectly certify zero.
        let mut corrupt=proof;corrupt[p.inputs.words+5]=Some(f32::NAN.to_bits());
        assert!(p.extinction_majorant(&abi,&corrupt,[[-1.,1.];3],[0.,1.]).is_err());
        Ok(())
    }
}

impl MediumProgram {
    pub fn needs_spectral(&self, abi: &Abi, words: &[Option<u32>]) -> Result<bool> {
        let entry=&self.coefficient_entry;
        effects::scalar_argument_spectral(&validated_module(&self.wgsl,abi)?,entry,4,2,words)
    }
}

#[cfg(test)]
mod renamed_native_tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang"]
    fn native_alias_does_not_change_analysis_entry() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        let source="struct Inputs {}; MediumCoefficients medium(MediumContext c,Inputs i) {MediumCoefficients m;m.absorption=float3(.5+clamp(c.position.y,0.,1.));return m;}";
        let mut p=Compiler::discover().compile_medium(&abi,source,Path::new("alias-medium.slang"),"alias")?;
        p.native_point=format!("{}_{}",p.native_point,"installed".repeat(20));
        assert!(!p.homogeneous_for_inputs(&abi,&[])?);
        assert!(p.emission_free_for_inputs(&abi,&[])?);
        assert!(!p.needs_spectral(&abi,&[])?);
        assert!(p.extinction_majorant(&abi,&[],[[-1.,1.];3],[0.,1.])? >= 1.5);
        Ok(())
    }
}

#[cfg(test)]
mod connection_profile_tests {
    use super::*;
    #[test]
    #[ignore="requires pinned Slang"]
    fn immutable_connection_profiles_distinguish_constants_exponentials_and_unknowns() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        for (factor,gradient,accepted) in [("1.",[0.,0.,0.],true),("exp(-.5*c.position.y)",[0.,-0.5,0.],true),("1.+.1*sin(c.position.y)",[0.;3],false)] {
            let body=if factor=="1." {"m.absorption=float3(.2);m.scattering=float3(.3);".to_owned()} else {format!("let density=({factor})*i.amount;m.absorption=.2*float3(density);m.scattering=.3*float3(density);")};
            let source=format!("struct Inputs {{float amount=1.;}}; MediumCoefficients medium(MediumContext c,Inputs i) {{MediumCoefficients m={{}};{body}return m;}}");
            let p=Compiler::discover().compile_medium(&abi,&source,Path::new("connection-profile.slang"),"connectionprofile")?;
            let words=p.inputs.pack(&BTreeMap::new(),|_,_|unreachable!())?;
            let known=words.iter().copied().map(Some).collect::<Vec<_>>();
            let profile=p.connection_profile_for_inputs(&abi,&known)?;
            assert_eq!(profile.is_some(),accepted,"{factor}");
            if let Some((a,b,g))=profile {assert_eq!(a,[0.2;3]);assert_eq!(b,[0.3;3]);assert_eq!(g,gradient);}
            if factor!="1." {assert!(p.connection_profile_for_inputs(&abi,&vec![None;words.len()])?.is_none(),"animated amplitude must remain linked");}
        }
        let fog=Compiler::discover().compile_medium(&abi,include_str!("../../vork-scene/tests/fixtures/medium.slang"),Path::new("fog-connection.slang"),"fogconnection")?;
        for height in [0.,0.5] {
            let inputs=serde_json::from_value(serde_json::json!({"density":0.7,"absorption":[0.2,0.2,0.2],"scattering":[0.3,0.3,0.3],"height_falloff":height}))?;
            let words=fog.inputs.pack(&inputs,|_,_|unreachable!())?;
            let (a,b,g)=fog.connection_profile_for_inputs(&abi,&words.iter().copied().map(Some).collect::<Vec<_>>())?.context("standard fog must retain its shared connection")?;
            assert_eq!(a,[0.2_f32*0.7;3]);assert_eq!(b,[0.3_f32*0.7;3]);assert_eq!(g,[0.,-height,0.]);
        }
        Ok(())
    }
}

impl MediumProgram {
    /// Conservative callback instruction count, separate from coefficient bounds.
    pub fn bounded_work_for_inputs(&self,abi:&Abi,words:&[Option<u32>])->Result<u32> {
        ensure!(self.work_source_stable,"medium work proof refuses target-dependent preprocessing or unreadable imports");
        ensure!(self.inputs.fields.iter().all(|field|!matches!(field.ty,Type::Geometry|Type::Image|Type::FloatGrid|Type::VectorGrid)),"medium work proof does not yet admit resource inputs");
        let module=validated_module(&self.wgsl,abi)?;
        effects::medium_work(&module,&self.coefficient_entry,words)
    }
}
