//! Output-image stage using the existing reflected numeric-input ABI.
use super::*;
#[derive(Debug, Serialize, Deserialize)]
pub struct CompositeProgram {
    pub wgsl: String,
    pub entry: String,
    pub inputs: InputLayout,
    pub dependencies: Vec<PathBuf>,
    pub warnings: String,
    /// Image slots read by the lowered program; None means a dynamic slot.
    pub image_reads: Option<Vec<u32>>,
}
impl Compiler {
    pub fn compile_composite(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
        tag: &str,
    ) -> Result<CompositeProgram> {
        self.compile_composite_internal(abi,source,origin,tag,None)
    }
    pub fn compile_composite_bound(&self,abi:&Abi,source:&str,origin:&Path,tag:&str,values:&BTreeMap<String,Value>)->Result<CompositeProgram> {
        self.compile_composite_internal(abi,source,origin,tag,Some(values))
    }
    fn compile_composite_internal(&self,abi:&Abi,source:&str,origin:&Path,tag:&str,values:Option<&BTreeMap<String,Value>>)->Result<CompositeProgram> {
        let compiled = self.compile_with_capacity(abi, source, origin, tag, None, false, true, false, |inputs, declarations| {
            ensure!(declarations.contains("composite"), "program requires float4 composite(CompositeContext ctx, Inputs inputs)");
            ensure!(inputs.fields.iter().all(|f| !matches!(f.ty, Type::Geometry | Type::Attribute(_) | Type::Aov | Type::FloatGrid | Type::VectorGrid)), "compositor inputs must be numeric or Image");
            let mut images=BTreeMap::new();
            let mut channels=Vec::<String>::new();
            if let Some(values)=values {
                inputs.pack(values,|name,value| {
                    // Binding identity, as the renderer keys it: a channel token, "aov:<name>" or "image:<name>".
                    use vork_scene::v2::Input;
                    let channel=match serde_json::from_value::<Input>(value.clone()) {
                        Ok(Input::Texture(t))=>format!("image:{}",t.image),
                        Ok(Input::Step(n))=>format!("step:{}",n.step),
                        Ok(Input::Channel(c))=>c.channel.token().to_owned(),
                        Ok(Input::Aov(a))=>format!("aov:{}",a.aov),
                        _=>anyhow::bail!("compositor input {name}: expected {{\"step\": earlier name}}, {{\"channel\": ...}}, {{\"aov\": ...}} or {{\"image\": ...}}; got {value}"),
                    };
                    let channel=channel.as_str();
                    let slot=if let Some(index)=channels.iter().position(|c|c==channel) {index as u32+1} else {channels.push(channel.to_owned());channels.len() as u32};
                    images.insert(name.to_owned(),slot);Ok(slot)
                })?;
            }
            let decode = inputs.slang_decode_images(&images);
            Ok(format!("[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {{}}\nexport void vork_composite_{tag}(float2 uv,float2 size,float time,uint base,out float4 r[1]) {{\n{decode}\nr[0]=composite(vork_composite_context(uv,size,time),inputs);\n}}"))
        })?;
        let entry = format!("vork_composite_{tag}_0_{tag}");
        ensure!(
            compiled.output.wgsl.contains(&format!("fn {entry}(")),
            "Slang did not emit {entry}"
        );
        let image_reads = effects::composite_reads(&compiled.output.wgsl, &entry, abi)?;
        Ok(CompositeProgram {
            image_reads,
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
    fn image_reads_follow_helpers_branches_and_dynamic_addresses() -> Result<()> {
        let abi=Abi {aov_count:0,context:Abi::spec_context(),closure:openpbr_fields()};
        for (source,expected) in [
            ("fn root()->vec4<f32>{return vec4(1.);}",Some(vec![])),
            ("fn helper(s:u32)->vec4<f32>{return vork_compositor_load(s,vec2(0));} fn root()->vec4<f32>{return helper(2u);}",Some(vec![2])),
            ("fn root(b:bool)->vec4<f32>{if b {return vork_compositor_load(0u,vec2(0));}return vork_compositor_load(3u,vec2(0));}",Some(vec![0,3])),
            ("fn root(s:u32)->vec4<f32>{return vork_compositor_load(s,vec2(0));}",None),
            ("fn root()->vec4<f32>{let extent=vork_compositor_size(0u);return vec4(extent,0.,1.);}",Some(vec![])),
        ] {assert_eq!(effects::composite_reads(source,"root",&abi)?,expected);}
        Ok(())
    }
    #[test]
    #[ignore = "requires pinned Slang"]
    fn reflected_composite_reads_image_and_defaults() -> Result<()> {
        let abi = Abi { aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let program = Compiler::discover().compile_composite(&abi,
            "struct Inputs { float gain=1.; }; float4 composite(CompositeContext ctx, Inputs inputs) { float2 uv=ctx.uv; float2 size=ctx.size; float time=ctx.time;  return compositor_sample(uv)*inputs.gain; }",
            Path::new("composite-test.slang"), "test")?;
        let channels = Compiler::discover().compile_composite(&abi,
            "struct Inputs { Image depth; Image color; }; float4 composite(CompositeContext ctx, Inputs inputs) { return compositor_load(inputs.depth,int2(ctx.uv*ctx.size)) + compositor_sample(inputs.color,ctx.uv); }",
            Path::new("composite-channels.slang"), "channels")?;
        assert!(channels.wgsl.contains("vork_compositor_load("));
        let bound=Compiler::discover().compile_composite_bound(&abi,
            "struct Inputs {Image plate;Image unused;};float4 composite(CompositeContext c,Inputs i){return compositor_sample(i.plate,c.uv);}",
            Path::new("bound-images.slang"),"bound",&BTreeMap::from([
                ("plate".into(),serde_json::json!({"image":"plate"})),
                ("unused".into(),serde_json::json!({"channel":"color"})),
            ]))?;
        assert_eq!(bound.image_reads,Some(vec![1]),"{}",bound.wgsl);
        assert!(
            channels
                .inputs
                .pack(
                    &BTreeMap::from([("depth".into(), serde_json::json!({"channel":"depth"}))]),
                    |_, _| Ok(1)
                )
                .is_ok()
        );
        assert!(program.wgsl.contains(SAMPLE_IMAGE));
        assert_eq!(program.image_reads,Some(vec![0]));
        assert!(
            program
                .inputs
                .pack(&BTreeMap::new(), |_, _| unreachable!())
                .is_ok()
        );
        assert!(
            Compiler::discover()
                .compile_composite(
                    &abi,
                    "struct Inputs {}; float3 environment(EnvironmentContext ctx, Inputs i){ float3 d=ctx.direction; return d;}",
                    Path::new("bad.slang"),
                    "bad"
                )
                .is_err()
        );
        Ok(())
    }
}
