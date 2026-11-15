//! Read numeric Inputs initializers through the compiler and its CPU evaluator.
use super::*;

impl Compiler {
    /// No shader stage executes: these accessors use the same Inputs construction
    /// as rendering. Unsupported/resource values remain unavailable individually.
    pub fn input_defaults(
        &self,
        abi: &Abi,
        source: &str,
        origin: &Path,
    ) -> Result<BTreeMap<String, Option<Value>>> {
        let tag = "inputdefaults";
        let compiled = self.compile(abi, source, origin, tag, false, |inputs, _| {
            let mut wrapper =
                "[shader(\"compute\")][numthreads(1,1,1)] void vork_entry() {}\n".to_owned();
            let decode = inputs.slang_decode();
            for (index, field) in inputs.fields.iter().enumerate() {
                if !matches!(
                    field.ty,
                    Type::Float | Type::Int | Type::Uint | Type::Bool | Type::Vector(_)
                ) {
                    continue;
                }
                let (ty, value) = if let Some(enumeration) = &field.enumeration {
                    let mut value = String::from("-1");
                    for (i, case) in enumeration.cases.iter().enumerate().rev() {
                        value = format!(
                            "inputs.{} == {}::{} ? {} : ({})",
                            field.name, enumeration.ty, case, i, value
                        );
                    }
                    (String::from("int"), value)
                } else {
                    (field.ty.slang(), format!("inputs.{}", field.name))
                };
                writeln!(
                    wrapper,
                    "export {ty} vork_input{index}_{tag}(uint base) {{ {decode} return {value}; }}"
                )?;
            }
            Ok(wrapper)
        })?;
        let module = validated_module(&compiled.output.wgsl, abi)?;
        let words = vec![0; compiled.inputs.words];
        let mut result = BTreeMap::new();
        for (index, field) in compiled.inputs.fields.iter().enumerate() {
            let entry = format!("vork_input{index}_{tag}_0_{tag}");
            let mut value = effects::input_value(&module, &entry, &words);
            if let Some(enumeration) = &field.enumeration {
                value = value
                    .and_then(|v| v.as_u64())
                    .and_then(|i| enumeration.cases.get(i as usize))
                    .map(|token| Value::String(token.to_ascii_lowercase()));
            }
            result.insert(field.name.clone(), value);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Slang"]
    fn typed_initializers_use_compiled_expressions_and_enum_identity() -> Result<()> {
        let abi = Abi {
            aov_count: 0,
            context: Abi::spec_context(),
            closure: openpbr_fields(),
        };
        let source = "import vork; float amount() {return 3.5;} enum Mode { First=17, Second=-3 }; struct Inputs { float value=amount(); float3 color=float3(.25,.5,.75); bool enabled=true; int signed_value=-7; uint large=4294967295u; Mode mode=Mode::Second; float zero; Image image; };";
        let values =
            Compiler::discover().input_defaults(&abi, source, Path::new("input-values.slang"))?;
        assert_eq!(values["value"], Some(serde_json::json!(3.5)));
        assert_eq!(values["color"], Some(serde_json::json!([0.25, 0.5, 0.75])));
        assert_eq!(values["enabled"], Some(serde_json::json!(true)));
        assert_eq!(values["signed_value"], Some(serde_json::json!(-7)));
        assert_eq!(values["large"], Some(serde_json::json!(4294967295u32)));
        assert_eq!(values["mode"], Some(serde_json::json!("second")));
        assert_eq!(values["zero"].as_ref().and_then(Value::as_f64), Some(0.));
        assert_eq!(values["image"], None);
        Ok(())
    }
}
