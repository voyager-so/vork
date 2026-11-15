//! Binding descriptions shared by host layouts and generated Slang modules.
//! Native reflection verifies this contract; it never supplies declarations.
use std::fmt::Write;

#[derive(Clone, Copy, Debug)]
pub enum Resource {
    ReadBuffer(&'static str),
    WriteBuffer(&'static str),
    Textures,
    Samplers,
}
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub name: &'static str,
    pub define: &'static str,
    pub resource: Resource,
}
// Binding number is the array index. All five resources occupy one Metal buffer
// slot, including the two argument buffers. Never count populated array entries.
pub const MATERIAL: [Binding; 5] = [
    Binding {
        name: "material_records",
        define: "VORK_MATERIAL_RECORDS_REGISTER",
        resource: Resource::ReadBuffer("MaterialRecord"),
    },
    Binding {
        name: "material_textures",
        define: "VORK_MATERIAL_TEXTURES_REGISTER",
        resource: Resource::Textures,
    },
    Binding {
        name: "material_samplers",
        define: "VORK_MATERIAL_SAMPLERS_REGISTER",
        resource: Resource::Samplers,
    },
    Binding {
        name: "material_words",
        define: "VORK_MATERIAL_WORDS_REGISTER",
        resource: Resource::ReadBuffer("uint"),
    },
    Binding {
        name: "shader_errors",
        define: "VORK_SHADER_ERRORS_REGISTER",
        resource: Resource::WriteBuffer("uint"),
    },
];
impl Binding {
    pub fn register(self, slot: u32) -> String {
        let prefix = match self.resource {
            Resource::ReadBuffer(_) => 't',
            Resource::WriteBuffer(_) => 'u',
            _ => 'b',
        };
        format!("{prefix}{slot}")
    }
}
pub fn material_registers(first: u32) -> Vec<String> {
    MATERIAL
        .iter()
        .enumerate()
        .map(|(i, b)| format!("-D{}={}", b.define, b.register(first + i as u32)))
        .collect()
}
pub fn material_source() -> String {
    let mut source = String::from("import vork_constants;\nimport material_layout;\n");
    for (binding, b) in MATERIAL.iter().enumerate() {
        writeln!(
            source,
            "#ifndef {}\n#define {} {}\n#endif",
            b.define,
            b.define,
            b.register(6 + binding as u32)
        )
        .unwrap();
        let ty = match b.resource {
            Resource::ReadBuffer(t) => format!("StructuredBuffer<{t}>"),
            Resource::WriteBuffer(t) => format!("RWStructuredBuffer<{t}>"),
            Resource::Textures | Resource::Samplers => {
                let ty = match b.resource {
                    Resource::Textures => "Texture2D",
                    Resource::Samplers => "SamplerState",
                    _ => unreachable!(),
                };
                writeln!(source,"#ifdef VORK_METAL\npublic struct {}Table {{ public {ty} handles[]; }};\n[[vk::binding({binding},1)]] public ParameterBlock<{}Table> {} : register({});\n#else\n[[vk::binding({binding},1)]] public {ty} {}[];\n#endif",b.name,b.name,b.name,b.define,b.name).unwrap();
                continue;
            }
        };
        writeln!(
            source,
            "[[vk::binding({binding},1)]] public {ty} {} : register({});",
            b.name, b.define
        )
        .unwrap();
    }
    source
}

#[cfg(test)]
mod tests {
    use super::*;

    // Run by check-metal-contracts with API validation both off and on. One
    // installed texture/sampler fallback must declare and bind exactly 8 bytes,
    // including the probe and shadow kernels reached by an ordinary render.
    #[test]
    #[ignore = "requires supported GPU and pinned Slang; set VORK_GPU_TESTS=1"]
    fn argument_buffer_scene_render() -> anyhow::Result<()> {
        let renderer = crate::test_renderer()?;
        let input = serde_json::json!({"version":2,
            "render":{"resolution":[16,16],"settings":{"samples":1}},
            "nodes":[
                {"type":"camera","translation":[0,0,4],"target":[0,0,0]},
                {"type":"mesh","geometry":crate::test_geometry::latitude_longitude(1.,16,8)},
                {"type":"light","shape":"rectangle","translation":[1,2,3],"target":[0,0,0],"size":2,"energy":20}
            ]});
        let base = std::path::Path::new(".");
        let mut installed = renderer.install(crate::test_scene(input, base)?, base)?;
        let mut frame = installed.create_frame_slot(&renderer, Default::default())?;
        let image = installed.render_frame(&renderer, &mut frame, 0.)?;
        assert_eq!(image.rgba.len(), 16 * 16 * 4);
        assert!(image.rgba.chunks_exact(4).any(|p| p[0] != 0));
        Ok(())
    }

    #[test]
    fn arrays_take_one_slot_independent_of_capacity() {
        let registers = material_registers(12);
        assert_eq!(registers[3], "-DVORK_MATERIAL_WORDS_REGISTER=t15");
        assert_eq!(registers[4], "-DVORK_SHADER_ERRORS_REGISTER=u16");
        let source = material_source();
        assert!(source.contains("handles[]"));
    }
}
