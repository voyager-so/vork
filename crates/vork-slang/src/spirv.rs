/// Outlining bounds Slang's optimization graph; the device compiler still needs
/// freedom to inline large closure returns rather than spill them at each hit.
/// SPIR-V Function Control's DontInline bit is a performance hint, not semantics.
pub fn allow_driver_inlining(code: &mut [u8]) -> anyhow::Result<()> {
    anyhow::ensure!(code.len() >= 20 && code.len() % 4 == 0, "malformed SPIR-V from Slang");
    anyhow::ensure!(u32::from_le_bytes(code[..4].try_into().unwrap()) == 0x07230203, "invalid SPIR-V magic from Slang");
    let mut offset = 20;
    while offset < code.len() {
        let instruction = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
        let words = (instruction >> 16) as usize;
        anyhow::ensure!(words > 0 && words <= (code.len() - offset) / 4, "malformed SPIR-V instruction from Slang");
        if instruction & 0xffff == 54 { // OpFunction: result type, result ID, control, function type.
            anyhow::ensure!(words == 5, "malformed SPIR-V function from Slang");
            let control = &mut code[offset + 12..offset + 16];
            let flags = u32::from_le_bytes(control.try_into().unwrap());
            control.copy_from_slice(&(flags & !0x2).to_le_bytes());
        }
        offset += words * 4;
    }
    Ok(())
}

