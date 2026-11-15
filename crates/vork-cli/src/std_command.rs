//! Optional standard-library CLI registration; the renderer has no dependency on it.
use anyhow::Result;

pub const HELP: &str = vork_std::HELP;

#[derive(clap::Args)]
#[command(after_help = HELP)]
pub struct Args {
    function: String,
    /// JSON argument; use - to read JSON from stdin.
    argument: String,
}
impl Args {
    pub fn run(&self) -> Result<()> {
        let argument = if self.argument == "-" {
            // PowerShell 5.1 pipes can begin with a UTF-8 byte-order mark.
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut std::io::stdin(), &mut bytes)?;
            String::from_utf8(vork_scene::diagnostics::without_bom(&bytes).to_vec())?
        } else {
            self.argument.clone()
        };
        let result=(|| -> Result<String> { Ok(if self.function=="environment" {
            let cache=vork_render::disk_cache::Store::host();
            let read=|key|vork_render::disk_cache::read(&cache.path("std-environment",1,key));
            let write=|key,bytes:&[u8]|{let _=vork_render::disk_cache::write(&cache.path("std-environment",1,key),bytes);};
            let options:serde_json::Value=serde_json::from_str(&argument)?;
            let file=options["file"].as_str().ok_or_else(||anyhow::anyhow!("environment file path"))?;
            if vork_render::footage::is_media(std::path::Path::new(file)) {
                let mut source=vork_render::footage::Source::open(std::path::Path::new(file))?;
                let fps:vork_scene::Time=serde_json::from_value(options.get("fps").cloned().unwrap_or(24.into()))?;
                let times=source.timeline(fps)?;
                let inputs=source.files().to_vec();
                vork_std::environment::prepare_frames(&argument,&inputs,&times,|time|source.frame(time,fps,u32::MAX),read,write)?
            } else {vork_std::environment::prepare(&argument,read,write)?}
        } else {vork_std::evaluate(&self.function,&argument)?}) })().map_err(|error| {
            // A std function evaluates its argument on the CPU: what it refuses is the authored call.
            vork_render::failure::scene(if serde_json::from_str::<serde_json::Value>(&argument).is_err() {
                error.context(format!("vork std {} expects a JSON argument, not a filename; pass JSON directly or read a file with `vork std {} - < scene.json`", self.function, self.function))
            } else { error })
        })?;
        output!("{result}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_name_is_optional() {
        assert!(Args { function: "rounded_box".into(), argument: "{}".into() }.run().is_ok());
        assert!(Args {
            function: "rounded_box".into(),
            argument: r#"{"name":"custom"}"#.into(),
        }.run().is_ok());
    }
}
