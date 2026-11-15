use anyhow::{Context, Result};
use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Read the scene from this file.
    scene: PathBuf,
    /// Supply a JSON list of questions or use - to read stdin.
    questions: String,
}

pub(crate) fn single_line(error: anyhow::Error) -> anyhow::Error {
    use vork_render::failure::{FailureKind, exit_code, suggests_setup};
    let category = match exit_code(&error) {
        2 => Some(FailureKind::Usage),
        3 => Some(FailureKind::Scene),
        5 => None,
        _ => Some(FailureKind::Device),
    };
    // The category is attached again below; its own line of the chain is not said twice.
    let label = category.map(|category| category.to_string());
    let mut message = error.chain().map(|cause| cause.to_string())
        .filter(|cause| Some(cause) != label.as_ref())
        .collect::<Vec<_>>().join(": ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if suggests_setup(&error) {
        message.push_str("; run vork setup to check the required tools");
    }
    match category {
        Some(category) => anyhow::anyhow!(message).context(category),
        None => std::io::Error::other(message).into(),
    }
}

impl Args {
    pub fn run(&self) -> Result<()> {
        let result = (|| {
            let bytes = if self.questions == "-" {
                let mut bytes = Vec::new();
                std::io::Read::read_to_end(&mut std::io::stdin(), &mut bytes)?;
                bytes
            } else {
                self.questions.as_bytes().to_vec()
            };
            let questions = serde_json::from_slice(vork_scene::diagnostics::without_bom(&bytes))
                .context("ask: supply a JSON list directly or use - to read stdin")
                .context(vork_render::failure::FailureKind::Usage)?;
            vork_render::ask::run(&self.scene, &questions)
        })()
        .map_err(single_line)?;
        output!("{}", serde_json::to_string(&result)?);
        Ok(())
    }
}
