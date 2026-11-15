//! Offline installation from the same sources carried in release archives.
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::{fs, io::ErrorKind, path::Path};

#[derive(Parser)]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Install the bundled vork skill for Claude Code and Codex.
    #[command(verbatim_doc_comment)]
    Add,
}

const FILES: &[(&str, &[u8])] = &[
    ("SKILL.md", include_bytes!("../../../skills/vork/SKILL.md")),
    (
        "references/shot.md",
        include_bytes!("../../../skills/vork/references/shot.md"),
    ),
    (
        "references/assets.md",
        include_bytes!("../../../skills/vork/references/assets.md"),
    ),
];
const DESTINATIONS: &[&str] = &[".agents/skills/vork", ".claude/skills/vork"];

impl Args {
    pub fn run(&self) -> Result<()> {
        match self.command {
            Command::Add => add(),
        }
    }
}

fn preflight(path: &Path, directory: bool) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.is_symlink()
                || if directory {
                    !metadata.is_dir()
                } else {
                    !metadata.is_file()
                }
            {
                anyhow::bail!(anyhow::anyhow!(
                    "{} conflicts with skill installation; move it aside and rerun `vork skills add`",
                    path.display()
                ).context(vork_render::failure::FailureKind::Usage));
            }
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("inspect {}", path.display())),
    }
    Ok(())
}

fn add() -> Result<()> {
    // Inspect both destinations completely before changing either one.
    for destination in DESTINATIONS {
        for (name, _) in FILES {
            let path = Path::new(destination).join(name);
            let mut parents: Vec<_> = path
                .ancestors()
                .skip(1)
                .filter(|p| !p.as_os_str().is_empty())
                .collect();
            parents.reverse();
            for parent in parents {
                preflight(parent, true)?;
            }
            preflight(&path, false)?;
        }
    }
    for destination in DESTINATIONS {
        for (name, content) in FILES {
            let path = Path::new(destination).join(name);
            if path.exists()
                && fs::read(&path).with_context(|| format!("read {}", path.display()))? == *content
            {
                continue;
            }
            fs::create_dir_all(path.parent().unwrap())
                .with_context(|| format!("create parent of {}", path.display()))?;
            fs::write(&path, content).with_context(|| format!("write {}", path.display()))?;
        }
    }
    for destination in DESTINATIONS {
        output!("{destination}/");
    }
    Ok(())
}
