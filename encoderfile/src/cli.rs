use super::templates::TEMPLATES;
use anyhow::{Result, bail};
use std::path::PathBuf;

use clap_derive::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Build(BuildArgs),
    Version{},
}

impl Commands {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Build(args) => args.run(),
            Self::Version {} => {
                println!("Encoderfile {}", env!("CARGO_PKG_VERSION"));
                Ok(())
            }
        }
    }
}

#[derive(Debug, Args)]
pub struct BuildArgs {
    #[arg(short = 'f')]
    config: PathBuf,
    #[arg(long = "output-dir")]
    output_dir: Option<PathBuf>,
    #[arg(long = "cache-dir")]
    cache_dir: Option<PathBuf>,
    #[arg(long = "no-build")]
    no_build: bool,
}

impl BuildArgs {
    fn run(self) -> Result<()> {
        let mut config = super::config::Config::load(&self.config)?;

        if let Some(o) = &self.output_dir {
            config.encoderfile.output_dir = o.to_path_buf();
        }

        if let Some(cache_dir) = &self.cache_dir {
            config.encoderfile.cache_dir = cache_dir.to_path_buf();
        }

        if self.no_build {
            config.encoderfile.build = false;
        }

        let write_dir = config.encoderfile.get_generated_dir();
        std::fs::create_dir_all(&write_dir)?;

        // create src/ directory
        std::fs::create_dir_all(write_dir.join("src/"))?;

        let ctx = config.encoderfile.to_tera_ctx()?;

        render("main.rs.tera", &ctx, &write_dir, "src/main.rs")?;
        render("Cargo.toml.tera", &ctx, &write_dir, "Cargo.toml")?;

        if config.encoderfile.build {
            let cargo_toml_path = write_dir.join("Cargo.toml").canonicalize()?;

            let manifest_dir = cargo_toml_path.to_str().unwrap();

            std::process::Command::new("cargo")
                .arg("build")
                .arg("--release")
                .arg("--manifest-path")
                .arg(manifest_dir)
                .status()?;

            let generated_path = config
                .encoderfile
                .get_generated_dir()
                .join("target/release/encoderfile");

            if !generated_path.exists() {
                bail!("ERROR: Generated path does not exist. This should not happen.")
            }

            // export encoderfile to output dir
            std::fs::rename(generated_path, config.encoderfile.get_output_dir()?)?;
        }

        Ok(())
    }
}

fn render(
    template_name: &str,
    ctx: &tera::Context,
    write_dir: &std::path::Path,
    out_path: &str,
) -> Result<()> {
    let rendered = TEMPLATES.render(template_name, ctx)?;

    let file = write_dir.join(out_path);

    std::fs::write(file, rendered)?;

    Ok(())
}
