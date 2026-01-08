use super::{model::ModelTypeExt as _, templates::TEMPLATES};
use anyhow::{Context, Result, bail};
use std::path::PathBuf;

use clap_derive::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Build an encoderfile.")]
    Build(BuildArgs),
    #[command(about = "Get Encoderfile version.")]
    Version(()),
    #[command(about = "Generate a new transform.")]
    NewTransform {
        #[arg(short = 'm', long = "model-type", help = "Model type")]
        model_type: String,
    },
}

impl Commands {
    pub fn run(self) -> Result<()> {
        match self {
            Self::Build(args) => args.run(),
            Self::Version(_) => {
                println!("Encoderfile {}", env!("CARGO_PKG_VERSION"));
                Ok(())
            }
            Self::NewTransform { model_type } => super::transforms::new_transform(model_type),
        }
    }
}

#[derive(Debug, Args)]
pub struct BuildArgs {
    #[arg(short = 'f', help = "Path to config file. Required.")]
    config: PathBuf,
    #[arg(
        short = 'o',
        long = "output-path",
        help = "Output path, e.g., `./my_model.encoderfile`. Optional"
    )]
    output_path: Option<PathBuf>,
    #[arg(
        long = "cache-dir",
        help = "Cache directory. This is used for build artifacts. Optional."
    )]
    cache_dir: Option<PathBuf>,
    #[arg(
        long = "no-build",
        help = "Skips build stage. Only generates files to directory in `cache_dir`. Defaults to False."
    )]
    no_build: bool,
}

impl BuildArgs {
    fn run(self) -> Result<()> {
        let mut config = super::config::BuildConfig::load(&self.config)?;

        // --- handle user flags ---------------------------------------------------
        if let Some(o) = &self.output_path {
            config.encoderfile.output_path = Some(o.to_path_buf());
        }

        if let Some(cache_dir) = &self.cache_dir {
            config.encoderfile.cache_dir = Some(cache_dir.to_path_buf());
        }

        // validate model config
        let model_config = config.encoderfile.model_config()?;

        // validate model
        config
            .encoderfile
            .model_type
            .validate_model(&config.encoderfile.path.model_weights_path()?)?;

        // validate transform
        crate::transforms::validate_transform(&config.encoderfile, &model_config)?;

        // setup write directory
        let write_dir = config.encoderfile.get_generated_dir();
        std::fs::create_dir_all(write_dir.join("src/"))
            .with_context(|| format!("Failed creating {}", write_dir.display()))?;

        // render templates
        let ctx = config.encoderfile.to_tera_ctx()?;

        render("main.rs.tera", &ctx, &write_dir, "src/main.rs")?;
        render("Cargo.toml.tera", &ctx, &write_dir, "Cargo.toml")?;

        // canonicalize paths
        let cargo_toml_path = write_dir
            .join("Cargo.toml")
            .canonicalize()
            .context("Canonicalizing Cargo.toml failed")?;

        // run cargo build with environment isolation
        let manifest_path = cargo_toml_path.to_string_lossy();
        let status = std::process::Command::new("cargo")
            .arg("build")
            .arg("--release")
            .arg("--manifest-path")
            .arg(&*manifest_path)
            // full workspace isolation (stops parent workspace detection)
            .env("CARGO_WORKSPACE_DIR", "/nonexistent")
            // ensure temp files stay local
            .env("CARGO_TARGET_DIR", write_dir.join("target"))
            .env("CARGO_HOME", write_dir.join(".cargo"))
            .status()
            .context("Failed to run cargo build")?;

        if !status.success() {
            bail!("cargo build failed with exit status {:?}", status);
        }

        // locate generated binary
        let generated_binary = write_dir.join("target/release/encoderfile");
        if !generated_binary.exists() {
            bail!(
                "ERROR: generated binary {:?} does not exist.",
                generated_binary
            );
        }

        // final output move (filesystem-safe)
        move_across_filesystems(&generated_binary, &config.encoderfile.output_path())?;

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

// -----------------------------------------------------------------------------
// Safe move across filesystems (avoids EXDEV)
// -----------------------------------------------------------------------------
fn move_across_filesystems(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    match std::fs::rename(src, dst) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::CrossesDevices => {
            std::fs::copy(src, dst).with_context(|| format!("copying {:?} → {:?}", src, dst))?;
            std::fs::remove_file(src).with_context(|| format!("removing {:?}", src))?;
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}
