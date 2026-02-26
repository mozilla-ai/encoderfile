use crate::builder::{base_binary::TargetSpec, terminal};

use super::GlobalArguments;
use anyhow::{Context, Result};
use std::path::PathBuf;

use clap_derive::Args;

#[derive(Debug, Args)]
pub struct BuildArgs {
    #[arg(short = 'f', help = "Path to config file. Required.")]
    pub config: PathBuf,
    #[arg(
        short = 'o',
        long = "output-path",
        help = "Output path, e.g., `./my_model.encoderfile`. Optional"
    )]
    pub output_path: Option<PathBuf>,
    #[arg(
        long = "base-binary-path",
        help = "Path to base binary to use. Optional."
    )]
    pub base_binary_path: Option<PathBuf>,
    #[arg(
        long = "platform",
        help = "Target platform to build. Follows standard rust target triple format."
    )]
    pub platform: Option<TargetSpec>,
    #[arg(
        long,
        help = "Encoderfile version override (defaults to current version)."
    )]
    pub version: Option<String>,
    #[arg(
        long = "no-download",
        help = "Disable downloading",
        default_value = "false"
    )]
    pub no_download: bool,
    #[arg(
        long = "directory", // working-dir???
        help = "Set the working directory for the build process. Optional.",
        default_value = None
    )]
    pub working_dir: Option<PathBuf>,
}

impl BuildArgs {
    pub fn run(&self, global: &GlobalArguments) -> Result<()> {
        terminal::info("Loading config...");
        let mut config = crate::builder::config::BuildConfig::load(&self.config)?;

        // change working dir if specified
        if let Some(working_dir) = &self.working_dir {
            std::env::set_current_dir(working_dir).context(format!(
                "Failed to change working directory to {:?}",
                working_dir.as_path()
            ))?;
        }

        // --- handle user flags ---------------------------------------------------
        if let Some(o) = &self.output_path {
            config.encoderfile.output_path = Some(o.to_path_buf());
        }

        if let Some(cache_dir) = &global.cache_dir {
            config.encoderfile.cache_dir = Some(cache_dir.to_path_buf());
        }

        if let Some(base_binary_path) = &self.base_binary_path {
            config.encoderfile.base_binary_path = Some(base_binary_path.to_path_buf())
        }

        super::super::builder::EncoderfileBuilder::new(config)
            .build(&self.version, self.no_download)
    }
}

#[cfg(feature = "dev-utils")]
pub fn test_build_args(
    config: impl Into<PathBuf>,
    base_binary_path: impl Into<PathBuf>,
) -> BuildArgs {
    BuildArgs {
        config: config.into(),
        output_path: None,
        base_binary_path: Some(base_binary_path.into()),
        platform: None,
        version: None,
        no_download: true,
        working_dir: None,
    }
}

#[cfg(feature = "dev-utils")]
pub fn test_build_args_working_dir(
    config: impl Into<PathBuf>,
    base_binary_path: impl Into<PathBuf>,
    working_dir: impl Into<PathBuf>,
) -> BuildArgs {
    BuildArgs {
        config: config.into(),
        output_path: None,
        base_binary_path: Some(base_binary_path.into()),
        platform: None,
        version: None,
        no_download: true,
        working_dir: Some(working_dir.into()),
    }
}
