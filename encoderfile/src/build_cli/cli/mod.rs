use anyhow::Result;
use std::path::PathBuf;

use clap_derive::{Args, Parser, Subcommand};

mod build;
mod inspect;
mod runtime;

#[cfg(feature = "dev-utils")]
pub use build::{test_build_args, test_build_args_working_dir};

pub use inspect::inspect_encoderfile;

#[derive(Debug, Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    #[command(flatten)]
    pub global_args: GlobalArguments,
}

#[derive(Debug, Default, Clone, Args)]
pub struct GlobalArguments {
    #[arg(
        long = "cache-dir",
        help = "Cache directory. This is used for build artifacts. Optional."
    )]
    cache_dir: Option<PathBuf>,
}

impl GlobalArguments {
    pub fn cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .unwrap_or(crate::build_cli::cache::default_cache_dir())
    }
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Build an encoderfile.")]
    Build(build::BuildArgs),
    #[command(about = "Get Encoderfile version.")]
    Version(()),
    #[command(subcommand, about = "Manage Encoderfile runtimes.")]
    Runtime(runtime::Runtime),
    #[command(about = "Generate a new transform.")]
    NewTransform {
        #[arg(short = 'm', long = "model-type", help = "Model type")]
        model_type: String,
    },
    #[command(about = "Inspect the metadata of an encoderfile.")]
    Inspect {
        #[arg(required = true, help = "Path to encoderfile.")]
        path: String,
    },
}

impl Commands {
    pub fn run(self, global: &GlobalArguments) -> Result<()> {
        match self {
            Self::Build(args) => args.run(global),
            Self::Version(_) => {
                println!("Encoderfile {}", env!("CARGO_PKG_VERSION"));
                Ok(())
            }
            Self::Runtime(r) => r.execute(global),
            Self::NewTransform { model_type } => super::transforms::new_transform(model_type),
            Self::Inspect { path } => {
                println!("{}", inspect::inspect_encoderfile(&path)?);
                Ok(())
            }
        }
    }
}
