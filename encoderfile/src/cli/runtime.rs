use anyhow::Result;
use clap_derive::{Args, Subcommand};

use crate::{
    base_binary::{BaseBinaryResolver, TargetSpec, list_downloaded_runtimes},
    cli::GlobalArguments,
};

#[derive(Debug, Subcommand)]
pub enum Runtime {
    Add(RuntimeArgs),
    List,
    Remove(RuntimeArgs),
}

#[derive(Debug, Clone, Args)]
pub struct RuntimeArgs {
    #[arg(help = "Target triple to download.")]
    pub target: TargetSpec,

    #[arg(long, help = "Version override (defaults to current version).")]
    pub version: Option<String>,
}

impl RuntimeArgs {
    fn add(&self, global: &GlobalArguments) -> Result<()> {
        let resolver = BaseBinaryResolver {
            cache_dir: &global.cache_dir(),
            base_binary_path: None,
            target: self.target.clone(),
            version: self.version.clone(),
        };

        let _out_path = resolver.resolve(true)?;

        Ok(())
    }

    fn remove(&self, global: &GlobalArguments) -> Result<()> {
        let resolver = BaseBinaryResolver {
            cache_dir: &global.cache_dir(),
            base_binary_path: None,
            target: self.target.clone(),
            version: self.version.clone(),
        };

        resolver.remove()
    }
}

impl Runtime {
    pub fn execute(&self, global: &GlobalArguments) -> Result<()> {
        match self {
            Self::Add(args) => args.add(global),
            Self::List => {
                let runtimes = list_downloaded_runtimes(global.cache_dir().as_path())?;

                // TODO: Make pretty
                for runtime in runtimes {
                    println!("{} {}", runtime.version, runtime.target)
                }

                Ok(())
            }
            Self::Remove(args) => args.remove(global),
        }
    }
}
