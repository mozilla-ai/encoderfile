use anyhow::Result;
use clap_derive::{Args, Subcommand};

use crate::{
    base_binary::{BaseBinaryResolver, TargetSpec},
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

        let _out_path = resolver.resolve()?;

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
                todo!()
            }
            Self::Remove(args) => args.remove(global),
        }
    }
}
