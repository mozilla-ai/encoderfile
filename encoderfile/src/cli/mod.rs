use crate::base_binary::{BaseBinaryResolver, TargetSpec};

use super::model::ModelTypeExt as _;
use anyhow::Result;
use encoderfile_core::{
    format::{
        assets::{AssetKind, AssetPlan, AssetSource, PlannedAsset},
        codec::EncoderfileCodec,
    },
    generated::manifest::Backend,
};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufWriter, Seek, Write},
    path::PathBuf,
};

use clap_derive::{Args, Parser, Subcommand};

mod runtime;

#[derive(Debug, Parser)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    #[command(flatten)]
    pub global_args: GlobalArguments,
}

#[derive(Debug, Clone, Args)]
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
            .unwrap_or(crate::cache::default_cache_dir())
    }
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    #[command(about = "Build an encoderfile.")]
    Build(BuildArgs),
    #[command(about = "Get Encoderfile version.")]
    Version(()),
    #[command(subcommand, about = "Manage Encoderfile runtimes.")]
    Runtime(runtime::Runtime),
    #[command(about = "Generate a new transform.")]
    NewTransform {
        #[arg(short = 'm', long = "model-type", help = "Model type")]
        model_type: String,
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
        long = "base-binary-path",
        help = "Path to base binary to use. Optional."
    )]
    base_binary_path: Option<PathBuf>,
    #[arg(
        long = "platform",
        help = "Target platform to build. Follows standard rust target triple format."
    )]
    platform: Option<TargetSpec>,
    #[arg(
        long,
        help = "Encoderfile version override (defaults to current version)."
    )]
    version: Option<String>,
    #[arg(
        long = "no-download",
        help = "Disable downloading",
        default_value = "false"
    )]
    no_download: bool,
}

impl BuildArgs {
    fn run(self, global: &GlobalArguments) -> Result<()> {
        let mut config = super::config::BuildConfig::load(&self.config)?;

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

        let target = config
            .encoderfile
            .target()?
            .unwrap_or(TargetSpec::detect_host()?);

        // load base binary
        let base_path = match &config.encoderfile.base_binary_path {
            Some(p) => p.clone(),
            None => {
                let cache_dir = config.encoderfile.cache_dir();

                let base_binary_path = config.encoderfile.base_binary_path.as_deref();

                let resolver = BaseBinaryResolver {
                    cache_dir: cache_dir.as_path(),
                    base_binary_path,
                    target,
                    version: self.version.clone(),
                };

                resolver.resolve(self.no_download)?
            }
        };

        let mut planned_assets: Vec<PlannedAsset<'_>> = Vec::new();

        // validate model config
        let model_config = config.encoderfile.model_config()?;

        planned_assets.push(PlannedAsset::from_asset_source(
            AssetSource::InMemory(Cow::Owned(serde_json::to_vec(&model_config)?)),
            AssetKind::ModelConfig,
        )?);

        // validate model
        let model_weights_path = config.encoderfile.path.model_weights_path()?;

        let model_asset = config
            .encoderfile
            .model_type
            .validate_model(&model_weights_path)?;

        planned_assets.push(model_asset);

        // validate transform
        if let Some(asset) =
            crate::transforms::validate_transform(&config.encoderfile, &model_config)?
        {
            planned_assets.push(asset);
        }

        // validate tokenizer
        let tokenizer_asset = crate::tokenizer::validate_tokenizer(&config.encoderfile)?;
        planned_assets.push(tokenizer_asset);

        // initialize final binary
        let out = File::create(config.encoderfile.output_path())?;
        let mut out = BufWriter::new(out);
        let mut base = File::open(base_path)?;

        // copy base binary to out
        std::io::copy(&mut base, &mut out)?;

        // get metadata start position
        let payload_start = out.stream_position()?;

        // create codec
        let codec = EncoderfileCodec::new(payload_start);

        // create asset plan
        let asset_plan = AssetPlan::new(planned_assets)?;

        // write to file
        codec.write(
            config.encoderfile.name.clone(),
            config.encoderfile.version.clone(),
            config.encoderfile.model_type.clone(),
            Backend::Cpu,
            &asset_plan,
            &mut out,
        )?;

        out.flush()?;

        Ok(())
    }
}
