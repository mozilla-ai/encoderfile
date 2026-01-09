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
        long = "base-binary-path",
        help = "Path to base binary to use. Optional."
    )]
    base_binary_path: Option<PathBuf>,
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

        if let Some(base_binary_path) = &self.base_binary_path {
            config.encoderfile.base_binary_path = Some(base_binary_path.to_path_buf())
        }

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

        // load base binary
        let base_path = match &config.encoderfile.base_binary_path {
            Some(p) => p.as_path(),
            None => unimplemented!("No support for downloading default binaries yet."),
        };

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
