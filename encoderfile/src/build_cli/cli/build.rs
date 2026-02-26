use crate::build_cli::{
    base_binary::{BaseBinaryResolver, TargetSpec},
    terminal,
};

use super::{super::model::ModelTypeExt as _, GlobalArguments};
use crate::{
    format::{
        assets::{AssetKind, AssetPlan, AssetSource, PlannedAsset},
        codec::EncoderfileCodec,
    },
    generated::manifest::Backend,
};
use anyhow::{Context, Result};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufWriter, Seek, Write},
    path::PathBuf,
};

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
        let mut config = crate::build_cli::config::BuildConfig::load(&self.config)?;

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

        let target = config
            .encoderfile
            .target()?
            .unwrap_or(TargetSpec::detect_host()?);

        // load base binary
        let base_path = {
            let cache_dir = config.encoderfile.cache_dir();
            let base_binary_path = config.encoderfile.base_binary_path.as_deref();

            let resolver = BaseBinaryResolver {
                cache_dir: cache_dir.as_path(),
                base_binary_path,
                target,
                version: self.version.clone(),
            };

            resolver.resolve(self.no_download)?
        };

        let mut planned_assets: Vec<PlannedAsset<'_>> = Vec::new();

        // validate model config
        let model_config = config.encoderfile.model_config()?;

        planned_assets.push(PlannedAsset::from_asset_source(
            AssetSource::InMemory(Cow::Owned(serde_json::to_vec(&model_config)?)),
            AssetKind::ModelConfig,
        )?);
        terminal::success("Model config validated");

        // validate model
        let model_weights_path = config.encoderfile.path.model_weights_path()?;

        let model_asset = config
            .encoderfile
            .model_type
            .validate_model(&model_weights_path)?;

        planned_assets.push(model_asset);
        terminal::success("Model weights validated");

        // validate transform
        if let Some(asset) =
            crate::build_cli::transforms::validate_transform(&config.encoderfile, &model_config)?
        {
            planned_assets.push(asset);
            terminal::success("Transform validated");
        }

        // validate tokenizer
        let tokenizer_asset = crate::build_cli::tokenizer::validate_tokenizer(&config.encoderfile)?;
        planned_assets.push(tokenizer_asset);
        terminal::success("Tokenizer validated");

        // initialize final binary
        terminal::info("Writing encoderfile...");
        let output_path = config.encoderfile.output_path();
        let out = File::create(output_path.clone()).context(format!(
            "Failed to create final encoderfile at {:?}",
            output_path.as_path()
        ))?;

        let mut out = BufWriter::new(out);
        let mut base = File::open(base_path.as_path()).context(format!(
            "Failed to open base binary at {:?}",
            base_path.as_path()
        ))?;

        // copy base binary to out
        std::io::copy(&mut base, &mut out).context(format!(
            "Failed to copy base binary to {:?}",
            output_path.as_path()
        ))?;

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

        terminal::success_kv("Encoderfile written to", output_path.display());

        Ok(())
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
