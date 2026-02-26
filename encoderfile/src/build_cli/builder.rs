use std::{
    borrow::Cow,
    fs::File,
    io::{BufWriter, Seek, Write},
};

use crate::{
    build_cli::{
        base_binary::{BaseBinaryResolver, TargetSpec},
        config::BuildConfig,
        model::ModelTypeExt as _,
        terminal,
    },
    format::{
        assets::{AssetKind, AssetPlan, AssetSource, PlannedAsset},
        codec::EncoderfileCodec,
    },
    generated::manifest::Backend,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EncoderfileBuilder {
    config: BuildConfig,
}

impl EncoderfileBuilder {
    pub fn new(config: BuildConfig) -> EncoderfileBuilder {
        Self { config }
    }

    pub fn build(&self, version: Option<String>, no_download: bool) -> Result<()> {
        let target = self
            .config
            .encoderfile
            .target()?
            .unwrap_or(TargetSpec::detect_host()?);

        // load base binary
        let base_path = {
            let cache_dir = self.config.encoderfile.cache_dir();
            let base_binary_path = self.config.encoderfile.base_binary_path.as_deref();

            let resolver = BaseBinaryResolver {
                cache_dir: cache_dir.as_path(),
                base_binary_path,
                target,
                version: version.clone(),
            };

            resolver.resolve(no_download)?
        };

        let mut planned_assets: Vec<PlannedAsset<'_>> = Vec::new();

        // validate model config
        let model_config = self.config.encoderfile.model_config()?;

        planned_assets.push(PlannedAsset::from_asset_source(
            AssetSource::InMemory(Cow::Owned(serde_json::to_vec(&model_config)?)),
            AssetKind::ModelConfig,
        )?);
        terminal::success("Model config validated");

        // validate model
        let model_weights_path = self.config.encoderfile.path.model_weights_path()?;

        let model_asset = self
            .config
            .encoderfile
            .model_type
            .validate_model(&model_weights_path)?;

        planned_assets.push(model_asset);
        terminal::success("Model weights validated");

        // validate transform
        if let Some(asset) = crate::build_cli::transforms::validate_transform(
            &self.config.encoderfile,
            &model_config,
        )? {
            planned_assets.push(asset);
            terminal::success("Transform validated");
        }

        // validate tokenizer
        let tokenizer_asset =
            crate::build_cli::tokenizer::validate_tokenizer(&self.config.encoderfile)?;
        planned_assets.push(tokenizer_asset);
        terminal::success("Tokenizer validated");

        // initialize final binary
        terminal::info("Writing encoderfile...");
        let output_path = self.config.encoderfile.output_path();
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
            self.config.encoderfile.name.clone(),
            self.config.encoderfile.version.clone(),
            self.config.encoderfile.model_type.clone(),
            Backend::Cpu,
            &asset_plan,
            &mut out,
        )?;

        out.flush()?;

        terminal::success_kv("Encoderfile written to", output_path.display());

        Ok(())
    }
}
