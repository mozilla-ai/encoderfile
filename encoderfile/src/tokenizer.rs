// IMPORTANT NOTE:
//
// Tokenizer configuration is NOT a stable, self-contained artifact.
//
// In practice, tokenizer behavior is split across:
//   - tokenizer.json (partially serialized runtime state) SOMETIMES in older models
//   - tokenizer_config.json (optional, inconsistently populated)
//   - implicit defaults inside the `tokenizers` library
//   - and values that affect inference but are *never serialized*
//
// This means:
//   - Missing fields fail silently and fall back to defaults
//   - Backwards compatibility is heuristic, not contractual
//   - Some critical values (e.g. pad_token_id) must be re-derived at runtime
//   - A "valid" tokenizer config can still produce subtly wrong results
//
// This code exists to aggressively reconstruct a deterministic TokenizerConfig
// for inference, emitting warnings where possible — but be aware:
//
// ⚠️ Incorrect or incomplete tokenizer configs may not crash.
// ⚠️ They may instead produce silently incorrect model outputs.
// ⚠️ If, khas v'shalem, something silently breaks in Encoderfile, I bet $5 it is going to be this feature.
//
// This is not ideal and will be revisited in v1.0.0 once we have an opportunity to make breaking changes
// in the way encoderfile.yml works, etc.

use anyhow::Result;
use encoderfile_core::{
    common::TokenizerConfig,
    format::assets::{AssetKind, AssetSource, PlannedAsset},
    runtime::TokenizerService,
};
use std::str::FromStr;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

use crate::config::{EncoderfileConfig, TokenizerPadStrategy};

pub fn validate_tokenizer<'a>(config: &'a EncoderfileConfig) -> Result<PlannedAsset<'a>> {
    let tokenizer =
        match Tokenizer::from_str(std::fs::read_to_string(config.path.tokenizer_path()?)?.as_str())
        {
            Ok(t) => t,
            Err(e) => anyhow::bail!("FATAL: Failed to load tokenizer: {:?}", e),
        };

    let config = config.validate_tokenizer_config(&tokenizer)?;

    let service = TokenizerService::new(tokenizer, config)?;

    let serialized = serde_json::to_vec(&service)?;

    PlannedAsset::from_asset_source(
        AssetSource::InMemory(std::borrow::Cow::Owned(serialized)),
        AssetKind::Tokenizer,
    )
}

impl EncoderfileConfig {
    pub fn validate_tokenizer_config(&self, tokenizer: &Tokenizer) -> Result<TokenizerConfig> {
        let mut config = match self.path.tokenizer_config_path()? {
            // if tokenizer_config.json is provided, use that
            Some(tokenizer_config_path) => {
                // open tokenizer_config
                let contents = std::fs::read_to_string(tokenizer_config_path)?;
                let tokenizer_config: serde_json::Value = serde_json::from_str(contents.as_str())?;

                tokenizer_config_from_json_value(tokenizer_config, tokenizer)?
            }
            // otherwise check for any values given in tokenizer.json (backwards compatibility)
            None => {
                // will fail here if neither are given
                from_tokenizer(tokenizer)?
            }
        };

        // TODO: insert any overrides from encoderfile.yml here
        let tokenizer_build_config = match &self.tokenizer {
            Some(t) => t,
            None => return Ok(config),
        };

        if let Some(s) = &tokenizer_build_config.pad_strategy {
            config.padding.strategy = match s {
                TokenizerPadStrategy::BatchLongest => PaddingStrategy::BatchLongest,
                TokenizerPadStrategy::Fixed { fixed } => PaddingStrategy::Fixed(*fixed),
            }
        };

        Ok(config)
    }
}

fn from_tokenizer(tokenizer: &Tokenizer) -> Result<TokenizerConfig> {
    let padding = match tokenizer.get_padding() {
        Some(p) => p.clone(),
        None => {
            let padding_params = PaddingParams::default();

            eprintln!(
                "WARNING: No padding params found in `tokenizer.json`. Using defaults: {:?}",
                &padding_params
            );

            padding_params
        }
    };

    Ok(TokenizerConfig { padding })
}

fn tokenizer_config_from_json_value(
    val: serde_json::Value,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<TokenizerConfig> {
    let mut builder = TokenizerConfigBuilder::new(
        val.as_object()
            .ok_or(anyhow::anyhow!("tokenizer_config.json must be an object"))?,
    );

    builder.field(
        "padding_side",
        |config, v| {
            let side = v
                .as_str()
                .ok_or(anyhow::anyhow!("padding_side must be a str"))?;

            config.padding.direction = match side {
                "left" => tokenizers::PaddingDirection::Left,
                "right" => tokenizers::PaddingDirection::Right,
                _ => anyhow::bail!("padding_side must be \"left\" or \"right\""),
            };

            Ok(())
        },
        |config| config.padding.direction,
    )?;

    builder.field(
        "pad_to_multiple_of",
        |config, v| {
            if v.is_null() {
                config.padding.pad_to_multiple_of = None;
                return Ok(());
            }

            config.padding.pad_to_multiple_of = v.as_u64().map(|i| Some(i as usize)).ok_or(
                anyhow::anyhow!("pad_to_multiple_of must be an unsigned int or null"),
            )?;

            Ok(())
        },
        |config| config.padding.pad_to_multiple_of,
    )?;

    builder.field(
        "pad_token",
        |config, v| {
            config.padding.pad_token = v
                .as_str()
                .ok_or(anyhow::anyhow!("pad_token must be a string"))?
                .to_string();

            Ok(())
        },
        |config| config.padding.pad_token.clone(),
    )?;

    builder.field(
        "pad_token_type_id",
        |config, v| {
            config.padding.pad_type_id = v
                .as_u64()
                .map(|i| i as u32)
                .ok_or(anyhow::anyhow!("pad_token_type_id must be an unsigned int"))?;

            Ok(())
        },
        |config| config.padding.pad_type_id,
    )?;

    // now we fetch pad_token_id manually because it doesn't get serialized into tokenizer_config.json!
    builder.set_pad_token_id(tokenizer)?;

    builder.build()
}

#[derive(Debug)]
struct TokenizerConfigBuilder<'a> {
    config: TokenizerConfig,
    val: &'a serde_json::value::Map<String, serde_json::Value>,
}

impl<'a> TokenizerConfigBuilder<'a> {
    fn new(val: &'a serde_json::value::Map<String, serde_json::Value>) -> Self {
        Self {
            config: TokenizerConfig::default(),
            val,
        }
    }

    fn build(self) -> Result<TokenizerConfig> {
        Ok(self.config)
    }

    fn set_pad_token_id(&mut self, tokenizer: &Tokenizer) -> Result<()> {
        let pad_token = self.config.padding.pad_token.as_str();
        self.config.padding.pad_id = tokenizer.token_to_id(pad_token).ok_or(anyhow::anyhow!(
            "pad_token set to {}, but token does not exist in tokenizer",
            pad_token
        ))?;

        Ok(())
    }

    fn field<P, D, V>(
        &mut self,
        field: &str,
        process_value_fn: P,
        default_value_fn: D,
    ) -> Result<()>
    where
        P: FnOnce(&mut TokenizerConfig, &serde_json::Value) -> Result<()>,
        D: FnOnce(&TokenizerConfig) -> V,
        V: std::fmt::Debug,
    {
        match self.val.get(field) {
            Some(v) => process_value_fn(&mut self.config, v),
            None => {
                if !self.val.contains_key(field) {
                    eprintln!(
                        "WARNING: No {} found in tokenizer_config.json. Using default: {:?}",
                        field,
                        default_value_fn(&self.config),
                    )
                }

                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use encoderfile_core::common::ModelType;

    use crate::config::{ModelPath, TokenizerBuildConfig};

    use super::*;

    fn load_tokenizer_from_path(path: &std::path::Path) -> Result<Tokenizer> {
        Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from path: {e:?}"))
    }

    #[test]
    fn test_validate_tokenizer() {
        let config = EncoderfileConfig {
            name: "my-model".into(),
            version: "0.0.1".into(),
            path: ModelPath::Directory("../models/embedding".into()),
            model_type: ModelType::Embedding,
            output_path: None,
            cache_dir: None,
            transform: None,
            tokenizer: None,
            validate_transform: false,
            base_binary_path: None,
        };

        let tokenizer = load_tokenizer_from_path(
            &config
                .path
                .tokenizer_path()
                .expect("Failed to load tokenizer"),
        )
        .expect("Failed to load tokenizer");

        let tokenizer_config = config
            .validate_tokenizer_config(&tokenizer)
            .expect("Failed to validate tokenizer config");

        assert_eq!(format!("{:?}", tokenizer_config.padding.direction), "Right");
        assert_eq!(
            format!("{:?}", tokenizer_config.padding.strategy),
            "BatchLongest"
        );
        assert_eq!(tokenizer_config.padding.pad_id, 0);
        assert_eq!(tokenizer_config.padding.pad_token, "[PAD]");
        assert!(tokenizer_config.padding.pad_to_multiple_of.is_none());
        assert_eq!(tokenizer_config.padding.pad_type_id, 0);
    }

    #[test]
    fn test_validate_tokenizer_fixed() {
        let config = EncoderfileConfig {
            name: "my-model".into(),
            version: "0.0.1".into(),
            path: ModelPath::Directory("../models/embedding".into()),
            model_type: ModelType::Embedding,
            output_path: None,
            cache_dir: None,
            transform: None,
            tokenizer: Some(TokenizerBuildConfig {
                pad_strategy: Some(TokenizerPadStrategy::Fixed { fixed: 512 }),
            }),
            validate_transform: false,
            base_binary_path: None,
        };

        let tokenizer = load_tokenizer_from_path(
            &config
                .path
                .tokenizer_path()
                .expect("Failed to load tokenizer"),
        )
        .expect("Failed to load tokenizer");

        let tokenizer_config = config
            .validate_tokenizer_config(&tokenizer)
            .expect("Failed to validate tokenizer config");

        assert_eq!(format!("{:?}", tokenizer_config.padding.direction), "Right");
        assert_eq!(
            format!("{:?}", tokenizer_config.padding.strategy),
            "Fixed(512)"
        );
        assert_eq!(tokenizer_config.padding.pad_id, 0);
        assert_eq!(tokenizer_config.padding.pad_token, "[PAD]");
        assert!(tokenizer_config.padding.pad_to_multiple_of.is_none());
        assert_eq!(tokenizer_config.padding.pad_type_id, 0);
    }

    #[test]
    fn test_validate_tokenizer_no_config() {
        let path = ModelPath::Directory("../models/token_classification".into());

        let explicit_path = ModelPath::Paths {
            model_config_path: path.model_config_path().unwrap(),
            model_weights_path: path.model_weights_path().unwrap(),
            tokenizer_path: path.tokenizer_path().unwrap(),
            tokenizer_config_path: None,
        };

        let config = EncoderfileConfig {
            name: "my-model".into(),
            version: "0.0.1".into(),
            path: explicit_path,
            model_type: ModelType::Embedding,
            output_path: None,
            cache_dir: None,
            transform: None,
            tokenizer: None,
            validate_transform: false,
            base_binary_path: None,
        };

        let tokenizer = load_tokenizer_from_path(
            &config
                .path
                .tokenizer_path()
                .expect("Failed to load tokenizer"),
        )
        .expect("Failed to load tokenizer");

        let tokenizer_config = config
            .validate_tokenizer_config(&tokenizer)
            .expect("Failed to validate tokenizer config");

        assert_eq!(format!("{:?}", tokenizer_config.padding.direction), "Right");
        assert_eq!(
            format!("{:?}", tokenizer_config.padding.strategy),
            "BatchLongest"
        );
        assert_eq!(tokenizer_config.padding.pad_id, 0);
        assert_eq!(tokenizer_config.padding.pad_token, "[PAD]");
        assert!(tokenizer_config.padding.pad_to_multiple_of.is_none());
        assert_eq!(tokenizer_config.padding.pad_type_id, 0);
    }
}
