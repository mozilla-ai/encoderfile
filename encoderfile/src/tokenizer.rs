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
// This is not ideal and will be revisited in v1.0.0.

use anyhow::Result;
use encoderfile_core::common::TokenizerConfig;
use std::str::FromStr;
use tokenizers::{PaddingParams, Tokenizer};

use crate::config::EncoderfileConfig;

impl EncoderfileConfig {
    pub fn validate_tokenizer(&self) -> Result<TokenizerConfig> {
        let tokenizer = match Tokenizer::from_str(
            std::fs::read_to_string(self.path.tokenizer_path()?)?.as_str(),
        ) {
            Ok(t) => t,
            Err(e) => anyhow::bail!("FATAL: Failed to load tokenizer: {:?}", e),
        };

        let config = match self.path.tokenizer_config_path()? {
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

        Ok(config)
    }
}

fn from_tokenizer(tokenizer: Tokenizer) -> Result<TokenizerConfig> {
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
    tokenizer: tokenizers::Tokenizer,
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
    builder.set_pad_token_id(&tokenizer)?;

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
