use crate::common::{Config, PadDirectionEnum, TokenizerConfig};
use crate::{common::ModelConfig, error::ApiError};
use anyhow::Result;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use tokenizers::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, tokenizer::Tokenizer,
};

static TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();

pub fn get_tokenizer(
    tokenizer_json: &str,
    config: &Arc<Config>,
    model_config: &Arc<ModelConfig>,
) -> Arc<Tokenizer> {
    TOKENIZER
        .get_or_init(|| {
            Arc::new(get_tokenizer_from_string(
                tokenizer_json,
                config,
                model_config,
            ))
        })
        .clone()
}

impl TokenizerConfig {
    pub fn padding_params(&self) -> Result<PaddingParams, ApiError> {
        let mut padding_params = PaddingParams::default();

        if let Some(strategy) = &self.pad_strategy {
            padding_params.strategy = match strategy {
                crate::common::PadStrategyEnum::Longest => PaddingStrategy::BatchLongest,
                crate::common::PadStrategyEnum::Fixed(i) => PaddingStrategy::Fixed(*i),
            };
        }

        if let Some(direction) = &self.pad_direction {
            padding_params.direction = match direction {
                PadDirectionEnum::Left => PaddingDirection::Left,
                PadDirectionEnum::Right => PaddingDirection::Right,
            };
        }

        if let Some(mo) = &self.pad_to_multiple_of {
            padding_params.pad_to_multiple_of = Some(*mo);
        }

        if let Some(pad_id) = &self.pad_id {
            padding_params.pad_id = *pad_id;
        }

        if let Some(pad_type_id) = &self.pad_type_id {
            padding_params.pad_type_id = *pad_type_id;
        }

        Ok(padding_params)
    }
}

pub fn get_tokenizer_from_string(
    s: &str,
    _config: &Arc<Config>,
    model_config: &Arc<ModelConfig>,
) -> Tokenizer {
    let pad_token_id = model_config.pad_token_id.unwrap();

    let mut tokenizer = match Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    let pad_token = match tokenizer.id_to_token(pad_token_id) {
        Some(tok) => tok,
        None => panic!("Model requires a padding token."),
    };

    if tokenizer.get_padding().is_none() {
        let params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: pad_token_id,
            pad_type_id: 0,
            pad_token,
        };

        tracing::warn!(
            "No padding strategy specified in tokenizer config. Setting default: {:?}",
            &params
        );
        tokenizer.with_padding(Some(params));
    }

    tokenizer
}

#[tracing::instrument(skip_all)]
pub fn encode_text(tokenizer: &Tokenizer, text: Vec<String>) -> Result<Vec<Encoding>, ApiError> {
    if text.is_empty() || text.iter().any(|i| i.is_empty()) {
        return Err(ApiError::InputError("Cannot tokenize empty string"));
    }

    tokenizer.encode_batch(text, true).map_err(|e| {
        tracing::error!("Error tokenizing text: {}", e);
        ApiError::InternalError("Error during tokenization")
    })
}
