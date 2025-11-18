use crate::{error::ApiError, runtime::config::ModelConfig};
use anyhow::Result;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, tokenizer::Tokenizer,
};

pub fn get_tokenizer_from_string(s: &str, config: &Arc<ModelConfig>) -> Tokenizer {
    let pad_token_id = config.pad_token_id;

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
