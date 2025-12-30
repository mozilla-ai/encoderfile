use crate::{common::Config, error::ApiError};
use anyhow::Result;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use tokenizers::{Encoding, tokenizer::Tokenizer};

static TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();

pub fn get_tokenizer(tokenizer_json: &str, ec_config: &Arc<Config>) -> Arc<Tokenizer> {
    TOKENIZER
        .get_or_init(|| Arc::new(get_tokenizer_from_string(tokenizer_json, ec_config)))
        .clone()
}

pub fn get_tokenizer_from_string(s: &str, ec_config: &Arc<Config>) -> Tokenizer {
    let mut tokenizer = match Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    tokenizer.with_padding(Some(ec_config.tokenizer.padding.clone()));

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
