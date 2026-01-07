use crate::{common::Config, error::ApiError};
use anyhow::Result;
use std::str::FromStr;
use std::sync::{Arc, OnceLock};
use tokenizers::{Encoding, tokenizer::Tokenizer};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TokenizerService {
    tokenizer: Tokenizer,
    config: crate::common::TokenizerConfig,
}

impl TokenizerService {
    pub fn init(mut self) -> Result<Self> {
        self.tokenizer
            .with_padding(Some(self.config.padding.clone()));

        Ok(self)
    }

    #[tracing::instrument(skip_all)]
    pub fn encode_text(&self, text: Vec<String>) -> Result<Vec<Encoding>, ApiError> {
        if text.is_empty() || text.iter().any(|i| i.is_empty()) {
            return Err(ApiError::InputError("Cannot tokenize empty string"));
        }

        self.tokenizer.encode_batch(text, true).map_err(|e| {
            tracing::error!("Error tokenizing text: {}", e);
            ApiError::InternalError("Error during tokenization")
        })
    }
}

static TOKENIZER: OnceLock<Arc<TokenizerService>> = OnceLock::new();

pub fn get_tokenizer(tokenizer_json: &str, ec_config: &Arc<Config>) -> Arc<TokenizerService> {
    TOKENIZER
        .get_or_init(|| Arc::new(get_tokenizer_from_string(tokenizer_json, ec_config)))
        .clone()
}

pub fn get_tokenizer_from_string(s: &str, ec_config: &Arc<Config>) -> TokenizerService {
    let tokenizer = match Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    let config = ec_config.tokenizer.clone();

    let service = TokenizerService { tokenizer, config };

    match service.init() {
        Ok(t) => t,
        Err(e) => {
            panic!("FATAL: Error loading tokenizer: {e:?}")
        }
    }
}
