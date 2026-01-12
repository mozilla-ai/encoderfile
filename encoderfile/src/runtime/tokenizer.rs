use crate::error::ApiError;
use anyhow::Result;
use tokenizers::{Encoding, tokenizer::Tokenizer};

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct TokenizerService {
    tokenizer: Tokenizer,
    config: crate::common::TokenizerConfig,
}

impl TokenizerService {
    pub fn new(tokenizer: Tokenizer, config: crate::common::TokenizerConfig) -> Result<Self> {
        let service = TokenizerService { tokenizer, config };

        service.init()
    }

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
