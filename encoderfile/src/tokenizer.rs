use crate::{
    assets::TOKENIZER_JSON,
    config::{ModelConfig, get_model_config},
    error::ApiError,
};
use anyhow::Result;
use std::str::FromStr;
use std::sync::OnceLock;
use tokenizers::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, tokenizer::Tokenizer,
};

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

pub fn get_tokenizer() -> &'static Tokenizer {
    let model_config = get_model_config();

    TOKENIZER.get_or_init(|| get_tokenizer_from_string(TOKENIZER_JSON, model_config))
}

pub fn get_tokenizer_from_string(s: &str, config: &ModelConfig) -> Tokenizer {
    let pad_token_id = config.pad_token_id;

    let mut tokenizer = match Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {:?}", e),
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

pub fn encode_text(tokenizer: &Tokenizer, text: Vec<String>) -> Result<Vec<Encoding>, ApiError> {
    if text.is_empty() || text.iter().any(|i| i.is_empty()) {
        return Err(ApiError::InputError("Cannot tokenize empty string"));
    }

    tokenizer.encode_batch(text, true).map_err(|e| {
        tracing::error!("Error tokenizing text: {}", e);
        ApiError::InternalError("Error during tokenization")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tokenizer_initializes_once() {
        let tokenizer1 = get_tokenizer();
        let tokenizer2 = get_tokenizer();

        // Should point to the same static instance
        let ptr1 = tokenizer1 as *const _;
        let ptr2 = tokenizer2 as *const _;
        assert_eq!(ptr1, ptr2, "Tokenizers should be the same instance");
    }

    #[test]
    fn test_empty_encode() {
        let tokenizer = get_tokenizer();
        let encoding = encode_text(tokenizer, vec![]);

        assert!(
            encoding.is_err(),
            "Encoder did not return error on empty input"
        );
    }

    #[test]
    fn test_empty_string_encode() {
        let tokenizer = get_tokenizer();
        let encoding = encode_text(tokenizer, vec!["hello, world!".to_string(), "".to_string()]);

        assert!(
            encoding.is_err(),
            "Encoder did not return error on empty string"
        );
    }

    #[test]
    fn test_encode_text_basic() {
        let text = "Hello world!".to_string();
        let tokenizer = get_tokenizer();
        let encoding = encode_text(tokenizer, vec![text.clone()])
            .expect("failed to encode text")
            .first()
            .expect("nothing encoded?")
            .clone();

        assert!(
            encoding.len() <= 16,
            "Encoding length should not exceed padding target"
        );
        assert!(encoding.get_tokens().len() <= 16);
        assert!(
            encoding
                .get_tokens()
                .iter()
                .any(|t| t.contains("hello") || t.contains("world")),
            "Tokens should contain text fragments"
        );
    }
}
