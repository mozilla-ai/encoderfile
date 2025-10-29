use crate::{assets::TOKENIZER_JSON, error::ApiError};
use anyhow::Result;
use std::str::FromStr;
use std::sync::OnceLock;
use tokenizers::{Encoding, tokenizer::Tokenizer};

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

pub fn get_tokenizer() -> &'static Tokenizer {
    TOKENIZER.get_or_init(|| match Tokenizer::from_str(TOKENIZER_JSON) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    })
}

pub async fn encode_text(text: Vec<String>) -> Result<Vec<Encoding>, ApiError> {
    if text.is_empty() || text.iter().any(|i| i.is_empty()) {
        return Err(ApiError::InputError("Cannot tokenize empty string"));
    }

    let tokenizer = get_tokenizer();

    tokenizer.encode_batch(text, true).map_err(|e| {
        tracing::error!("Error tokenizing text: {}", e);
        ApiError::InternalError("Error during tokenization")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_get_tokenizer_initializes_once() {
        let tokenizer1 = get_tokenizer();
        let tokenizer2 = get_tokenizer();

        // Should point to the same static instance
        let ptr1 = tokenizer1 as *const _;
        let ptr2 = tokenizer2 as *const _;
        assert_eq!(ptr1, ptr2, "Tokenizers should be the same instance");
    }

    #[tokio::test]
    async fn test_encode_text_basic() {
        let text = "Hello world!".to_string();
        let encoding = encode_text(vec![text.clone()])
            .await
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
