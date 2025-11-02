use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    config::get_model_config,
    error::ApiError,
    inference::{self, embedding::TokenEmbedding, model::get_model, tokenizer::get_tokenizer},
};

pub fn embedding(request: impl Into<EmbeddingRequest>) -> Result<EmbeddingResponse, ApiError> {
    let request = request.into();

    let tokenizer = get_tokenizer();
    let session = get_model();
    let config = get_model_config();

    let encodings = inference::tokenizer::encode_text(tokenizer, request.inputs)?;

    let results = inference::embedding::embedding(session, config, encodings, request.normalize)?;

    Ok(EmbeddingResponse {
        results,
        model_id: crate::assets::MODEL_ID.to_string(),
        metadata: request.metadata,
    })
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
    pub normalize: bool,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<crate::generated::embedding::EmbeddingRequest> for EmbeddingRequest {
    fn from(val: crate::generated::embedding::EmbeddingRequest) -> Self {
        Self {
            inputs: val.inputs,
            normalize: val.normalize,
            metadata: Some(val.metadata),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    results: Vec<Vec<TokenEmbedding>>,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

impl From<EmbeddingResponse> for crate::generated::embedding::EmbeddingResponse {
    fn from(val: EmbeddingResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|embs| embs.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or(HashMap::new()),
        }
    }
}
