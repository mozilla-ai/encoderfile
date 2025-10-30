use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    error::ApiError,
    inference::{self, embedding::TokenEmbedding, model::get_model, tokenizer::get_tokenizer},
};

pub fn embedding(request: impl Into<EmbeddingRequest>) -> Result<EmbeddingResponse, ApiError> {
    let request = request.into();

    let tokenizer = get_tokenizer();
    let session = get_model();

    let encodings = inference::tokenizer::encode_text(tokenizer, request.inputs)?;

    let results = inference::embedding::embedding(session, encodings, request.normalize)?;

    Ok(EmbeddingResponse {
        results,
        model_id: crate::assets::MODEL_ID.to_string(),
        metadata: request.metadata,
    })
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    inputs: Vec<String>,
    normalize: bool,
    metadata: HashMap<String, String>,
}

impl From<crate::generated::embedding::EmbeddingRequest> for EmbeddingRequest {
    fn from(val: crate::generated::embedding::EmbeddingRequest) -> Self {
        Self {
            inputs: val.inputs,
            normalize: val.normalize,
            metadata: val.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    results: Vec<Vec<TokenEmbedding>>,
    model_id: String,
    metadata: HashMap<String, String>,
}

impl From<EmbeddingResponse> for crate::generated::embedding::EmbeddingResponse {
    fn from(val: EmbeddingResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|embs| embs.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata,
        }
    }
}
