use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    error::ApiError,
    generated::embedding::TokenEmbeddingSequence,
    inference::{
        self, embedding::TokenEmbedding, inference::get_model,
        sequence_classification::SequenceClassificationResult,
        token_classification::TokenClassificationResult, tokenizer::get_tokenizer,
    },
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

pub fn sequence_classification(
    request: impl Into<SequenceClassificationRequest>,
) -> Result<SequenceClassificationResponse, ApiError> {
    let request = request.into();
    let tokenizer = get_tokenizer();
    let session = get_model();

    let encodings = inference::tokenizer::encode_text(tokenizer, request.inputs)?;

    let results = inference::sequence_classification::sequence_classification(session, encodings)?;

    Ok(SequenceClassificationResponse {
        results,
        model_id: crate::assets::MODEL_ID.to_string(),
        metadata: request.metadata,
    })
}

pub fn token_classification(
    request: impl Into<TokenClassificationRequest>,
) -> Result<TokenClassificationResponse, ApiError> {
    let request = request.into();
    let tokenizer = get_tokenizer();
    let session = get_model();

    let encodings = inference::tokenizer::encode_text(tokenizer, request.inputs)?;

    let results = inference::token_classification::token_classification(session, encodings)?;

    Ok(TokenClassificationResponse {
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
            results: val
                .results
                .into_iter()
                .map(|embs| TokenEmbeddingSequence {
                    embeddings: embs.into_iter().map(|i| i.into()).collect(),
                })
                .collect(),
            model_id: val.model_id,
            metadata: val.metadata,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SequenceClassificationRequest {
    inputs: Vec<String>,
    metadata: HashMap<String, String>,
}

impl From<crate::generated::sequence_classification::SequenceClassificationRequest>
    for SequenceClassificationRequest
{
    fn from(val: crate::generated::sequence_classification::SequenceClassificationRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: val.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SequenceClassificationResponse {
    results: Vec<SequenceClassificationResult>,
    model_id: String,
    metadata: HashMap<String, String>,
}

impl From<SequenceClassificationResponse>
    for crate::generated::sequence_classification::SequenceClassificationResponse
{
    fn from(val: SequenceClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TokenClassificationRequest {
    inputs: Vec<String>,
    metadata: HashMap<String, String>,
}

impl From<crate::generated::token_classification::TokenClassificationRequest>
    for TokenClassificationRequest
{
    fn from(val: crate::generated::token_classification::TokenClassificationRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: val.metadata,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TokenClassificationResponse {
    results: Vec<TokenClassificationResult>,
    model_id: String,
    metadata: HashMap<String, String>,
}

impl From<TokenClassificationResponse>
    for crate::generated::token_classification::TokenClassificationResponse
{
    fn from(val: TokenClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: crate::assets::MODEL_ID.to_string(),
            metadata: val.metadata,
        }
    }
}
