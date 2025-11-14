use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct SentenceEmbeddingRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<crate::generated::sentence_embedding::SentenceEmbeddingRequest>
    for SentenceEmbeddingRequest
{
    fn from(val: crate::generated::sentence_embedding::SentenceEmbeddingRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

#[derive(Debug, Serialize, ToSchema, JsonSchema, utoipa::ToResponse)]
pub struct SentenceEmbeddingResponse {
    pub results: Vec<SentenceEmbedding>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<crate::generated::sentence_embedding::SentenceEmbeddingResponse>
    for SentenceEmbeddingResponse
{
    fn from(val: crate::generated::sentence_embedding::SentenceEmbeddingResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: Some(val.metadata),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct SentenceEmbedding {
    embedding: Vec<f32>,
}

impl From<crate::generated::sentence_embedding::SentenceEmbedding> for SentenceEmbedding {
    fn from(val: crate::generated::sentence_embedding::SentenceEmbedding) -> Self {
        Self {
            embedding: val.embedding,
        }
    }
}
