use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<crate::generated::embedding::EmbeddingRequest> for EmbeddingRequest {
    fn from(val: crate::generated::embedding::EmbeddingRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

#[derive(Debug, Serialize, ToSchema, utoipa::ToResponse)]
pub struct EmbeddingResponse {
    pub results: Vec<TokenEmbeddingSequence>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<EmbeddingResponse> for crate::generated::embedding::EmbeddingResponse {
    fn from(val: EmbeddingResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|embs| embs.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TokenEmbeddingSequence {
    pub embeddings: Vec<TokenEmbedding>,
}

impl From<TokenEmbeddingSequence> for crate::generated::embedding::TokenEmbeddingSequence {
    fn from(val: TokenEmbeddingSequence) -> Self {
        Self {
            embeddings: val.embeddings.into_iter().map(|i| i.into()).collect(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TokenEmbedding {
    pub embedding: Vec<f32>,
    pub token_info: Option<super::token::TokenInfo>,
}

impl From<TokenEmbedding> for crate::generated::embedding::TokenEmbedding {
    fn from(val: TokenEmbedding) -> Self {
        crate::generated::embedding::TokenEmbedding {
            embedding: val.embedding,
            token_info: val.token_info.map(|i| i.into()),
        }
    }
}
