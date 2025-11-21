use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, ToSchema, JsonSchema, utoipa::ToResponse)]
pub struct EmbeddingResponse {
    pub results: Vec<TokenEmbeddingSequence>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, ToSchema, JsonSchema)]
pub struct TokenEmbeddingSequence {
    pub embeddings: Vec<TokenEmbedding>,
}

#[derive(Debug, Serialize, ToSchema, JsonSchema)]
pub struct TokenEmbedding {
    pub embedding: Vec<f32>,
    pub token_info: Option<super::token::TokenInfo>,
}
