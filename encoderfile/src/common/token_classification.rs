use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema)]
pub struct TokenClassificationRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<crate::generated::token_classification::TokenClassificationRequest>
    for TokenClassificationRequest
{
    fn from(val: crate::generated::token_classification::TokenClassificationRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

#[derive(Debug, Serialize, ToSchema, utoipa::ToResponse)]
pub struct TokenClassificationResponse {
    pub results: Vec<TokenClassificationResult>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<TokenClassificationResponse>
    for crate::generated::token_classification::TokenClassificationResponse
{
    fn from(val: TokenClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TokenClassificationResult {
    pub tokens: Vec<TokenClassification>,
}

impl From<TokenClassificationResult>
    for crate::generated::token_classification::TokenClassificationResult
{
    fn from(val: TokenClassificationResult) -> Self {
        Self {
            tokens: val.tokens.into_iter().map(|i| i.into()).collect(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema)]
pub struct TokenClassification {
    pub token_info: super::token::TokenInfo,
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub label: String,
    pub score: f32,
}

impl From<TokenClassification> for crate::generated::token_classification::TokenClassification {
    fn from(val: TokenClassification) -> Self {
        Self {
            token_info: Some(val.token_info.into()),
            logits: val.logits,
            scores: val.scores,
            label: val.label,
            score: val.score,
        }
    }
}
