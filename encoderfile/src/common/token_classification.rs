use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct TokenClassificationRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl super::FromCliInput for TokenClassificationRequest {
    fn from_cli_input(inputs: Vec<String>) -> Self {
        Self {
            inputs,
            metadata: Some(HashMap::default()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema, utoipa::ToResponse)]
pub struct TokenClassificationResponse {
    pub results: Vec<TokenClassificationResult>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct TokenClassificationResult {
    pub tokens: Vec<TokenClassification>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct TokenClassification {
    pub token_info: super::token::TokenInfo,
    pub scores: Vec<f32>,
    pub label: String,
    pub score: f32,
}
