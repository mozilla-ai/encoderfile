use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct SequenceClassificationRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl super::FromCliInput for SequenceClassificationRequest {
    fn from_cli_input(inputs: Vec<String>) -> Self {
        Self {
            inputs,
            metadata: Some(HashMap::default()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema, utoipa::ToResponse)]
pub struct SequenceClassificationResponse {
    pub results: Vec<SequenceClassificationResult>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, JsonSchema)]
pub struct SequenceClassificationResult {
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub predicted_index: u32,
    pub predicted_label: Option<String>,
}
