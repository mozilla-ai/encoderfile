use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::ToSchema;

#[derive(Debug, Deserialize, ToSchema, JsonSchema)]
pub struct SequenceClassificationRequest {
    pub inputs: Vec<String>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<crate::generated::sequence_classification::SequenceClassificationRequest>
    for SequenceClassificationRequest
{
    fn from(val: crate::generated::sequence_classification::SequenceClassificationRequest) -> Self {
        Self {
            inputs: val.inputs,
            metadata: Some(val.metadata),
        }
    }
}

#[derive(Debug, Serialize, ToSchema, JsonSchema, utoipa::ToResponse)]
pub struct SequenceClassificationResponse {
    pub results: Vec<SequenceClassificationResult>,
    pub model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

impl From<SequenceClassificationResponse>
    for crate::generated::sequence_classification::SequenceClassificationResponse
{
    fn from(val: SequenceClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: val.model_id,
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

#[derive(Debug, Serialize, ToSchema, JsonSchema)]
pub struct SequenceClassificationResult {
    pub logits: Vec<f32>,
    pub scores: Vec<f32>,
    pub predicted_index: u32,
    pub predicted_label: Option<String>,
}

impl From<SequenceClassificationResult>
    for crate::generated::sequence_classification::SequenceClassificationResult
{
    fn from(val: SequenceClassificationResult) -> Self {
        Self {
            logits: val.logits,
            scores: val.scores,
            predicted_index: val.predicted_index,
            predicted_label: val.predicted_label,
        }
    }
}
