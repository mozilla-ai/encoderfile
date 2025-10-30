use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    error::ApiError,
    inference::{
        self, model::get_model, sequence_classification::SequenceClassificationResult,
        tokenizer::get_tokenizer,
    },
};

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

#[derive(Debug, Deserialize)]
pub struct SequenceClassificationRequest {
    pub inputs: Vec<String>,
    pub metadata: HashMap<String, String>,
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
