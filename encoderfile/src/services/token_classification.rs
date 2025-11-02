use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{
    error::ApiError,
    inference::{self, token_classification::TokenClassificationResult},
    state::AppState,
};

pub fn token_classification(
    request: impl Into<TokenClassificationRequest>,
    state: &AppState,
) -> Result<TokenClassificationResponse, ApiError> {
    let request = request.into();
    let session = state.session.lock();

    let encodings = crate::tokenizer::encode_text(state.tokenizer, request.inputs)?;

    let results =
        inference::token_classification::token_classification(session, state.config, encodings)?;

    Ok(TokenClassificationResponse {
        results,
        model_id: crate::assets::MODEL_ID.to_string(),
        metadata: request.metadata,
    })
}

#[derive(Debug, Deserialize)]
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

#[derive(Debug, Serialize)]
pub struct TokenClassificationResponse {
    results: Vec<TokenClassificationResult>,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, String>>,
}

impl From<TokenClassificationResponse>
    for crate::generated::token_classification::TokenClassificationResponse
{
    fn from(val: TokenClassificationResponse) -> Self {
        Self {
            results: val.results.into_iter().map(|i| i.into()).collect(),
            model_id: crate::assets::MODEL_ID.to_string(),
            metadata: val.metadata.unwrap_or(HashMap::new()),
        }
    }
}
