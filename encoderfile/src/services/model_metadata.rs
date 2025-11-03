use std::collections::HashMap;
use serde::Serialize;

use crate::{config::ModelType, state::AppState};

pub fn get_model_metadata(state: &AppState) -> GetModelMetadataResponse {
    GetModelMetadataResponse {
        model_id: state.model_id.clone(),
        model_type: state.model_type.clone(),
        id2label: state.config.id2label.clone(),
    }
}

#[derive(Debug, Serialize)]
pub struct GetModelMetadataResponse {
    model_id: String,
    model_type: ModelType,
    id2label: Option<HashMap<u32, String>>,
}

impl From<GetModelMetadataResponse> for crate::generated::encoderfile::GetModelMetadataResponse {
    fn from(val: GetModelMetadataResponse) -> Self {
        Self {
            model_id: val.model_id,
            model_type: crate::generated::encoderfile::ModelType::from(val.model_type).into(),
            id2label: val.id2label.unwrap_or(HashMap::new()),
        }
    }
}
