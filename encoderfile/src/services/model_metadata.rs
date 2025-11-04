use crate::{common::model_metadata::GetModelMetadataResponse, state::AppState};

pub fn get_model_metadata(state: &AppState) -> GetModelMetadataResponse {
    GetModelMetadataResponse {
        model_id: state.model_id.clone(),
        model_type: state.model_type.clone(),
        id2label: state.config.id2label.clone(),
    }
}
