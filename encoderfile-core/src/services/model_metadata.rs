use crate::{
    common::{GetModelMetadataResponse, model_type::ModelTypeSpec},
    runtime::AppState,
};

pub fn get_model_metadata<T: ModelTypeSpec>(state: &AppState<T>) -> GetModelMetadataResponse {
    GetModelMetadataResponse {
        model_id: state.model_id.clone(),
        model_type: T::enum_val(),
        id2label: state.config.id2label.clone(),
    }
}
