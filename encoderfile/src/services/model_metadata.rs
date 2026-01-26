use crate::{
    common::{GetModelMetadataResponse, model_type::ModelTypeSpec},
    runtime::AppState,
};

pub trait Metadata {
    fn metadata(&self) -> GetModelMetadataResponse;
}

impl<T: ModelTypeSpec> Metadata for AppState<T> {
    fn metadata(&self) -> GetModelMetadataResponse {
        GetModelMetadataResponse {
            model_id: self.config.name.clone(),
            model_type: T::enum_val(),
            id2label: self.model_config.id2label.clone(),
        }
    }
}
