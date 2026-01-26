use std::collections::HashMap;

use crate::{
    common::{GetModelMetadataResponse, ModelType, model_type::ModelTypeSpec},
    runtime::AppState,
};

pub trait Metadata {
    fn metadata(&self) -> GetModelMetadataResponse {
        GetModelMetadataResponse {
            model_id: self.model_id(),
            model_type: self.model_type(),
            id2label: self.id2label(),
        }
    }

    fn model_id(&self) -> String;

    fn model_type(&self) -> ModelType;

    fn id2label(&self) -> Option<HashMap<u32, String>>;
}

impl<T: ModelTypeSpec> Metadata for AppState<T> {
    fn model_id(&self) -> String {
        self.config.name.clone()
    }

    fn model_type(&self) -> ModelType {
        T::enum_val()
    }

    fn id2label(&self) -> Option<HashMap<u32, String>> {
        self.model_config.id2label.clone()
    }
}
