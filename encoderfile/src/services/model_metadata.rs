use std::collections::HashMap;

use crate::{
    common::{GetModelMetadataResponse, model_type::{ModelType, ModelTypeSpec}}, dev_utils::TaskType, dev_utils::InputType, runtime::AppState
};

pub trait ClassifierMetadata {
    fn id2label(&self) -> Option<HashMap<u32, String>>;
}

pub trait Metadata {
    fn metadata(&self) -> GetModelMetadataResponse {
        GetModelMetadataResponse {
            model_id: self.model_id(),
            model_type: self.model_type(),
            id2label: None,
        }
    }

    fn model_id(&self) -> String;

    fn model_type(&self) -> ModelType;

}

impl<T: ModelTypeSpec + InputType + TaskType> Metadata for AppState<T> {
    fn model_id(&self) -> String {
        self.config.name.clone()
    }

    fn model_type(&self) -> ModelType {
        T::enum_val()
    }

}
