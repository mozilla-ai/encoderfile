use std::collections::HashMap;
use std::fmt::Debug;

use crate::{
    common::{GetModelMetadataResponse, model_type::{ModelType, ModelTypeSpec}}, runtime::{AppState, TaskType, InputType},
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

impl<T: ModelTypeSpec + InputType + TaskType> Metadata for AppState<T>
    where <T as TaskType>::State: Debug,
        <T as InputType>::State: Debug,
{
    fn model_id(&self) -> String {
        self.config.name.clone()
    }

    fn model_type(&self) -> ModelType {
        T::enum_val()
    }

}
