use std::collections::HashMap;

use crate::{
    common::{
        GetModelMetadataResponse,
        model_type::{ModelType, ModelTypeSpec},
    },
    runtime::{AppState, ClassifierState, FeatureExtractorState, InputType, TaskType},
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

trait TaskStateMetadata {
    fn id2label(&self) -> Option<HashMap<u32, String>>;
}

impl TaskStateMetadata for ClassifierState {
    fn id2label(&self) -> Option<HashMap<u32, String>> {
        println!("ClassifierState: {:?}", self);
        self.id2label.clone()
    }
}

impl TaskStateMetadata for FeatureExtractorState {
    fn id2label(&self) -> Option<HashMap<u32, String>> {
        None
    }
}

impl<T: ModelTypeSpec + InputType + TaskType> Metadata for AppState<T>
where
    <T as TaskType>::State: TaskStateMetadata,
{
    fn model_id(&self) -> String {
        self.config.name.clone()
    }

    fn model_type(&self) -> ModelType {
        T::enum_val()
    }

    fn id2label(&self) -> Option<HashMap<u32, String>> {
        self.task_state.id2label()
    }
}
