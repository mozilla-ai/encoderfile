use std::{marker::PhantomData, sync::Arc};
use serde::{Deserialize, Serialize};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{Config, ModelConfig, model_type::{ModelType, ModelTypeSpec, self}}, runtime::TokenizerService, transforms::DEFAULT_LIBS
};

pub type AppState<T> = Arc<EncoderfileState<T>>;

pub trait TaskType {
    type TaskState;
    // fn get_task_state(dir: &str) -> Self::TaskState;
}

pub trait InputType {
    type InputState;
    // fn get_input_state(dir: &str) -> Self::InputState;
}

macro_rules! input_state_impl {
    ($model_type:ty, $state_type:ty) => {
        impl InputType for $model_type {
            type InputState = $state_type;
        }
    };
}

input_state_impl!(model_type::Embedding, TextInputState);
input_state_impl!(model_type::SentenceEmbedding, TextInputState);
input_state_impl!(model_type::SequenceClassification, TextInputState);
input_state_impl!(model_type::TokenClassification, TextInputState);
input_state_impl!(model_type::ImageClassification, ImageInputState);

macro_rules! task_state_impl {
    ($model_type:ty, $state_type:ty) => {
        impl TaskType for $model_type {
            type TaskState = $state_type;
        }
    };
}

task_state_impl!(model_type::SequenceClassification, ClassifierState);
task_state_impl!(model_type::TokenClassification, ClassifierState);
task_state_impl!(model_type::ImageClassification, ClassifierState);
task_state_impl!(model_type::Embedding, FeatureExtractorState);
task_state_impl!(model_type::SentenceEmbedding, FeatureExtractorState);


pub struct TextInputState {
    pub tokenizer: TokenizerService,
    pub model_config: ModelConfig,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageInputState {
    pub num_channels: usize,
    pub height: Option<usize>,
    pub width: Option<usize>,
    pub image_size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassifierState {
    pub id2label: Option<std::collections::HashMap<u32, String>>,
    pub label2id: Option<std::collections::HashMap<String, u32>>,
    pub num_labels: Option<usize>,
}
impl ClassifierState {
    pub fn id2label(&self, id: u32) -> Option<&str> {
        self.id2label.as_ref()?.get(&id).map(|s| s.as_str())
    }

    pub fn label2id(&self, label: &str) -> Option<u32> {
        self.label2id.as_ref()?.get(label).copied()
    }

    pub fn num_labels(&self) -> Option<usize> {
        if self.num_labels.is_some() {
            return self.num_labels;
        }

        if let Some(id2label) = &self.id2label {
            return Some(id2label.len());
        }

        if let Some(label2id) = &self.label2id {
            return Some(label2id.len());
        }

        None
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureExtractorState {}

#[derive(Debug)]
pub struct EncoderfileState<T: ModelTypeSpec + InputType + TaskType> {
    pub config: Config,
    pub session: Mutex<Session>,
    pub per_model_input_state: T::InputState,
    pub per_task_state: T::TaskState,
    pub lua_libs: Vec<mlua::StdLib>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec + InputType + TaskType> EncoderfileState<T> {
    pub fn new(
        config: Config,
        session: Mutex<Session>,
        per_model_input_state: T::InputState,
        per_task_state: T::TaskState,
    ) -> EncoderfileState<T> {
        let lua_libs = match config.lua_libs {
            Some(ref libs) => Vec::<mlua::StdLib>::from(libs),
            None => DEFAULT_LIBS.to_vec(),
        };
        EncoderfileState {
            config,
            session,
            per_model_input_state,
            per_task_state,
            lua_libs,
            _marker: PhantomData,
        }
    }

    pub fn transform_str(&self) -> Option<String> {
        self.config.transform.clone()
    }

    pub fn lua_libs(&self) -> &Vec<mlua::StdLib> {
        &self.lua_libs
    }

    pub fn model_type() -> ModelType {
        T::enum_val()
    }
}
