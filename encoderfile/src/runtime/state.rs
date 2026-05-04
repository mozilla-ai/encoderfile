use std::{marker::PhantomData, sync::Arc};
use serde::{Deserialize, Serialize};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{Config, ModelConfig, model_type::{ModelType, ModelTypeSpec}}, dev_utils::{InputType, TaskType}, runtime::TokenizerService, transforms::DEFAULT_LIBS
};

pub type AppState<T> = Arc<EncoderfileState<T>>;


pub struct TextInputState {
    pub tokenizer: TokenizerService,
    pub model_config: ModelConfig,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageInputState {
    pub num_channels: usize,
    pub height: usize,
    pub width: usize,
    pub image_size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassifierState {
    pub id2label: Option<std::collections::HashMap<u32, String>>,
    pub label2id: Option<std::collections::HashMap<String, u32>>,
    pub num_labels: Option<usize>,
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
