use std::{marker::PhantomData, sync::Arc};
use serde::{Deserialize, Serialize};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{Config, ModelConfig, model_type::{ModelType, ModelTypeSpec, self}}, runtime::TokenizerService, transforms::DEFAULT_LIBS
};

pub type AppState<T> = Arc<EncoderfileState<T>>;

#[derive(PartialEq)]
pub enum Task {
    Classification,
    FeatureExtraction,
}

#[derive(PartialEq)]
pub enum Input {
    Text,
    Image,
}

pub trait TaskType {
    const TASK: Task;
    fn task_type_val(&self) -> Task {
        Self::task_type()
    }
    fn task_type() -> Task {
        Self::TASK
    }
    type TaskState;
}

pub trait InputType {
    const INPUT: Input;
    fn input_type_val(&self) -> Input {
        Self::input_type()
    }
    fn input_type() -> Input {
        Self::INPUT
    }
    type InputState;
}

pub struct TextInputState {
    pub tokenizer: TokenizerService,
    pub model_config: ModelConfig,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageInputState {
    pub num_channels: u32,
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub image_size: Option<u32>,
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

macro_rules! input_state_impl {
    ($model_type:ty, $state_type:ty, $input:expr) => {
        impl InputType for $model_type {
            const INPUT: Input = $input;
            type InputState = $state_type;
        }
    };
}

input_state_impl!(model_type::Embedding, TextInputState, Input::Text);
input_state_impl!(model_type::SentenceEmbedding, TextInputState, Input::Text);
input_state_impl!(model_type::SequenceClassification, TextInputState, Input::Text);
input_state_impl!(model_type::TokenClassification, TextInputState, Input::Text);
input_state_impl!(model_type::ImageClassification, ImageInputState, Input::Image);

macro_rules! task_state_impl {
    ($model_type:ty, $state_type:ty, $task:expr) => {
        impl TaskType for $model_type {
            const TASK: Task = $task;
            type TaskState = $state_type;
        }
    };
}

task_state_impl!(model_type::SequenceClassification, ClassifierState, Task::Classification);
task_state_impl!(model_type::TokenClassification, ClassifierState, Task::Classification);
task_state_impl!(model_type::ImageClassification, ClassifierState, Task::Classification);
task_state_impl!(model_type::Embedding, FeatureExtractorState, Task::FeatureExtraction);
task_state_impl!(model_type::SentenceEmbedding, FeatureExtractorState, Task::FeatureExtraction);

macro_rules! input_type_impl {
    [ $( $x:ident ),* $(,)? ] => {
        impl ModelType {
            pub fn input_type(&self) -> crate::runtime::Input {
                match self {
                    $(
                        ModelType::$x => model_type::$x::input_type(),
                    )*
                }
            }
            pub fn task_type(&self) -> crate::runtime::Task {
                match self {
                    $(
                        ModelType::$x => model_type::$x::task_type(),
                    )*
                }
            }
        }   
    }
}
input_type_impl![
    Embedding,
    SequenceClassification,
    TokenClassification,
    SentenceEmbedding,
    ImageClassification
];

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
