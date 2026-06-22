use mlua::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    io::{Read, Seek},
    marker::PhantomData,
    sync::Arc,
};

use ort::session::Session;
use parking_lot::Mutex;

use crate::{
    common::{
        Config, ModelConfig,
        model_type::{self, ModelType, ModelTypeSpec},
    },
    runtime::TokenizerService,
    runtime::loader::EncoderfileLoader,
    transforms::DEFAULT_LIBS,
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
    type State: Debug;
}

pub trait InputType {
    const INPUT: Input;
    fn input_type_val(&self) -> Input {
        Self::input_type()
    }
    fn input_type() -> Input {
        Self::INPUT
    }
    type State: Debug;
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TextInputState {
    // TODO check Clone impl
    pub tokenizer: TokenizerService,
    pub model_config: ModelConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageInputState {
    pub config: ImageConfig,
    pub preprocessing: ImagePreprocessing,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageConfig {
    pub num_channels: u32,
    pub image_size: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImagePreprocessing {
    pub rescale_factor: Option<f32>,
    pub image_mean: Option<Vec<f32>>,
    pub image_std: Option<Vec<f32>>,
    pub do_normalize: Option<bool>,
    pub do_rescale: Option<bool>,
    pub do_resize: Option<bool>,
    pub image_processor_type: Option<String>,
    pub size: Option<ImageSize>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ImageSize {
    pub height: Option<u32>,
    pub width: Option<u32>,
    pub shortest_edge: Option<u32>,
}

impl LuaUserData for ImageInputState {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("num_channels", |_, this| Ok(this.config.num_channels));
        fields.add_field_method_get("image_size", |_, this| Ok(this.config.image_size));
        fields.add_field_method_get("rescale_factor", |_, this| {
            Ok(this.preprocessing.rescale_factor)
        });
        fields.add_field_method_get("image_mean", |_, this| {
            Ok(this.preprocessing.image_mean.clone())
        });
        fields.add_field_method_get("image_std", |_, this| {
            Ok(this.preprocessing.image_std.clone())
        });
        fields.add_field_method_get("do_normalize", |_, this| {
            Ok(this.preprocessing.do_normalize)
        });
        fields.add_field_method_get("do_rescale", |_, this| Ok(this.preprocessing.do_rescale));
        fields.add_field_method_get("do_resize", |_, this| Ok(this.preprocessing.do_resize));
        fields.add_field_method_get("size_height", |_, this| {
            Ok(this.preprocessing.size.as_ref().and_then(|s| s.height))
        });
        fields.add_field_method_get("size_width", |_, this| {
            Ok(this.preprocessing.size.as_ref().and_then(|s| s.width))
        });
        fields.add_field_method_get("size_shortest_edge", |_, this| {
            Ok(this
                .preprocessing
                .size
                .as_ref()
                .and_then(|s| s.shortest_edge))
        });
    }
}

impl LuaUserData for TextInputState {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("model_type", |_, this| {
            Ok(this.model_config.model_type.clone())
        });
        fields.add_field_method_get("num_labels", |_, this| Ok(this.model_config.num_labels()));
        fields.add_field_method_get("id2label", |_, this| Ok(this.model_config.id2label.clone()));
        fields.add_field_method_get("label2id", |_, this| Ok(this.model_config.label2id.clone()));
    }
}

impl LuaUserData for ClassifierState {
    fn add_fields<F: LuaUserDataFields<Self>>(fields: &mut F) {
        fields.add_field_method_get("num_labels", |_, this| Ok(this.num_labels()));
        fields.add_field_method_get("id2label", |_, this| Ok(this.id2label.clone()));
        fields.add_field_method_get("label2id", |_, this| Ok(this.label2id.clone()));
    }
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

fn text_input_state_try_from_loader<'a, R>(
    loader: &mut EncoderfileLoader<'a, R>,
) -> Result<TextInputState, anyhow::Error>
where
    R: Read + Seek,
{
    let tokenizer = loader.tokenizer()?;
    let model_config = loader.model_config()?;
    Ok(TextInputState {
        tokenizer,
        model_config,
    })
}

fn image_input_state_try_from_loader<'a, R>(
    loader: &mut EncoderfileLoader<'a, R>,
) -> Result<ImageInputState, anyhow::Error>
where
    R: Read + Seek,
{
    let model_config = loader.model_config()?;
    let preprocessor_config = loader.image_preprocessor_config()?;
    Ok(ImageInputState {
        config: ImageConfig {
            num_channels: model_config
                .num_channels
                .ok_or_else(|| anyhow::anyhow!("num_channels is required for image models"))?,
            image_size: model_config.image_size,
        },
        preprocessing: ImagePreprocessing {
            rescale_factor: preprocessor_config.rescale_factor,
            image_mean: preprocessor_config.image_mean,
            image_std: preprocessor_config.image_std,
            do_normalize: preprocessor_config.do_normalize,
            do_rescale: preprocessor_config.do_rescale,
            do_resize: preprocessor_config.do_resize,
            image_processor_type: preprocessor_config.image_processor_type,
            size: preprocessor_config.size,
        },
    })
}

fn classifier_state_try_from_loader<'a, R>(
    loader: &mut EncoderfileLoader<'a, R>,
) -> Result<ClassifierState, anyhow::Error>
where
    R: Read + Seek,
{
    let model_config = loader.model_config()?.clone();
    Ok(ClassifierState {
        id2label: model_config.id2label.clone(),
        label2id: model_config.label2id.clone(),
        num_labels: model_config.num_labels(),
    })
}

fn feature_extractor_state_try_from_loader<'a, R>(
    _loader: &mut EncoderfileLoader<'a, R>,
) -> Result<FeatureExtractorState, anyhow::Error>
where
    R: Read + Seek,
{
    Ok(FeatureExtractorState {})
}

macro_rules! state_from_source_impl {
    ($base_type:tt, $state_type:ty, $state_fun:ident) => {
        impl<'a, 'borrow, R> TryFrom<&'borrow mut EncoderfileLoader<'a, R>> for $state_type
        where
            R: Read + Seek,
        {
            type Error = anyhow::Error;

            fn try_from(
                loader: &'borrow mut EncoderfileLoader<'a, R>,
            ) -> Result<Self, Self::Error> {
                $state_fun::<R>(loader)
            }
        }
    };
}

state_from_source_impl!(InputType, TextInputState, text_input_state_try_from_loader);
state_from_source_impl!(
    InputType,
    ImageInputState,
    image_input_state_try_from_loader
);
state_from_source_impl!(TaskType, ClassifierState, classifier_state_try_from_loader);
state_from_source_impl!(
    TaskType,
    FeatureExtractorState,
    feature_extractor_state_try_from_loader
);

macro_rules! input_state_impl {
    ($model_type:ty, $state_type:ty, $input:expr) => {
        impl InputType for $model_type {
            const INPUT: Input = $input;
            type State = $state_type;
        }
    };
}

input_state_impl!(model_type::Embedding, TextInputState, Input::Text);
input_state_impl!(model_type::SentenceEmbedding, TextInputState, Input::Text);
input_state_impl!(
    model_type::SequenceClassification,
    TextInputState,
    Input::Text
);
input_state_impl!(model_type::TokenClassification, TextInputState, Input::Text);
input_state_impl!(
    model_type::ImageClassification,
    ImageInputState,
    Input::Image
);

macro_rules! task_state_impl {
    ($model_type:ty, $state_type:ty, $task:expr) => {
        impl TaskType for $model_type {
            const TASK: Task = $task;
            type State = $state_type;
        }
    };
}

task_state_impl!(
    model_type::SequenceClassification,
    ClassifierState,
    Task::Classification
);
task_state_impl!(
    model_type::TokenClassification,
    ClassifierState,
    Task::Classification
);
task_state_impl!(
    model_type::ImageClassification,
    ClassifierState,
    Task::Classification
);
task_state_impl!(
    model_type::Embedding,
    FeatureExtractorState,
    Task::FeatureExtraction
);
task_state_impl!(
    model_type::SentenceEmbedding,
    FeatureExtractorState,
    Task::FeatureExtraction
);

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
    pub model_input_state: <T as InputType>::State,
    pub task_state: <T as TaskType>::State,
    pub lua_libs: Vec<mlua::StdLib>,
    _marker: PhantomData<T>,
}

impl<T: ModelTypeSpec + InputType + TaskType> EncoderfileState<T> {
    pub fn new(
        config: Config,
        session: Mutex<Session>,
        model_input_state: <T as InputType>::State,
        task_state: <T as TaskType>::State,
    ) -> EncoderfileState<T> {
        let lua_libs = match config.lua_libs {
            Some(ref libs) => Vec::<mlua::StdLib>::from(libs),
            None => DEFAULT_LIBS.to_vec(),
        };
        EncoderfileState {
            config,
            session,
            model_input_state,
            task_state,
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
