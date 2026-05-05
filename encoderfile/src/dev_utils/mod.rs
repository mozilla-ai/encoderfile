use crate::{
    common::{
        Config, TokenizerConfig,
        model_type::{self, ModelTypeSpec},
    },
    runtime::{
        AppState,
        EncoderfileState,
        FeatureExtractorState,
        ClassifierState,
        TextInputState,
        ImageInputState,
        InputType,
        TaskType,
    },
};
use ort::session::Session;
use parking_lot::Mutex;
use std::str::FromStr;
use std::{fs::File, io::BufReader};

const EMBEDDING_DIR: &str = "../models/embedding";
// CHECK sentence embedding????
const SEQUENCE_CLASSIFICATION_DIR: &str = "../models/sequence_classification";
const TOKEN_CLASSIFICATION_DIR: &str = "../models/token_classification";

pub fn get_state<T: ModelTypeSpec + InputTypeFromFile + TaskTypeFromFile>(dir: &str) -> AppState<T>
{
    let config = Config {
        name: "my-model".to_string(),
        version: "0.0.1".to_string(),
        model_type: T::enum_val(),
        transform: None,
        lua_libs: None,
    };

    let session = get_model(dir);

    EncoderfileState::new(
        config,
        session,
        T::get_input_state(dir),
        T::get_task_state(dir),
    ).into()
}

pub trait TaskTypeFromFile: TaskType {
    fn get_task_state(dir: &str) -> Self::TaskState;
}

pub trait InputTypeFromFile: InputType {
    fn get_input_state(dir: &str) -> Self::InputState;
}

pub fn get_reader(dir: &str) -> BufReader<File> {
    let file = File::open(format!("{}/{}", dir, "config.json")).expect("Config not found");
    BufReader::new(file)
}

// Input types
fn get_text_input_state(dir: &str) -> TextInputState {
    let reader = get_reader(dir);
    let tokenizer = get_tokenizer(dir);
    let model_config = serde_json::from_reader(reader).expect("Invalid model config");

    TextInputState { tokenizer, model_config }
}

fn get_image_input_state(dir: &str) -> ImageInputState {
    let reader = get_reader(dir);
    let incomplete_state: ImageInputState = serde_json::from_reader(reader).expect("Invalid model config");
    ImageInputState {
        num_channels: incomplete_state.num_channels,
        height: incomplete_state.height.or(Some(incomplete_state.image_size)),
        width: incomplete_state.width.or(Some(incomplete_state.image_size)),
        image_size: incomplete_state.image_size,
    }
}

macro_rules! input_state_impl {
    ($model_type:ty, $state_fun:ident) => {
        impl InputTypeFromFile for $model_type {
            fn get_input_state(dir: &str) -> Self::InputState {
                $state_fun(dir)
            }
        }
    };
}

input_state_impl!(model_type::SequenceClassification, get_text_input_state);
input_state_impl!(model_type::TokenClassification, get_text_input_state);
input_state_impl!(model_type::ImageClassification, get_image_input_state);
input_state_impl!(model_type::Embedding, get_text_input_state);
input_state_impl!(model_type::SentenceEmbedding, get_text_input_state);


// Task types
fn get_class_task_state(dir: &str) -> ClassifierState {
    let reader = get_reader(dir);
    serde_json::from_reader(reader).expect("Invalid model config")
}

fn get_feature_task_state(_dir: &str) -> FeatureExtractorState {
    FeatureExtractorState {}
}

macro_rules! task_state_impl {
    ($model_type:ty, $state_fun:ident) => {
        impl TaskTypeFromFile for $model_type {
            fn get_task_state(dir: &str) -> Self::TaskState {
                $state_fun(dir)
            }
        }
    };
}

task_state_impl!(model_type::SequenceClassification, get_class_task_state);
task_state_impl!(model_type::TokenClassification, get_class_task_state);
task_state_impl!(model_type::ImageClassification, get_class_task_state);
task_state_impl!(model_type::Embedding, get_feature_task_state);
task_state_impl!(model_type::SentenceEmbedding, get_feature_task_state);



pub fn embedding_state() -> AppState<model_type::Embedding>
{
    get_state(EMBEDDING_DIR)
}

pub fn sentence_embedding_state() -> AppState<model_type::SentenceEmbedding>
{
    get_state(EMBEDDING_DIR)
}

pub fn sequence_classification_state() -> AppState<model_type::SequenceClassification>
{
    get_state(SEQUENCE_CLASSIFICATION_DIR)
}

pub fn token_classification_state() -> AppState<model_type::TokenClassification>
{
    get_state(TOKEN_CLASSIFICATION_DIR)
}

fn get_tokenizer(dir: &str) -> crate::runtime::TokenizerService {
    let tokenizer_str = std::fs::read_to_string(format!("{}/{}", dir, "tokenizer.json"))
        .expect("Tokenizer json not found");

    get_tokenizer_from_string(tokenizer_str.as_str())
}

fn get_model(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}

fn get_tokenizer_from_string(s: &str) -> crate::runtime::TokenizerService {
    let tokenizer = match tokenizers::tokenizer::Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    crate::runtime::TokenizerService::new(tokenizer, TokenizerConfig::default())
        .expect("Error loading tokenizer")
}
