use crate::{
    common::{
        Config, TokenizerConfig,
        model_type::{self, ModelTypeSpec},
    },
    runtime::{
        AppState,
        EncoderfileState,
        FeatureExtractorState,
        ORTSessionBuilder,
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
use std::{fs::File, io::BufReader, fmt::Debug};

const EMBEDDING_DIR: &str = "../models/embedding";
const SEQUENCE_CLASSIFICATION_DIR: &str = "../models/sequence_classification";
const TOKEN_CLASSIFICATION_DIR: &str = "../models/token_classification";
const IMAGE_CLASSIFICATION_DIR: &str = "../models/image_classification";

pub fn get_state<'a, T: ModelTypeSpec + InputType + TaskType>(dir: &'a str) -> AppState<T>
    where <T as InputType>::State: TryFrom<&'a str>,
          <<T as InputType>::State as TryFrom<&'a str>>::Error: Debug,
          <T as TaskType>::State: TryFrom<&'a str>,
          <<T as TaskType>::State as TryFrom<&'a str>>::Error: Debug,
{
    let config = Config {
        name: "my-model".to_string(),
        version: "0.0.1".to_string(),
        model_type: T::enum_val(),
        transform: None,
        lua_libs: None,
    };

    let session = get_model(dir);

    let model_input_state = <T as InputType>::State::try_from(dir).expect("could not load model input state from file");
    let model_task_state = <T as TaskType>::State::try_from(dir).expect("could not load model task state from file");

    EncoderfileState::new(
        config,
        session,
        model_input_state,
        model_task_state,
    ).into()
}

pub trait TaskTypeFromFile: TaskType {
    fn get_task_state(dir: &str) -> Result<Self::State, anyhow::Error>;
}

pub fn get_reader(dir: &str) -> BufReader<File> {
    let file = File::open(format!("{}/{}", dir, "config.json")).expect("Config not found");
    BufReader::new(file)
}

// Input types
fn get_text_input_state(dir: &str) -> Result<TextInputState, anyhow::Error> {
    let reader = get_reader(dir);
    let tokenizer = get_tokenizer(dir);
    let model_config = serde_json::from_reader(reader)?;

    Ok(TextInputState { tokenizer, model_config })
}

fn get_image_input_state(dir: &str) -> Result<ImageInputState, anyhow::Error> {
    let reader = get_reader(dir);
    let incomplete_state: ImageInputState = serde_json::from_reader(reader)?;
    Ok(ImageInputState {
        num_channels: incomplete_state.num_channels,
        height: incomplete_state.height.or(incomplete_state.image_size),
        width: incomplete_state.width.or(incomplete_state.image_size),
        image_size: incomplete_state.image_size,
    })
}

macro_rules! state_impl {
    ($input_type:ty, $state_fun:ident) => {
        impl TryFrom<&str> for $input_type {
            type Error = anyhow::Error;

            fn try_from(dir: &str) -> Result<Self, Self::Error> {
                $state_fun(dir)
            }
        }
    };
}

state_impl!(TextInputState, get_text_input_state);
state_impl!(ImageInputState, get_image_input_state);
state_impl!(ClassifierState, get_class_task_state);
state_impl!(FeatureExtractorState, get_feature_task_state);

// Task types
fn get_class_task_state(dir: &str) -> Result<ClassifierState, anyhow::Error> {
    let reader = get_reader(dir);
    let state: ClassifierState = serde_json::from_reader(reader)?;
    Ok(state)
}

fn get_feature_task_state(_dir: &str) -> Result<FeatureExtractorState, anyhow::Error> {
    Ok(FeatureExtractorState {})
}


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

pub fn image_classification_state() -> AppState<model_type::ImageClassification>
{
    get_state(IMAGE_CLASSIFICATION_DIR)
}

fn get_tokenizer(dir: &str) -> crate::runtime::TokenizerService {
    let tokenizer_str = std::fs::read_to_string(format!("{}/{}", dir, "tokenizer.json"))
        .expect("Tokenizer json not found");

    get_tokenizer_from_string(tokenizer_str.as_str())
}

fn get_model(dir: &str) -> Mutex<Session> {
    ORTSessionBuilder::default()
        .from_file(format!("{}/{}", dir, "model.onnx"))
        .expect("Failed to load model")
        .into()
}

fn get_tokenizer_from_string(s: &str) -> crate::runtime::TokenizerService {
    let tokenizer = match tokenizers::tokenizer::Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    crate::runtime::TokenizerService::new(tokenizer, TokenizerConfig::default())
        .expect("Error loading tokenizer")
}
