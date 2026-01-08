use crate::{
    common::{
        Config, ModelConfig,
        model_type::{self, ModelTypeSpec},
    },
    runtime::AppState,
};
use ort::session::Session;
use parking_lot::Mutex;
use std::str::FromStr;
use std::{fs::File, io::BufReader, sync::Arc};

const EMBEDDING_DIR: &str = "../models/embedding";
const SEQUENCE_CLASSIFICATION_DIR: &str = "../models/sequence_classification";
const TOKEN_CLASSIFICATION_DIR: &str = "../models/token_classification";

pub fn get_state<T: ModelTypeSpec>(dir: &str) -> AppState<T> {
    let config = Arc::new(Config {
        name: "my-model".to_string(),
        version: "0.0.1".to_string(),
        model_type: T::enum_val(),
        transform: None,
        tokenizer: Default::default(),
    });

    let model_config = Arc::new(get_model_config(dir));
    let tokenizer = Arc::new(get_tokenizer(dir, &config));
    let session = Arc::new(get_model(dir));

    AppState::new(config, session, tokenizer, model_config)
}

pub fn embedding_state() -> AppState<model_type::Embedding> {
    get_state(EMBEDDING_DIR)
}

pub fn sentence_embedding_state() -> AppState<model_type::SentenceEmbedding> {
    get_state(EMBEDDING_DIR)
}

pub fn sequence_classification_state() -> AppState<model_type::SequenceClassification> {
    get_state(SEQUENCE_CLASSIFICATION_DIR)
}

pub fn token_classification_state() -> AppState<model_type::TokenClassification> {
    get_state(TOKEN_CLASSIFICATION_DIR)
}

fn get_model_config(dir: &str) -> ModelConfig {
    let file = File::open(format!("{}/{}", dir, "config.json")).expect("Config not found");
    let reader = BufReader::new(file);

    // Deserialize into struct
    serde_json::from_reader(reader).expect("Invalid model config")
}

fn get_tokenizer(dir: &str, ec_config: &Arc<Config>) -> crate::runtime::TokenizerService {
    let tokenizer_str = std::fs::read_to_string(format!("{}/{}", dir, "tokenizer.json"))
        .expect("Tokenizer json not found");

    get_tokenizer_from_string(tokenizer_str.as_str(), ec_config)
}

fn get_model(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}

fn get_tokenizer_from_string(s: &str, ec_config: &Arc<Config>) -> crate::runtime::TokenizerService {
    let tokenizer = match tokenizers::tokenizer::Tokenizer::from_str(s) {
        Ok(t) => t,
        Err(e) => panic!("FATAL: Error loading tokenizer: {e:?}"),
    };

    let config = ec_config.tokenizer.clone();

    crate::runtime::TokenizerService::new(tokenizer, config).expect("Error loading tokenizer")
}
