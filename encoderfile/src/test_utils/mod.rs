use crate::{
    common::ModelType,
    runtime::{AppState, ModelConfig},
    transforms::Transform,
};
use ort::session::Session;
use parking_lot::Mutex;
use std::{fs::File, io::BufReader, sync::Arc};

const EMBEDDING_DIR: &str = "../models/embedding";
const SEQUENCE_CLASSIFICATION_DIR: &str = "../models/sequence_classification";
const TOKEN_CLASSIFICATION_DIR: &str = "../models/token_classification";

pub fn get_state(dir: &str, model_type: ModelType) -> AppState {
    let config = Arc::new(get_config(dir));
    let tokenizer = Arc::new(get_tokenizer(dir, &config));
    let session = Arc::new(get_model(dir));

    AppState {
        session,
        tokenizer,
        config,
        model_type,
        model_id: "test-model".to_string(),
        transform_factory: || Transform::new("").unwrap(),
    }
}

pub fn embedding_state() -> AppState {
    get_state(EMBEDDING_DIR, ModelType::Embedding)
}

pub fn sentence_embedding_state() -> AppState {
    get_state(EMBEDDING_DIR, ModelType::SentenceEmbedding)
}

pub fn sequence_classification_state() -> AppState {
    get_state(
        SEQUENCE_CLASSIFICATION_DIR,
        ModelType::SequenceClassification,
    )
}

pub fn token_classification_state() -> AppState {
    get_state(TOKEN_CLASSIFICATION_DIR, ModelType::TokenClassification)
}

fn get_config(dir: &str) -> ModelConfig {
    let file = File::open(format!("{}/{}", dir, "config.json")).expect("Config not found");
    let reader = BufReader::new(file);

    // Deserialize into struct
    serde_json::from_reader(reader).expect("Invalid model config")
}

fn get_tokenizer(dir: &str, config: &Arc<ModelConfig>) -> tokenizers::Tokenizer {
    let tokenizer_str = std::fs::read_to_string(format!("{}/{}", dir, "tokenizer.json"))
        .expect("Tokenizer json not found");

    crate::runtime::get_tokenizer_from_string(tokenizer_str.as_str(), config)
}

fn get_model(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}
