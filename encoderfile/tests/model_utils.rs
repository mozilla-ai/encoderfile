use encoderfile::{config::{ModelConfig, ModelType}, state::AppState};
use ort::session::Session;
use parking_lot::Mutex;
use std::{fs::File, io::BufReader, sync::Arc};

pub const EMBEDDING_DIR: &'static str = "../models/embedding";
pub const SEQUENCE_CLASSIFICATION_DIR: &'static str = "../models/sequence_classification";
pub const TOKEN_CLASSIFICATION_DIR: &'static str = "../models/token_classification";

pub fn get_state(dir: &str, model_type: ModelType) -> AppState {
    let config = Arc::new(get_config(dir));
    let tokenizer = Arc::new(get_tokenizer(dir, &config));
    let session = Arc::new(get_model(dir));
    
    AppState {
        session,
        tokenizer,
        config,
        model_type
    }

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

    encoderfile::tokenizer::get_tokenizer_from_string(tokenizer_str.as_str(), &config)
}

fn get_model<'a>(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}
