use encoderfile::config::ModelConfig;
use ort::session::Session;
use parking_lot::Mutex;
use std::{fs::File, io::BufReader};

pub const EMBEDDING_DIR: &'static str = "../models/embedding";
pub const SEQUENCE_CLASSIFICATION_DIR: &'static str = "../models/sequence_classification";
pub const TOKEN_CLASSIFICATION_DIR: &'static str = "../models/token_classification";

pub fn get_config(dir: &str) -> ModelConfig {
    let file = File::open(format!("{}/{}", dir, "config.json")).expect("Config not found");
    let reader = BufReader::new(file);

    // Deserialize into struct
    serde_json::from_reader(reader).expect("Invalid model config")
}

pub fn get_tokenizer(dir: &str) -> tokenizers::Tokenizer {
    let config = get_config(dir);
    let tokenizer_str = std::fs::read_to_string(format!("{}/{}", dir, "tokenizer.json"))
        .expect("Tokenizer json not found");

    encoderfile::tokenizer::get_tokenizer_from_string(tokenizer_str.as_str(), &config)
}

pub fn get_model<'a>(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}
