use encoderfile::config::ModelConfig;
use ort::session::Session;
use parking_lot::Mutex;
use std::{fs::File, io::BufReader};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};

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
    let pad_token_id = config.pad_token_id;

    let mut tokenizer = tokenizers::Tokenizer::from_file(format!("{}/{}", dir, "tokenizer.json"))
        .expect("Failed to load tokenizer");

    let pad_token = match tokenizer.id_to_token(pad_token_id) {
        Some(tok) => tok,
        None => panic!("Model requires a padding token."),
    };

    if tokenizer.get_padding().is_none() {
        let params = PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: pad_token_id,
            pad_type_id: 0,
            pad_token,
        };

        tracing::warn!(
            "No padding strategy specified in tokenizer config. Setting default: {:?}",
            &params
        );
        tokenizer.with_padding(Some(params));
    }

    tokenizer
}

pub fn get_model<'a>(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}
