use encoderfile::{
    config::ModelConfig,
    inference::{
        embedding::embedding, sequence_classification::sequence_classification,
        token_classification::token_classification, tokenizer::encode_text,
    },
};
use ort::session::Session;
use parking_lot::Mutex;
use std::{fs::File, io::BufReader};
use tokenizers::{PaddingDirection, PaddingParams, PaddingStrategy};

const EMBEDDING_DIR: &'static str = "./models/embedding";
const SEQUENCE_CLASSIFICATION_DIR: &'static str = "./models/sequence_classification";
const TOKEN_CLASSIFICATION_DIR: &'static str = "./models/token_classification";

fn get_config(dir: &str) -> ModelConfig {
    let file = File::open(format!("{}/{}", dir, "config.json")).expect("Config not found");
    let reader = BufReader::new(file);

    // Deserialize into struct
    serde_json::from_reader(reader).expect("Invalid model config")
}

fn get_tokenizer(dir: &str) -> tokenizers::Tokenizer {
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

fn get_model<'a>(dir: &str) -> Mutex<Session> {
    Mutex::new(
        ort::session::Session::builder()
            .expect("Failed to load session")
            .commit_from_file(format!("{}/{}", dir, "model.onnx"))
            .expect("Failed to load model"),
    )
}

#[test]
fn test_tokenizers() {
    for model in vec![
        EMBEDDING_DIR,
        SEQUENCE_CLASSIFICATION_DIR,
        TOKEN_CLASSIFICATION_DIR,
    ] {
        let tokenizer = get_tokenizer(model);
        encode_text(
            &tokenizer,
            vec![
                "hello world".to_string(),
                "the quick brown fox jumps over the lazy dog".to_string(),
            ],
        )
        .expect("Failed to encode text");
    }
}

#[test]
fn test_embedding_model() {
    let tokenizer = get_tokenizer(EMBEDDING_DIR);
    let session = get_model(EMBEDDING_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    let results =
        embedding(session_lock, encodings.clone(), true).expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
fn test_sequence_classification_model() {
    let tokenizer = get_tokenizer(SEQUENCE_CLASSIFICATION_DIR);
    let session = get_model(SEQUENCE_CLASSIFICATION_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    let results = sequence_classification(session_lock, encodings.clone())
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
fn test_token_classification_model() {
    let tokenizer = get_tokenizer(TOKEN_CLASSIFICATION_DIR);
    let session = get_model(TOKEN_CLASSIFICATION_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    let results =
        token_classification(session_lock, encodings.clone()).expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}
