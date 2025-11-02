use encoderfile::inference::{
    embedding::embedding, sequence_classification::sequence_classification,
    token_classification::token_classification,
};
use encoderfile::tokenizer::encode_text;

mod model_utils;

use model_utils::*;

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
    let config = get_config(EMBEDDING_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    let results = embedding(session_lock, &config, encodings.clone(), true)
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_embedding_inference_with_bad_model() {
    let tokenizer = get_tokenizer(SEQUENCE_CLASSIFICATION_DIR);
    let session = get_model(SEQUENCE_CLASSIFICATION_DIR);
    let config = get_config(SEQUENCE_CLASSIFICATION_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    embedding(session_lock, &config, encodings.clone(), true).expect("Failed to compute results");
}

#[test]
fn test_sequence_classification_model() {
    let tokenizer = get_tokenizer(SEQUENCE_CLASSIFICATION_DIR);
    let session = get_model(SEQUENCE_CLASSIFICATION_DIR);
    let config = get_config(SEQUENCE_CLASSIFICATION_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    let results = sequence_classification(session_lock, &config, encodings.clone())
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_sequence_classification_inference_with_bad_model() {
    let tokenizer = get_tokenizer(EMBEDDING_DIR);
    let session = get_model(EMBEDDING_DIR);
    let config = get_config(EMBEDDING_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    sequence_classification(session_lock, &config, encodings.clone())
        .expect("Failed to compute results");
}

#[test]
fn test_token_classification_model() {
    let tokenizer = get_tokenizer(TOKEN_CLASSIFICATION_DIR);
    let session = get_model(TOKEN_CLASSIFICATION_DIR);
    let config = get_config(TOKEN_CLASSIFICATION_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    let results = token_classification(session_lock, &config, encodings.clone())
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_token_classification_inference_with_bad_model() {
    let tokenizer = get_tokenizer(EMBEDDING_DIR);
    let session = get_model(EMBEDDING_DIR);
    let config = get_config(EMBEDDING_DIR);

    let encodings = encode_text(
        &tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = session.lock();

    token_classification(session_lock, &config, encodings.clone())
        .expect("Failed to compute results");
}
