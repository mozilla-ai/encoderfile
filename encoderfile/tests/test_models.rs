use encoderfile::inference::{
    embedding::embedding, sequence_classification::sequence_classification,
    token_classification::token_classification,
};
use encoderfile::runtime::tokenizer::encode_text;

mod model_utils;

use model_utils::*;

#[test]
fn test_embedding_model() {
    let state = embedding_state();

    let encodings = encode_text(
        &state.tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let results = embedding(session_lock, &state.config, encodings.clone(), true)
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_embedding_inference_with_bad_model() {
    let state = token_classification_state();

    let encodings = encode_text(
        &state.tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = state.session.lock();

    embedding(session_lock, &state.config, encodings.clone(), true)
        .expect("Failed to compute results");
}

#[test]
fn test_sequence_classification_model() {
    let state = sequence_classification_state();

    let encodings = encode_text(
        &state.tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let results = sequence_classification(session_lock, &state.config, encodings.clone())
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_sequence_classification_inference_with_bad_model() {
    let state = embedding_state();

    let encodings = encode_text(
        &state.tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = state.session.lock();

    sequence_classification(session_lock, &state.config, encodings.clone())
        .expect("Failed to compute results");
}

#[test]
fn test_token_classification_model() {
    let state = token_classification_state();

    let encodings = encode_text(
        &state.tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let results = token_classification(session_lock, &state.config, encodings.clone())
        .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_token_classification_inference_with_bad_model() {
    let state = sequence_classification_state();

    let encodings = encode_text(
        &state.tokenizer,
        vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ],
    )
    .expect("Failed to encode text");

    let session_lock = state.session.lock();

    token_classification(session_lock, &state.config, encodings.clone())
        .expect("Failed to compute results");
}
