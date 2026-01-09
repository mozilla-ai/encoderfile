use encoderfile_core::dev_utils::*;
use encoderfile_core::inference::{
    embedding::embedding, sequence_classification::sequence_classification,
    token_classification::token_classification,
};
use encoderfile_core::transforms::Transform;

#[test]
fn test_embedding_model() {
    let state = embedding_state();

    let encodings = state
        .tokenizer
        .encode_text(vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ])
        .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let transform = Transform::new(None).expect("Failed to create_transform");

    let results =
        embedding(session_lock, &transform, encodings.clone()).expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_embedding_inference_with_bad_model() {
    let state = token_classification_state();

    let encodings = state
        .tokenizer
        .encode_text(vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ])
        .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let transform = Transform::new(None).expect("Failed to create_transform");

    embedding(session_lock, &transform, encodings.clone()).expect("Failed to compute results");
}

#[test]
fn test_sequence_classification_model() {
    let state = sequence_classification_state();

    let encodings = state
        .tokenizer
        .encode_text(vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ])
        .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let transform = Transform::new(None).expect("Failed to create_transform");

    let results = sequence_classification(
        session_lock,
        &transform,
        &state.model_config,
        encodings.clone(),
    )
    .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_sequence_classification_inference_with_bad_model() {
    let state = embedding_state();

    let encodings = state
        .tokenizer
        .encode_text(vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ])
        .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let transform = Transform::new(None).expect("Failed to create_transform");

    sequence_classification(
        session_lock,
        &transform,
        &state.model_config,
        encodings.clone(),
    )
    .expect("Failed to compute results");
}

#[test]
fn test_token_classification_model() {
    let state = token_classification_state();

    let encodings = state
        .tokenizer
        .encode_text(vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ])
        .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let transform = Transform::new(None).expect("Failed to create_transform");

    let results = token_classification(
        session_lock,
        &transform,
        &state.model_config,
        encodings.clone(),
    )
    .expect("Failed to compute results");

    assert!(results.len() == encodings.len());
}

#[test]
#[should_panic]
fn test_token_classification_inference_with_bad_model() {
    let state = sequence_classification_state();

    let encodings = state
        .tokenizer
        .encode_text(vec![
            "hello world".to_string(),
            "the quick brown fox jumps over the lazy dog".to_string(),
        ])
        .expect("Failed to encode text");

    let session_lock = state.session.lock();

    let transform = Transform::new(None).expect("Failed to create_transform");

    token_classification(
        session_lock,
        &transform,
        &state.model_config,
        encodings.clone(),
    )
    .expect("Failed to compute results");
}
