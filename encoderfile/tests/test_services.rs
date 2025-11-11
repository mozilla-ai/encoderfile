use encoderfile::{
    common::{EmbeddingRequest, SequenceClassificationRequest, TokenClassificationRequest},
    services::{embedding, sequence_classification, token_classification},
    test_utils::*,
};

#[test]
pub fn test_embedding_service() {
    let state = embedding_state();
    let request = EmbeddingRequest {
        inputs: vec!["hello world".to_string()],
        metadata: None,
    };

    let response = embedding(request, &state).expect("Failed to compute embeddings");

    assert!(response.results.len() == 1, "Didn't return one result");
    assert!(
        response.metadata.is_none(),
        "Metadata should be returned None"
    );
}

#[test]
pub fn test_sequence_classification_service() {
    let state = sequence_classification_state();
    let request = SequenceClassificationRequest {
        inputs: vec!["hello world".to_string()],
        metadata: None,
    };

    let response = sequence_classification(request, &state).expect("Failed to compute embeddings");

    assert!(response.results.len() == 1, "Didn't return one result");
    assert!(
        response.metadata.is_none(),
        "Metadata should be returned None"
    );
}

#[test]
pub fn test_token_classification_service() {
    let state = token_classification_state();
    let request = TokenClassificationRequest {
        inputs: vec!["hello world".to_string()],
        metadata: None,
    };

    let response = token_classification(request, &state).expect("Failed to compute embeddings");

    assert!(response.results.len() == 1, "Didn't return one result");
    assert!(
        response.metadata.is_none(),
        "Metadata should be returned None"
    );
}
