use encoderfile::{
    common::{
        EmbeddingRequest, SentenceEmbeddingRequest, SequenceClassificationRequest,
        TokenClassificationRequest,
    },
    dev_utils::*,
    services::Inference,
};

#[test]
pub fn test_embedding_service() {
    let state = embedding_state();
    let request = EmbeddingRequest {
        inputs: vec!["hello world".to_string()],
        metadata: None,
    };

    let response = state
        .inference(request)
        .expect("Failed to compute embeddings");

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

    let response = state
        .inference(request)
        .expect("Failed to compute embeddings");

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

    let response = state
        .inference(request)
        .expect("Failed to compute embeddings");

    assert!(response.results.len() == 1, "Didn't return one result");
    assert!(
        response.metadata.is_none(),
        "Metadata should be returned None"
    );
}

#[test]
pub fn test_sentence_embedding_service() {
    let state = sentence_embedding_state();
    let request = SentenceEmbeddingRequest {
        inputs: vec!["hello world".to_string()],
        metadata: None,
    };

    let response = state
        .inference(request)
        .expect("Failed to compute embeddings");

    assert!(response.results.len() == 1, "Didn't return one result");
    assert!(
        response.metadata.is_none(),
        "Metadata should be returned None"
    );
}
