use encoderfile::{
    common::{EmbeddingRequest, SequenceClassificationRequest, TokenClassificationRequest},
    test_utils::{embedding_state, sequence_classification_state, token_classification_state},
    transport::http::{embedding::embedding, sequence_classification::sequence_classification, token_classification::token_classification}
};
use axum::{Json, extract::State, http::StatusCode};

macro_rules! test_empty_input {
    ($state:expr, $req:expr, $route_fn:ident) => {{
        let (code, _msg) = $route_fn(State($state), Json($req))
            .await
            .err()
            .expect("Did not error on empty input");

        assert_eq!(code, StatusCode::UNPROCESSABLE_ENTITY);
    }};
}

macro_rules! test_successful_route {
    ($state:expr, $req:expr, $route_fn:ident) => {{
        let req = $req;
        let n_inputs = req.inputs.len();
        let metadata_is_none = req.metadata.is_none();

        let Json(resp) = $route_fn(State($state), Json(req))
            .await
            .expect("Expected successful call");

        assert_eq!(resp.metadata.is_none(), metadata_is_none);
        assert_eq!(resp.results.len(), n_inputs);
    }}
}

#[tokio::test]
async fn test_embedding_route() {
    test_successful_route!(
        embedding_state(),
        EmbeddingRequest {
            inputs: vec!["This is a test".to_string(), "This is also a test".to_string()],
            normalize: true,
            metadata: None,
        },
        embedding
    );
}

#[tokio::test]
async fn test_embedding_route_empty() {
    test_empty_input!(
        embedding_state(),
        EmbeddingRequest {
            inputs: vec![],
            normalize: true,
            metadata: None,
        },
        embedding
    );
}

#[tokio::test]
async fn test_sequence_classification_route () {
    test_successful_route!(
        sequence_classification_state(),
        SequenceClassificationRequest {
            inputs: vec!["this is a test".to_string(), "this is also a test".to_string()],
            metadata: None,
        },
        sequence_classification
    )
}

#[tokio::test]
async fn test_sequence_classification_route_empty() {
    test_empty_input!(
        sequence_classification_state(),
        SequenceClassificationRequest {
            inputs: vec![],
            metadata: None,
        },
        sequence_classification
    );
}

#[tokio::test]
async fn test_token_classification_route() {
    test_successful_route!(
        token_classification_state(),
        TokenClassificationRequest {
            inputs: vec!["this is a test".to_string(), "this is also a test".to_string()],
            metadata: None,
        },
        token_classification
    )
}

#[tokio::test]
async fn test_token_classification_route_empty() {
    test_empty_input!(
        token_classification_state(),
        TokenClassificationRequest {
            inputs: vec![],
            metadata: None,
        },
        token_classification
    );
}
