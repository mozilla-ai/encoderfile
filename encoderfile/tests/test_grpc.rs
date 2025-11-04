use std::collections::HashMap;

use encoderfile::{
    generated::{
        embedding::{EmbeddingRequest, EmbeddingResponse},
        encoderfile::{
            GetModelMetadataRequest, GetModelMetadataResponse, embedding_server::Embedding,
            sequence_classification_server::SequenceClassification,
            token_classification_server::TokenClassification,
        },
        sequence_classification::{SequenceClassificationRequest, SequenceClassificationResponse},
        token_classification::{TokenClassificationRequest, TokenClassificationResponse},
    },
    transport::grpc::{
        EmbeddingService, SequenceClassificationService, TokenClassificationService,
    },
};

mod model_utils;

use model_utils::*;

#[tokio::test]
pub async fn test_embedding_get_model_metadata() {
    let service = EmbeddingService::new(embedding_state());

    let request = tonic::Request::new(GetModelMetadataRequest {});

    let response: GetModelMetadataResponse = service
        .get_model_metadata(request)
        .await
        .unwrap()
        .into_inner();

    assert!(
        response.id2label == HashMap::new(),
        "id2label is not an empty dict"
    );
}

#[tokio::test]
pub async fn test_embedding_service() {
    let service = EmbeddingService::new(embedding_state());

    let request = EmbeddingRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        normalize: true,
        metadata: HashMap::new(),
    };

    let response: EmbeddingResponse = service
        .predict(tonic::Request::new(request))
        .await
        .unwrap()
        .into_inner();

    assert!(response.results.len() == 2, "Mismatched number of results");

    assert!(response.metadata.is_empty(), "Metadata isn't empty");
}

#[tokio::test]
pub async fn test_sequence_get_model_metadata() {
    let service = SequenceClassificationService::new(sequence_classification_state());

    let request = tonic::Request::new(GetModelMetadataRequest {});

    let response: GetModelMetadataResponse = service
        .get_model_metadata(request)
        .await
        .unwrap()
        .into_inner();

    assert!(
        response.id2label != HashMap::new(),
        "id2label is an empty dict"
    );
}

#[tokio::test]
pub async fn test_sequence_service() {
    let service = SequenceClassificationService::new(sequence_classification_state());

    let request = SequenceClassificationRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        metadata: HashMap::new(),
    };

    let response: SequenceClassificationResponse = service
        .predict(tonic::Request::new(request))
        .await
        .unwrap()
        .into_inner();

    assert!(response.results.len() == 2, "Mismatched number of results");

    assert!(response.metadata.is_empty(), "Metadata isn't empty");
}

#[tokio::test]
pub async fn test_token_cls_get_model_metadata() {
    let service = TokenClassificationService::new(token_classification_state());

    let request = tonic::Request::new(GetModelMetadataRequest {});

    let response: GetModelMetadataResponse = service
        .get_model_metadata(request)
        .await
        .unwrap()
        .into_inner();

    assert!(
        response.id2label != HashMap::new(),
        "id2label is an empty dict"
    );
}

#[tokio::test]
pub async fn test_token_cls_service() {
    let service = TokenClassificationService::new(token_classification_state());

    let request = TokenClassificationRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        metadata: HashMap::new(),
    };

    let response: TokenClassificationResponse = service
        .predict(tonic::Request::new(request))
        .await
        .unwrap()
        .into_inner();

    assert!(response.results.len() == 2, "Mismatched number of results");

    assert!(response.metadata.is_empty(), "Metadata isn't empty");
}
