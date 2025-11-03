use std::collections::HashMap;

use encoderfile::{
    generated::encoderfile::{
        GetModelMetadataRequest, GetModelMetadataResponse, embedding_server::Embedding,
        sequence_classification_server::SequenceClassification,
        token_classification_server::TokenClassification,
    },
    grpc::{EmbeddingService, SequenceClassificationService, TokenClassificationService},
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
