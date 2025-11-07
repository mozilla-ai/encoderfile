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

macro_rules! assert_code {
    ($resp:ident, $code:ident) => {{
        let correct_err = match $resp {
            Ok(_) => false,
            Err(e) => match e.code() {
                tonic::Code::$code => true,
                _ => false,
            },
        };

        assert!(correct_err, "Empty input doesn't result in correct code")
    }};
}

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
pub async fn test_embedding_service_empty() {
    let service = EmbeddingService::new(embedding_state());

    let request = EmbeddingRequest {
        inputs: vec![],
        normalize: true,
        metadata: HashMap::new(),
    };

    let response = service.predict(tonic::Request::new(request)).await;

    assert_code!(response, InvalidArgument);
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
pub async fn test_sequence_service_empty() {
    let service = SequenceClassificationService::new(sequence_classification_state());

    let request = SequenceClassificationRequest {
        inputs: vec![],
        metadata: HashMap::new(),
    };

    let response = service.predict(tonic::Request::new(request)).await;

    assert_code!(response, InvalidArgument);
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

#[tokio::test]
pub async fn test_token_cls_service_empty() {
    let service = TokenClassificationService::new(token_classification_state());

    let request = TokenClassificationRequest {
        inputs: vec![],
        metadata: HashMap::new(),
    };

    let response = service.predict(tonic::Request::new(request)).await;

    assert_code!(response, InvalidArgument);
}
