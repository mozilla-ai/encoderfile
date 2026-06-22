use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use encoderfile::{
    dev_utils::*,
    generated::{
        embedding::{
            EmbeddingRequest, EmbeddingResponse, embedding_inference_server::EmbeddingInference,
        },
        image_classification::{
            ImageClassificationRequest, ImageClassificationResponse,
            image_classification_inference_server::ImageClassificationInference,
        },
        image_types::ImageInput,
        metadata::{GetModelMetadataRequest, GetModelMetadataResponse},
        sentence_embedding::{
            SentenceEmbeddingRequest, SentenceEmbeddingResponse,
            sentence_embedding_inference_server::SentenceEmbeddingInference,
        },
        sequence_classification::{
            SequenceClassificationRequest, SequenceClassificationResponse,
            sequence_classification_inference_server::SequenceClassificationInference,
        },
        token_classification::{
            TokenClassificationRequest, TokenClassificationResponse,
            token_classification_inference_server::TokenClassificationInference,
        },
    },
    transport::grpc::GrpcService,
};

macro_rules! test_grpc_service {
    (
        $mod_name:ident,
        $create_service:expr,
        $has_labels:expr,
        $predict_request:expr,
        $predict_response_ty:ty
    ) => {
        mod $mod_name {
            use super::*;

            #[tokio::test]
            async fn test_get_model_metadata() {
                let service = $create_service;

                let request = tonic::Request::new(GetModelMetadataRequest {});

                let response: GetModelMetadataResponse = service
                    .get_model_metadata(request)
                    .await
                    .unwrap()
                    .into_inner();

                if $has_labels {
                    assert!(!response.id2label.is_empty(), "id2label is an empty dict")
                } else {
                    assert!(
                        response.id2label.is_empty(),
                        "id2label is not an empty dict"
                    );
                }
            }

            #[tokio::test]
            async fn test_predict() {
                let service = $create_service;
                let n_inps = $predict_request.inputs.len();
                let request = tonic::Request::new($predict_request);

                let response: $predict_response_ty =
                    service.predict(request).await.unwrap().into_inner();

                assert!(
                    response.results.len() == n_inps,
                    "Mismatched number of results"
                );
                assert!(response.metadata.is_empty(), "Metadata isn't empty");
            }

            #[tokio::test]
            async fn test_predict_empty() {
                let service = $create_service;
                let mut inp = $predict_request;
                inp.inputs = vec![];
                let request = tonic::Request::new(inp);

                let response = service.predict(request).await;

                let correct_err = match response {
                    Ok(_) => false,
                    Err(e) => match e.code() {
                        tonic::Code::InvalidArgument => true,
                        _ => false,
                    },
                };

                assert!(correct_err, "Empty input doesn't result in correct code")
            }
        }
    };
}

test_grpc_service!(
    embedding_grpc_tests,
    { GrpcService::new(embedding_state()) },
    false,
    EmbeddingRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        metadata: HashMap::new(),
    },
    EmbeddingResponse
);

test_grpc_service!(
    sequence_classification_tests,
    { GrpcService::new(sequence_classification_state()) },
    true,
    SequenceClassificationRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        metadata: HashMap::new(),
    },
    SequenceClassificationResponse
);

test_grpc_service!(
    token_classification_tests,
    { GrpcService::new(token_classification_state()) },
    true,
    TokenClassificationRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        metadata: HashMap::new(),
    },
    TokenClassificationResponse
);

test_grpc_service!(
    sentence_embedding_tests,
    { GrpcService::new(sentence_embedding_state()) },
    false,
    SentenceEmbeddingRequest {
        inputs: vec!["hello world".to_string(), "the quick brown fox".to_string()],
        metadata: HashMap::new(),
    },
    SentenceEmbeddingResponse
);

const TEST_IMAGE_PATH: &str = "../test-pictures/yoga01.jpg";

fn get_file_bytes(filename: &str) -> Vec<u8> {
    let mut file = File::open(filename).expect("Failed to open test image");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)
        .expect("Failed to read test image");
    buffer
}

test_grpc_service!(
    image_classification_tests,
    { GrpcService::new(image_classification_state()) },
    true,
    ImageClassificationRequest {
        inputs: [TEST_IMAGE_PATH, TEST_IMAGE_PATH]
            .iter()
            .map(|s| ImageInput {
                image: get_file_bytes(s)
            })
            .collect(),
        metadata: HashMap::new(),
    },
    ImageClassificationResponse
);
