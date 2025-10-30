use crate::{
    config::{ModelType, get_model_type},
    generated::{
        embedding,
        encoderfile::{
            embedding_server::{Embedding, EmbeddingServer},
            sequence_classification_server::{
                SequenceClassification, SequenceClassificationServer,
            },
            token_classification_server::{TokenClassification, TokenClassificationServer},
        },
        sequence_classification, token_classification,
    },
};

pub fn router() -> axum::Router {
    let builder = tonic::service::Routes::builder().routes();

    match get_model_type() {
        ModelType::Embedding => builder.add_service(EmbeddingServer::new(EmbeddingService)),
        ModelType::SequenceClassification => builder.add_service(
            SequenceClassificationServer::new(SequenceClassificationService),
        ),
        ModelType::TokenClassification => {
            builder.add_service(TokenClassificationServer::new(TokenClassificationService))
        }
    }
    .into_axum_router()
}

macro_rules! generate_grpc_server {
    ($service_name:ident, $request_path:path, $response_path:path, $trait_path:path, $fn_path:path) => {
        #[derive(Debug, Default)]
        pub struct $service_name;

        #[tonic::async_trait]
        impl $trait_path for $service_name {
            async fn predict(
                &self,
                request: tonic::Request<$request_path>,
            ) -> Result<tonic::Response<$response_path>, tonic::Status> {
                Ok(tonic::Response::new(
                    $fn_path(request.into_inner())
                        .map_err(|e| e.to_tonic_status())?
                        .into(),
                ))
            }
        }
    };
}

generate_grpc_server!(
    EmbeddingService,
    embedding::EmbeddingRequest,
    embedding::EmbeddingResponse,
    Embedding,
    crate::services::embedding
);

generate_grpc_server!(
    SequenceClassificationService,
    sequence_classification::SequenceClassificationRequest,
    sequence_classification::SequenceClassificationResponse,
    SequenceClassification,
    crate::services::sequence_classification
);

generate_grpc_server!(
    TokenClassificationService,
    token_classification::TokenClassificationRequest,
    token_classification::TokenClassificationResponse,
    TokenClassification,
    crate::services::token_classification
);
