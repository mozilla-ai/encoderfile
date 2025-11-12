use crate::{
    common::ModelType,
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
    runtime::AppState,
};

#[cfg(not(tarpaulin_include))]
pub fn router(state: AppState) -> axum::Router {
    let builder = tonic::service::Routes::builder().routes();

    match &state.model_type {
        ModelType::Embedding => {
            builder.add_service(EmbeddingServer::new(EmbeddingService::new(state)))
        }
        ModelType::SequenceClassification => builder.add_service(
            SequenceClassificationServer::new(SequenceClassificationService::new(state)),
        ),
        ModelType::TokenClassification => builder.add_service(TokenClassificationServer::new(
            TokenClassificationService::new(state),
        )),
    }
    .into_axum_router()
}

macro_rules! generate_grpc_server {
    ($service_name:ident, $request_path:path, $response_path:path, $trait_path:path, $fn_path:path) => {
        #[derive(Debug)]
        pub struct $service_name {
            state: $crate::runtime::AppState,
        }

        impl $service_name {
            pub fn new(state: $crate::runtime::AppState) -> Self {
                Self { state }
            }
        }

        #[tonic::async_trait]
        impl $trait_path for $service_name {
            async fn predict(
                &self,
                request: tonic::Request<$request_path>,
            ) -> Result<tonic::Response<$response_path>, tonic::Status> {
                Ok(tonic::Response::new(
                    $fn_path(request.into_inner(), &self.state)
                        .map_err(|e| e.to_tonic_status())?
                        .into(),
                ))
            }

            async fn get_model_metadata(
                &self,
                _request: tonic::Request<crate::generated::encoderfile::GetModelMetadataRequest>,
            ) -> Result<
                tonic::Response<crate::generated::encoderfile::GetModelMetadataResponse>,
                tonic::Status,
            > {
                Ok(tonic::Response::new(
                    $crate::services::get_model_metadata(&self.state).into(),
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
