use crate::{
    common::ModelType,
    generated::{
        embedding::{
            self,
            embedding_inference_server::{EmbeddingInference, EmbeddingInferenceServer},
        },
        sentence_embedding::{
            self,
            sentence_embedding_inference_server::{
                SentenceEmbeddingInference, SentenceEmbeddingInferenceServer,
            },
        },
        sequence_classification::{
            self,
            sequence_classification_inference_server::{
                SequenceClassificationInference, SequenceClassificationInferenceServer,
            },
        },
        token_classification::{
            self,
            token_classification_inference_server::{
                TokenClassificationInference, TokenClassificationInferenceServer,
            },
        },
    },
    runtime::AppState,
};

mod error;

#[cfg(not(tarpaulin_include))]
pub fn router(state: AppState) -> axum::Router {
    let builder = tonic::service::Routes::builder().routes();

    match &state.model_type {
        ModelType::Embedding => {
            builder.add_service(EmbeddingInferenceServer::new(EmbeddingService::new(state)))
        }
        ModelType::SequenceClassification => builder.add_service(
            SequenceClassificationInferenceServer::new(SequenceClassificationService::new(state)),
        ),
        ModelType::TokenClassification => builder.add_service(
            TokenClassificationInferenceServer::new(TokenClassificationService::new(state)),
        ),
        ModelType::SentenceEmbedding => builder.add_service(SentenceEmbeddingInferenceServer::new(
            SentenceEmbeddingService::new(state),
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
                _request: tonic::Request<crate::generated::metadata::GetModelMetadataRequest>,
            ) -> Result<
                tonic::Response<crate::generated::metadata::GetModelMetadataResponse>,
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
    EmbeddingInference,
    crate::services::embedding
);

generate_grpc_server!(
    SequenceClassificationService,
    sequence_classification::SequenceClassificationRequest,
    sequence_classification::SequenceClassificationResponse,
    SequenceClassificationInference,
    crate::services::sequence_classification
);

generate_grpc_server!(
    TokenClassificationService,
    token_classification::TokenClassificationRequest,
    token_classification::TokenClassificationResponse,
    TokenClassificationInference,
    crate::services::token_classification
);

generate_grpc_server!(
    SentenceEmbeddingService,
    sentence_embedding::SentenceEmbeddingRequest,
    sentence_embedding::SentenceEmbeddingResponse,
    SentenceEmbeddingInference,
    crate::services::sentence_embedding
);
