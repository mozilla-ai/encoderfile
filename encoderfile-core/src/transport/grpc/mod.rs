#[cfg(not(tarpaulin_include))]
use crate::common::model_type::{self, ModelTypeSpec};
use crate::{
    generated::{embedding, sentence_embedding, sequence_classification, token_classification},
    runtime::AppState,
};

mod error;

pub trait GrpcRouter {
    type ModelType: ModelTypeSpec;

    fn router(state: AppState<Self::ModelType>) -> axum::Router;
}

pub struct GrpcService<T: ModelTypeSpec> {
    state: crate::runtime::AppState<T>,
}

impl<T: ModelTypeSpec> GrpcService<T> {
    pub fn new(state: crate::runtime::AppState<T>) -> Self {
        Self { state }
    }
}

macro_rules! generate_grpc_server {
    (
        $model_type:ident,
        $generated_mod:ident,
        $server_mod:ident,
        $request_path:ident,
        $response_path:ident,
        $trait_path:ident,
        $server_type:ident,
        $fn_path:path
    ) => {
        impl GrpcRouter for model_type::$model_type {
            type ModelType = model_type::$model_type;

            fn router(state: AppState<Self::ModelType>) -> axum::Router {
                tonic::service::Routes::builder()
                    .routes()
                    .add_service($generated_mod::$server_mod::$server_type::new(
                        GrpcService::new(state),
                    ))
                    .into_axum_router()
            }
        }

        #[tonic::async_trait]
        impl $crate::generated::$generated_mod::$server_mod::$trait_path
            for GrpcService<model_type::$model_type>
        {
            async fn predict(
                &self,
                request: tonic::Request<$crate::generated::$generated_mod::$request_path>,
            ) -> Result<
                tonic::Response<$crate::generated::$generated_mod::$response_path>,
                tonic::Status,
            > {
                Ok(tonic::Response::new(
                    $fn_path(request.into_inner(), &self.state)
                        .map_err(|e| e.to_tonic_status())?
                        .into(),
                ))
            }

            async fn get_model_metadata(
                &self,
                _request: tonic::Request<$crate::generated::metadata::GetModelMetadataRequest>,
            ) -> Result<
                tonic::Response<$crate::generated::metadata::GetModelMetadataResponse>,
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
    Embedding,
    embedding,
    embedding_inference_server,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingInference,
    EmbeddingInferenceServer,
    crate::services::embedding
);

generate_grpc_server!(
    SequenceClassification,
    sequence_classification,
    sequence_classification_inference_server,
    SequenceClassificationRequest,
    SequenceClassificationResponse,
    SequenceClassificationInference,
    SequenceClassificationInferenceServer,
    crate::services::sequence_classification
);

generate_grpc_server!(
    TokenClassification,
    token_classification,
    token_classification_inference_server,
    TokenClassificationRequest,
    TokenClassificationResponse,
    TokenClassificationInference,
    TokenClassificationInferenceServer,
    crate::services::token_classification
);

generate_grpc_server!(
    SentenceEmbedding,
    sentence_embedding,
    sentence_embedding_inference_server,
    SentenceEmbeddingRequest,
    SentenceEmbeddingResponse,
    SentenceEmbeddingInference,
    SentenceEmbeddingInferenceServer,
    crate::services::sentence_embedding
);
