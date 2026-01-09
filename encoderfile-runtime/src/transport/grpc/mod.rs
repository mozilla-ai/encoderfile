use encoderfile_core::{
    common::model_type::{self, ModelTypeSpec},
    generated::{embedding, sentence_embedding, sequence_classification, token_classification},
    runtime::AppState,
    services::Inference,
};

use error::ToTonicStatus;

mod error;

pub trait GrpcRouter: ModelTypeSpec
where
    Self: Sized,
{
    fn router(state: AppState<Self>) -> axum::Router;
}

pub struct GrpcService<T: ModelTypeSpec> {
    state: encoderfile_core::runtime::AppState<T>,
}

impl<T: ModelTypeSpec> GrpcService<T> {
    pub fn new(state: encoderfile_core::runtime::AppState<T>) -> Self {
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
        $server_type:ident
    ) => {
        impl GrpcRouter for model_type::$model_type {
            fn router(state: AppState<Self>) -> axum::Router {
                tonic::service::Routes::builder()
                    .routes()
                    .add_service($generated_mod::$server_mod::$server_type::new(
                        GrpcService::new(state),
                    ))
                    .into_axum_router()
            }
        }

        #[tonic::async_trait]
        impl encoderfile_core::generated::$generated_mod::$server_mod::$trait_path
            for GrpcService<model_type::$model_type>
        {
            async fn predict(
                &self,
                request: tonic::Request<encoderfile_core::generated::$generated_mod::$request_path>,
            ) -> Result<
                tonic::Response<encoderfile_core::generated::$generated_mod::$response_path>,
                tonic::Status,
            > {
                Ok(tonic::Response::new(
                    self.state
                        .inference(request.into_inner())
                        .map_err(|e| e.to_tonic_status())?
                        .into(),
                ))
            }

            async fn get_model_metadata(
                &self,
                _request: tonic::Request<
                    encoderfile_core::generated::metadata::GetModelMetadataRequest,
                >,
            ) -> Result<
                tonic::Response<encoderfile_core::generated::metadata::GetModelMetadataResponse>,
                tonic::Status,
            > {
                Ok(tonic::Response::new(
                    encoderfile_core::services::get_model_metadata(&self.state).into(),
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
    EmbeddingInferenceServer
);

generate_grpc_server!(
    SequenceClassification,
    sequence_classification,
    sequence_classification_inference_server,
    SequenceClassificationRequest,
    SequenceClassificationResponse,
    SequenceClassificationInference,
    SequenceClassificationInferenceServer
);

generate_grpc_server!(
    TokenClassification,
    token_classification,
    token_classification_inference_server,
    TokenClassificationRequest,
    TokenClassificationResponse,
    TokenClassificationInference,
    TokenClassificationInferenceServer
);

generate_grpc_server!(
    SentenceEmbedding,
    sentence_embedding,
    sentence_embedding_inference_server,
    SentenceEmbeddingRequest,
    SentenceEmbeddingResponse,
    SentenceEmbeddingInference,
    SentenceEmbeddingInferenceServer
);
