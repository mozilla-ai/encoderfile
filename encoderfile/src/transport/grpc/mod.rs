use crate::common::model_type;
use crate::{
    generated::{embedding, sentence_embedding, sequence_classification, token_classification},
    runtime::AppState,
    services::{Inference, Metadata},
};

mod error;

pub trait GrpcRouter
where
    Self: Sized + Clone + Send + Sync + 'static,
{
    fn grpc_router(self) -> axum::Router;
}

pub struct GrpcService<S: Inference + Metadata> {
    state: S,
}

impl<S: Inference + Metadata> GrpcService<S> {
    pub fn new(state: S) -> Self {
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
        impl GrpcRouter for AppState<model_type::$model_type> {
            fn grpc_router(self) -> axum::Router {
                tonic::service::Routes::builder()
                    .routes()
                    .add_service($generated_mod::$server_mod::$server_type::new(
                        GrpcService::new(self),
                    ))
                    .into_axum_router()
            }
        }

        #[tonic::async_trait]
        impl $crate::generated::$generated_mod::$server_mod::$trait_path
            for GrpcService<AppState<model_type::$model_type>>
        {
            async fn predict(
                &self,
                request: tonic::Request<$crate::generated::$generated_mod::$request_path>,
            ) -> Result<
                tonic::Response<$crate::generated::$generated_mod::$response_path>,
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
                _request: tonic::Request<$crate::generated::metadata::GetModelMetadataRequest>,
            ) -> Result<
                tonic::Response<$crate::generated::metadata::GetModelMetadataResponse>,
                tonic::Status,
            > {
                Ok(tonic::Response::new(self.state.metadata().into()))
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
