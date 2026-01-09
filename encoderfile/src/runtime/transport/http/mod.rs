use encoderfile_core::{common::model_type::ModelTypeSpec, runtime::AppState};

mod base;
mod error;

pub trait HttpRouter: ModelTypeSpec {
    fn http_router(state: AppState<Self>) -> axum::Router;
}

macro_rules! predict_endpoint {
    ($mod_name:ident, $model_type:ident) => {
        mod $mod_name {
            use super::base;
            use axum::{Json, extract::State, response::IntoResponse};
            use encoderfile_core::{runtime::AppState, services::Inference};
            use utoipa::OpenApi;

            type ModelType = encoderfile_core::common::model_type::$model_type;
            type PredictInput = <AppState<ModelType> as Inference>::Input;
            type PredictOutput = <AppState<ModelType> as Inference>::Output;

            #[derive(Debug, utoipa::OpenApi)]
            #[openapi(
                paths(predict, base::health, base::get_model_metadata, openapi),
                components(schemas(
                    PredictInput,
                    PredictOutput,
                    encoderfile_core::common::GetModelMetadataResponse,
                ))
            )]
            pub struct ApiDoc;

            #[utoipa::path(
                        get,
                        path = base::OPENAPI_ENDPOINT,
                        responses(
                            (status = 200, description = "Successful")
                        )
                    )]
            pub async fn openapi() -> impl IntoResponse {
                Json(ApiDoc::openapi())
            }

            #[utoipa::path(
                        post,
                        path = base::PREDICT_ENDPOINT,
                        request_body = PredictInput,
                        responses(
                            (status = 200, response = PredictOutput)
                        ),
                    )]
            pub async fn predict(
                State(state): State<AppState<ModelType>>,
                Json(req): Json<PredictInput>,
            ) -> impl IntoResponse {
                super::base::predict(State(state), Json(req)).await
            }

            impl super::HttpRouter for ModelType {
                fn http_router(state: AppState<ModelType>) -> axum::Router {
                    axum::Router::new()
                        .route("/health", axum::routing::get(base::health))
                        .route(
                            "/model",
                            axum::routing::get(base::get_model_metadata::<ModelType>),
                        )
                        .route("/predict", axum::routing::post(predict))
                        .route("/openapi.json", axum::routing::get(openapi))
                        .with_state(state)
                }
            }
        }
    };
}

predict_endpoint!(embedding, Embedding);
predict_endpoint!(sequence_classification, SequenceClassification);
predict_endpoint!(token_classification, TokenClassification);
predict_endpoint!(sentence_embedding, SentenceEmbedding);
