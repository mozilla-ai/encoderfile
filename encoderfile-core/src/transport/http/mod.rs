use crate::{common::model_type::ModelTypeSpec, runtime::AppState};

mod base;
mod error;

pub trait HttpRouter: ModelTypeSpec
where
    Self: Sized,
{
    fn router(state: AppState<Self>) -> axum::Router;
}

#[rustfmt::skip]
macro_rules! generate_http {
    ($model_type:ident, $fn_name:ident, $request_body:ident, $return_model:ident, $fn_path:path) => {
        pub mod $fn_name {
            use axum::{Json, extract::State, response::IntoResponse};
            use $crate::common::{$request_body, $return_model, GetModelMetadataResponse};

            type AppState_ = $crate::runtime::AppState<$crate::common::model_type::$model_type>;

            impl super::HttpRouter for $crate::common::model_type::$model_type {
                fn router(state: crate::runtime::AppState<Self>) -> axum::Router {
                    axum::Router::new()
                    .route("/health", axum::routing::get(super::base::health))
                    .route("/model", axum::routing::get(super::base::get_model_metadata))
                    .route("/predict", axum::routing::post($fn_name))
                    .route("/openapi.json", axum::routing::get(openapi))
                    .with_state(state)
                }
            }

            #[derive(Debug, utoipa::OpenApi)]
            #[openapi(
                paths(
                    super::base::health,
                    super::base::get_model_metadata,
                    $fn_name,
                    openapi
                ),
                components(
                    responses(
                        $return_model,
                        GetModelMetadataResponse,
                    )
                )
            )]
            pub struct ApiDoc;

            #[utoipa::path(
                get,
                path = "/openapi.json",
                responses(
                    (status = 200, description = "Successful")
                )
            )]
            pub async fn openapi() -> impl IntoResponse {
                use utoipa::OpenApi;

                Json(ApiDoc::openapi())
            }

            #[utoipa::path(
                post,
                path = "/predict",
                request_body = $request_body,
                responses(
                    (status = 200, response = $return_model)
                )
            )]
            pub async fn $fn_name(
                State(state): State<AppState_>,
                Json(req): Json<$request_body>,
            ) -> Result<Json<$return_model>, (axum::http::StatusCode, std::borrow::Cow<'static, str>)> {
                super::base::inference_endpoint::<crate::common::model_type::$model_type>(State(state), Json(req)).await
            }
        }
    };
}

generate_http!(
    Embedding,
    embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    crate::services::embedding
);
generate_http!(
    SequenceClassification,
    sequence_classification,
    SequenceClassificationRequest,
    SequenceClassificationResponse,
    crate::services::sequence_classification
);
generate_http!(
    TokenClassification,
    token_classification,
    TokenClassificationRequest,
    TokenClassificationResponse,
    crate::services::token_classification
);
generate_http!(
    SentenceEmbedding,
    sentence_embedding,
    SentenceEmbeddingRequest,
    SentenceEmbeddingResponse,
    crate::services::sentence_embedding
);
