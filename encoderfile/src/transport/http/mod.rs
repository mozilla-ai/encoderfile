use crate::{common::ModelType, runtime::AppState};

mod base;

#[rustfmt::skip]
macro_rules! generate_http {
    ($fn_name:ident, $request_body:ident, $return_model:ident, $fn_path:path) => {
        pub mod $fn_name {
            use axum::{Json, extract::State, response::IntoResponse};
            use $crate::common::{$request_body, $return_model, GetModelMetadataResponse};

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

            pub fn get_router(state: crate::runtime::AppState) -> axum::Router {
                axum::Router::new()
                    .route("/health", axum::routing::get(super::base::health))
                    .route("/model", axum::routing::get(super::base::get_model_metadata))
                    .route("/predict", axum::routing::post($fn_name))
                    .route("/openapi.json", axum::routing::get(openapi))
                    .with_state(state)
            }

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
                State(state): State<$crate::runtime::AppState>,
                Json(req): Json<$request_body>,
            ) -> Result<Json<$return_model>, (axum::http::StatusCode, &'static str)> {
                $fn_path(req, &state)
                    .map(|r| Json(r))
                    .map_err(|e| e.to_axum_status())
            }
        }
    };
}

pub fn router(state: AppState) -> axum::Router {
    match &state.model_type {
        ModelType::Embedding => embedding::get_router(state),
        ModelType::SequenceClassification => sequence_classification::get_router(state),
        ModelType::TokenClassification => token_classification::get_router(state),
    }
}

generate_http!(
    embedding,
    EmbeddingRequest,
    EmbeddingResponse,
    crate::services::embedding
);
generate_http!(
    sequence_classification,
    SequenceClassificationRequest,
    SequenceClassificationResponse,
    crate::services::sequence_classification
);
generate_http!(
    token_classification,
    TokenClassificationRequest,
    TokenClassificationResponse,
    crate::services::token_classification
);
