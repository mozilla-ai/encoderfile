use crate::{
    common::model_type::ModelTypeSpec,
    runtime::AppState,
    services::{Inference, Metadata},
};
use axum::{Json, extract::State, response::IntoResponse};

pub const HEALTH_ENDPOINT: &str = "/health";
pub const MODEL_METADATA_ENDPOINT: &str = "/model";
pub const PREDICT_ENDPOINT: &str = "/predict";
pub const OPENAPI_ENDPOINT: &str = "/openapi.json";

#[utoipa::path(
    get,
    path = HEALTH_ENDPOINT,
    responses(
        (status = 200, description = "Successful")
    )
)]
pub async fn health() -> impl IntoResponse {
    Json("OK!")
}

#[utoipa::path(
    get,
    path = MODEL_METADATA_ENDPOINT,
    responses(
        (status = 200, response = crate::common::GetModelMetadataResponse)
    ),
)]
pub async fn get_model_metadata<T: ModelTypeSpec>(
    State(state): State<AppState<T>>,
) -> impl IntoResponse {
    Json(state.metadata())
}

pub async fn predict<T: ModelTypeSpec>(
    State(state): State<AppState<T>>,
    Json(req): Json<<AppState<T> as Inference>::Input>,
) -> impl IntoResponse
where
    AppState<T>: Inference,
{
    state
        .inference(req)
        .map(Json)
        .map_err(|e| e.to_axum_status())
}
