use crate::state::AppState;
use axum::{Json, extract::State, response::IntoResponse};

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Successful")
    )
)]
pub async fn health() -> impl IntoResponse {
    Json("OK!")
}

#[utoipa::path(
    get,
    path = "/model",
    responses(
        (status = 200, response = crate::common::GetModelMetadataResponse)
    ),
)]
pub async fn get_model_metadata(State(state): State<AppState>) -> impl IntoResponse {
    Json(crate::services::get_model_metadata(&state))
}
