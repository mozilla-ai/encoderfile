use crate::state::AppState;
use axum::{Json, extract::State, response::IntoResponse, routing::get};

pub fn get_base_router() -> axum::Router<AppState> {
    axum::Router::new()
        .route("/health", get(health))
        .route("/model", get(get_model_metadata))
}

#[utoipa::path(
    get,
    path = "/health",
    responses(
        (status = 200, description = "Successful")
    )
)]
async fn health() -> impl IntoResponse {
    Json("OK!")
}

#[utoipa::path(
    get,
    path = "/model",
    responses(
        (status = 200, response = crate::common::GetModelMetadataResponse)
    ),
)]
async fn get_model_metadata(State(state): State<AppState>) -> impl IntoResponse {
    Json(crate::services::get_model_metadata(&state))
}
