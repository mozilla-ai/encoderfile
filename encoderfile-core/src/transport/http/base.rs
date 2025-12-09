use crate::{common::model_type::ModelTypeSpec, runtime::AppState};
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
pub async fn get_model_metadata<T: ModelTypeSpec>(
    State(state): State<AppState<T>>,
) -> impl IntoResponse {
    Json(crate::services::get_model_metadata(&state))
}
