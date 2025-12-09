use crate::{common::model_type::ModelTypeSpec, runtime::AppState, services::Inference};
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

pub async fn inference_endpoint<T: ModelTypeSpec>(
    State(state): State<AppState<T>>,
    Json(req): Json<<AppState<T> as Inference>::Input>,
) -> Result<
    Json<<AppState<T> as Inference>::Output>,
    (axum::http::StatusCode, std::borrow::Cow<'static, str>),
>
where
    AppState<T>: Inference,
{
    state
        .inference(req)
        .map(|i| Json(i))
        .map_err(|i| i.to_axum_status())
}
