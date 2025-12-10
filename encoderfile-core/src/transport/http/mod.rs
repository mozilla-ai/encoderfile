use crate::{common::model_type::ModelTypeSpec, runtime::AppState, services::Inference};

mod base;
mod error;

pub fn router<T>(state: AppState<T>) -> axum::Router
where
    T: ModelTypeSpec,
    AppState<T>: Inference,
{
    axum::Router::new()
        .route("/health", axum::routing::get(base::health))
        .route("/model", axum::routing::get(base::get_model_metadata))
        .route("/predict", axum::routing::post(base::predict::<T>))
        .route("/openapi.json", axum::routing::get(base::openapi))
        .with_state(state)
}
