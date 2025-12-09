use serde::{Serialize, de::DeserializeOwned};
use utoipa::ToSchema;

use crate::{common::model_type::ModelTypeSpec, runtime::AppState, services::Inference};

mod base;
mod error;

pub fn router<T>(state: AppState<T>) -> axum::Router
where
    T: ModelTypeSpec + 'static,
    AppState<T>: Inference,
    <AppState<T> as Inference>::Input: ToSchema + DeserializeOwned + Send + 'static,
    <AppState<T> as Inference>::Output: ToSchema + Serialize,
{
    axum::Router::new()
        .route("/health", axum::routing::get(base::health))
        .route("/model", axum::routing::get(base::get_model_metadata))
        .route("/predict", axum::routing::post(base::predict::<T>))
        .route("/openapi.json", axum::routing::get(base::openapi))
        .with_state(state)
}
