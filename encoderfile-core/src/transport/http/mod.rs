use crate::{common::model_type::ModelTypeSpec, runtime::AppState, services::Inference};

mod base;
mod error;

pub trait HttpRouter: ModelTypeSpec {
    fn http_router(state: AppState<Self>) -> axum::Router;
}

impl<T: ModelTypeSpec> HttpRouter for T
where
    AppState<T>: Inference,
{
    fn http_router(state: AppState<T>) -> axum::Router {
        axum::Router::new()
            .route("/health", axum::routing::get(base::health))
            .route("/model", axum::routing::get(base::get_model_metadata))
            .route("/predict", axum::routing::post(base::predict::<T>))
            .route("/openapi.json", axum::routing::get(base::openapi))
            .with_state(state)
    }
}
