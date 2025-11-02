use axum::{
    Json,
    response::IntoResponse,
    routing::{get, post},
    extract::State,
};

use crate::{
    config::{ModelType, get_model_type},
    services, state::AppState,
};

const PREDICT_ROUTE: &'static str = "/predict";

pub fn router(state: AppState) -> axum::Router {
    let router = axum::Router::new().route("/health", get(|| async { "OK" }));

    match get_model_type() {
        ModelType::Embedding => router.route(PREDICT_ROUTE, post(embedding)),
        ModelType::SequenceClassification => {
            router.route(PREDICT_ROUTE, post(sequence_classification))
        }
        ModelType::TokenClassification => router.route(PREDICT_ROUTE, post(token_classification)),
    }
    .with_state(state)
}

macro_rules! generate_route {
    ($fn_name:ident, $request_path:path, $fn_path:path) => {
        async fn $fn_name(State(state): State<AppState>, Json(req): Json<$request_path>) -> impl IntoResponse {
            $fn_path(req, &state)
                .map(|r| Json(r))
                .map_err(|e| e.to_axum_status())
        }
    };
}

generate_route!(embedding, services::EmbeddingRequest, services::embedding);
generate_route!(
    sequence_classification,
    services::SequenceClassificationRequest,
    services::sequence_classification
);
generate_route!(
    token_classification,
    services::TokenClassificationRequest,
    services::token_classification
);
