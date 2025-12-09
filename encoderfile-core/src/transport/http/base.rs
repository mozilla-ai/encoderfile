use crate::{common::model_type::ModelTypeSpec, runtime::AppState, services::Inference};
use axum::{Json, extract::State, response::IntoResponse};
use utoipa::{
    OpenApi, PartialSchema,
    openapi::{ContentBuilder, path::Operation},
};

pub const HEALTH_ENDPOINT: &'static str = "/health";
pub const MODEL_METADATA_ENDPOINT: &'static str = "/model";
pub const PREDICT_ENDPOINT: &'static str = "/predict";

#[derive(Debug, utoipa::OpenApi)]
#[openapi(
    paths(health, get_model_metadata, openapi),
    components(responses(crate::common::GetModelMetadataResponse,))
)]
pub struct ApiDoc;

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
    Json(crate::services::get_model_metadata(&state))
}

pub async fn predict<'de, T>(
    State(state): State<AppState<T>>,
    Json(req): Json<<AppState<T> as Inference>::Input>,
) -> Result<
    Json<<AppState<T> as Inference>::Output>,
    (axum::http::StatusCode, std::borrow::Cow<'static, str>),
>
where
    T: ModelTypeSpec,
    AppState<T>: Inference,
{
    state
        .inference(req)
        .map(Json)
        .map_err(|e| e.to_axum_status())
}

#[utoipa::path(
    get,
    path = "/openapi.json",
    responses(
        (status = 200, description = "Successful")
    )
)]
pub async fn openapi<'de, T: ModelTypeSpec>() -> impl IntoResponse
where
    T: ModelTypeSpec,
    AppState<T>: Inference,
    <AppState<T> as Inference>::Input: utoipa::ToSchema,
    <AppState<T> as Inference>::Output: utoipa::ToSchema,
{
    let mut openapi = ApiDoc::openapi();

    // insert custom /predict endpoint (thanks generics lol)
    let (path, operation) = inference_endpoint_openapi::<T>();
    openapi.paths.paths.insert(path.to_string(), operation);

    Json(openapi)
}

fn inference_endpoint_openapi<'de, T>() -> (&'static str, utoipa::openapi::path::PathItem)
where
    T: ModelTypeSpec,
    AppState<T>: Inference,
    <AppState<T> as Inference>::Input: utoipa::ToSchema,
    <AppState<T> as Inference>::Output: utoipa::ToSchema,
{
    let input_schema = <AppState<T> as Inference>::Input::schema();
    let output_schema = <AppState<T> as Inference>::Output::schema();

    let post = Operation::builder()
        .request_body(Some(
            utoipa::openapi::request_body::RequestBody::builder()
                .content(
                    "application/json",
                    ContentBuilder::new().schema(Some(input_schema)).build(),
                )
                .required(Some(utoipa::openapi::Required::True))
                .build(),
        ))
        .response(
            "200",
            utoipa::openapi::Response::builder().content(
                "application/json",
                ContentBuilder::new().schema(Some(output_schema)).build(),
            ),
        )
        .build();

    (
        PREDICT_ENDPOINT,
        utoipa::openapi::PathItem::new(utoipa::openapi::HttpMethod::Post, post),
    )
}
