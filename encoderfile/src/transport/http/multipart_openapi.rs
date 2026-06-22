use crate::common::model_type::ImageClassification;
use crate::common::{ImageClassificationRequest, ImageClassificationResponse, ImageInfo};
use crate::runtime::AppState;
use crate::services::Inference;
use axum::{
    Json,
    extract::{Multipart, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use utoipa::OpenApi;

pub const MULTIPART_PREDICT_ENDPOINT: &str = "/predict/multipart";
pub const MULTIPART_OPENAPI_ENDPOINT: &str = "/predict/multipart/openapi.json";

#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct MultipartPredictBody {
    /// Arbitrary JSON payload sent in the multipart part named `payload`.
    pub payload: serde_json::Value,

    /// Binary attachments sent as repeated `files` multipart parts.
    #[schema(value_type = Vec<String>)]
    pub files: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, utoipa::ToSchema)]
pub struct ParsedAttachment {
    pub file_name: Option<String>,
    pub content_type: Option<String>,
    pub size_bytes: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum MultipartApiError {
    #[error("missing required multipart field 'payload'")]
    MissingPayload,
    #[error("invalid json in 'payload' field")]
    InvalidPayload,
    #[error("multipart parse error: {0}")]
    Multipart(String),
    #[error("failed to construct request from multipart: {0}")]
    RequestConstruction(String),
}

impl IntoResponse for MultipartApiError {
    fn into_response(self) -> Response {
        let status = match self {
            Self::MissingPayload | Self::InvalidPayload => StatusCode::UNPROCESSABLE_ENTITY,
            Self::RequestConstruction(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Multipart(_) => StatusCode::BAD_REQUEST,
        };

        (status, self.to_string()).into_response()
    }
}

/// Trait for converting multipart payload and attachments into a typed request.
pub trait FromMultipart: Sized {
    /// Construct an instance from a JSON payload and list of attachment bytes.
    fn from_multipart(
        payload: serde_json::Value,
        attachments: Vec<(Option<String>, Option<String>, bytes::Bytes)>,
    ) -> Result<Self, MultipartApiError>;
}

impl FromMultipart for ImageClassificationRequest {
    fn from_multipart(
        payload: serde_json::Value,
        attachments: Vec<(Option<String>, Option<String>, bytes::Bytes)>,
    ) -> Result<Self, MultipartApiError> {
        let images = attachments
            .into_iter()
            .map(|(_file_name, _content_type, image_bytes)| {
                let format = image::guess_format(&image_bytes).map_err(|e| {
                    MultipartApiError::RequestConstruction(format!(
                        "Failed to detect image format: {}",
                        e
                    ))
                })?;
                Ok(ImageInfo {
                    image_bytes,
                    image_format: format,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let metadata = if payload.is_null() || payload == serde_json::json!({}) {
            Some(HashMap::default())
        } else {
            serde_json::from_value(payload)
                .ok()
                .or(Some(HashMap::default()))
        };

        Ok(Self { images, metadata })
    }
}

#[derive(Debug, utoipa::OpenApi)]
#[openapi(
    paths(post_multipart),
    components(schemas(MultipartPredictBody, ImageClassificationResponse, ParsedAttachment))
)]
pub struct MultipartApiDoc;

#[utoipa::path(
    get,
    path = MULTIPART_OPENAPI_ENDPOINT,
    responses(
        (status = 200, description = "Successful")
    )
)]
pub async fn openapi() -> impl IntoResponse {
    Json(MultipartApiDoc::openapi())
}

#[utoipa::path(
    post,
    path = MULTIPART_PREDICT_ENDPOINT,
    request_body(
        content = MultipartPredictBody,
        content_type = "multipart/form-data",
        description = "Multipart payload with a JSON part named 'payload' and 0..N binary parts named 'files'"
    ),
    responses(
        (status = 200, body = ImageClassificationResponse),
        (status = 422, description = "Missing or invalid payload JSON"),
        (status = 400, description = "Invalid multipart body")
    )
)]
pub async fn post_multipart(
    state: State<AppState<ImageClassification>>,
    mut multipart: Multipart,
) -> Result<Json<ImageClassificationResponse>, MultipartApiError> {
    parse_multipart(state, &mut multipart).await
}

/// Generic multipart parser that extracts payload and attachments.
pub async fn parse_multipart(
    State(state): State<AppState<ImageClassification>>,
    multipart: &mut Multipart,
) -> Result<Json<ImageClassificationResponse>, MultipartApiError> {
    let mut payload: Option<serde_json::Value> = None;
    let mut attachments = Vec::new();
    let mut attachment_metadata = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| MultipartApiError::Multipart(e.to_string()))?
    {
        let name = field.name().map(ToOwned::to_owned);
        let file_name = field.file_name().map(ToOwned::to_owned);
        let content_type = field.content_type().map(ToOwned::to_owned);
        let bytes = field
            .bytes()
            .await
            .map_err(|e| MultipartApiError::Multipart(e.to_string()))?;

        match name.as_deref() {
            Some("payload") => {
                payload = Some(
                    serde_json::from_slice(&bytes)
                        .map_err(|_| MultipartApiError::InvalidPayload)?,
                );
            }
            Some("files") => {
                attachment_metadata.push(ParsedAttachment {
                    file_name: file_name.clone(),
                    content_type: content_type.clone(),
                    size_bytes: bytes.len(),
                });
                attachments.push((file_name, content_type, bytes));
            }
            _ => {}
        }
    }

    let payload = payload.ok_or(MultipartApiError::MissingPayload)?;

    // Convert to typed request
    let request = ImageClassificationRequest::from_multipart(payload.clone(), attachments)?;
    let result = state
        .inference(request)
        .map(Json)
        .map_err(|e| MultipartApiError::RequestConstruction(format!("Inference error: {}", e)))?;

    Ok(result)
}

/// Generic handler that converts multipart request into typed request.
pub async fn post_multipart_typed<R: FromMultipart>(
    State(state): State<AppState<ImageClassification>>,
    mut multipart: Multipart,
) -> Result<Json<ImageClassificationResponse>, MultipartApiError> {
    let mut payload: Option<serde_json::Value> = None;
    let mut attachments = Vec::new();
    let mut attachment_metadata = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| MultipartApiError::Multipart(e.to_string()))?
    {
        let name = field.name().map(ToOwned::to_owned);
        let file_name = field.file_name().map(ToOwned::to_owned);
        let content_type = field.content_type().map(ToOwned::to_owned);
        let bytes = field
            .bytes()
            .await
            .map_err(|e| MultipartApiError::Multipart(e.to_string()))?;

        match name.as_deref() {
            Some("payload") => {
                payload = Some(
                    serde_json::from_slice(&bytes)
                        .map_err(|_| MultipartApiError::InvalidPayload)?,
                );
            }
            Some("files") => {
                attachment_metadata.push(ParsedAttachment {
                    file_name: file_name.clone(),
                    content_type: content_type.clone(),
                    size_bytes: bytes.len(),
                });
                attachments.push((file_name, content_type, bytes));
            }
            _ => {}
        }
    }

    let payload = payload.ok_or(MultipartApiError::MissingPayload)?;

    // Convert to typed request
    let request = ImageClassificationRequest::from_multipart(payload.clone(), attachments)?;
    let result = state
        .inference(request)
        .map(Json)
        .map_err(|e| MultipartApiError::RequestConstruction(format!("Inference error: {}", e)))?;

    Ok(result)
}

/// HttpRouter implementation for ImageClassification model type.
/// Combines standard model serving endpoints with multipart file upload capability.
impl super::HttpRouter for crate::runtime::AppState<ImageClassification> {
    fn http_router(self) -> axum::Router {
        axum::Router::new()
            .route("/health", axum::routing::get(super::base::health))
            .route(
                "/model",
                axum::routing::get(super::base::get_model_metadata::<Self>),
            )
            .route("/predict", axum::routing::post(predict_handler))
            .route("/openapi.json", axum::routing::get(standard_openapi))
            .route(
                MULTIPART_PREDICT_ENDPOINT,
                axum::routing::post(post_multipart_image_classification),
            )
            .route(MULTIPART_OPENAPI_ENDPOINT, axum::routing::get(openapi))
            .with_state(self)
    }
}

/// Multipart handler specialized for ImageClassificationRequest.
async fn post_multipart_image_classification(
    state: State<AppState<ImageClassification>>,
    multipart: Multipart,
) -> Result<Json<ImageClassificationResponse>, MultipartApiError> {
    post_multipart_typed::<crate::common::ImageClassificationRequest>(state, multipart).await
}

/// Standard predict endpoint for ImageClassification.
async fn predict_handler(
    State(state): State<AppState<ImageClassification>>,
    Json(req): Json<<AppState<ImageClassification> as crate::services::Inference>::Input>,
) -> impl IntoResponse {
    super::base::predict(State(state), Json(req)).await
}

/// Standard OpenAPI endpoint for ImageClassification model service (without multipart).
async fn standard_openapi() -> impl IntoResponse {
    Json(serde_json::json!({
        "openapi": "3.0.0",
        "info": {
            "title": "ImageClassification Model API",
            "version": "1.0.0"
        },
        "paths": {
            "/health": {
                "get": {
                    "responses": {
                        "200": { "description": "Successful" }
                    }
                }
            },
            "/model": {
                "get": {
                    "responses": {
                        "200": { "description": "Successful" }
                    }
                }
            },
            "/predict": {
                "post": {
                    "responses": {
                        "200": { "description": "Successful" }
                    }
                }
            }
        }
    }))
}
