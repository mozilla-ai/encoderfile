use bytes::Bytes;
use image::ImageFormat;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageInfo {
    pub image_bytes: Bytes,
    pub image_format: ImageFormat,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct ImageLabelScore {
    pub label: String,
    pub score: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct ImageLabels {
    pub labels: Vec<ImageLabelScore>,
}
