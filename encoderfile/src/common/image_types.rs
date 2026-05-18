use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use image::ImageFormat;
use bytes::Bytes;

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


