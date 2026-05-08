use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Read};
use utoipa::ToSchema;
use anyhow::Result;
use crate::common::FromReadInput;
use image::ImageFormat;
use bytes::Bytes;
use crate::transport::http::multipart_openapi::{FromMultipart, MultipartApiError};

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


