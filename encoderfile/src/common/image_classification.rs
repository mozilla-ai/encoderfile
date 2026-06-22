use crate::common::FromReadInput;
use crate::common::image_types::{ImageInfo, ImageLabelScore};
use anyhow::Result;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io::Read};
use utoipa::ToSchema;

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageClassificationRequest {
    pub images: Vec<ImageInfo>,
    pub metadata: Option<HashMap<String, String>>,
}

impl super::FromCliInput for ImageClassificationRequest {
    fn from_cli_input(inputs: Vec<String>) -> Self {
        let images = inputs
            .into_iter()
            .map(|path| {
                let image_data = std::fs::read(path).expect("Failed to read image file");
                let format =
                    image::guess_format(&image_data).expect("Failed to guess image format");
                ImageInfo {
                    image_bytes: Bytes::from(image_data),
                    image_format: format,
                }
            })
            .collect();

        Self {
            images,
            metadata: Some(HashMap::default()),
        }
    }
}

impl FromReadInput for ImageClassificationRequest {
    fn from_read_input(input: Vec<&mut impl Read>) -> Result<Self> {
        let images = input
            .into_iter()
            .map(|reader| {
                let mut image_data = Vec::new();
                reader
                    .read_to_end(&mut image_data)
                    .map_err(|e| anyhow::anyhow!("Failed to read image data: {}", e))?;
                let format = image::guess_format(&image_data)
                    .map_err(|e| anyhow::anyhow!("Failed to guess image format: {}", e))?;
                Ok(ImageInfo {
                    image_bytes: Bytes::from(image_data),
                    image_format: format,
                })
            })
            .collect::<Result<Vec<ImageInfo>>>()?;

        Ok(Self {
            images,
            metadata: Some(HashMap::default()),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, ToSchema, utoipa::ToResponse)]
pub struct ImageClassificationResponse {
    pub results: Vec<ImageClassificationResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct ImageClassificationResult {
    pub labels: Vec<ImageLabelScore>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageFormat;
    use std::fs::File;

    #[test]
    fn test_image_classification_request_from_read_input() {
        let mut file =
            File::open("../test-pictures/yoga01.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec)
            .expect("Failed to create request from read input");

        assert_eq!(request.images.len(), 1);
        assert_eq!(request.images[0].image_format, ImageFormat::Jpeg);
        assert!(!request.images[0].image_bytes.is_empty());
    }
}
