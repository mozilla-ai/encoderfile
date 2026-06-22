use crate::{common, generated::image_types::ImageLabels};

tonic::include_proto!("encoderfile.image_classification");

impl From<ImageClassificationRequest> for common::ImageClassificationRequest {
    fn from(val: ImageClassificationRequest) -> Self {
        let images = val
            .inputs
            .into_iter()
            .map(|input| {
                common::ImageInfo {
                    image_bytes: bytes::Bytes::from(input.image),
                    image_format: image::ImageFormat::Png, // TODO: detect format properly
                }
            })
            .collect();
        Self {
            images,
            metadata: if val.metadata.is_empty() {
                None
            } else {
                Some(val.metadata)
            },
        }
    }
}

impl From<common::ImageClassificationResponse> for ImageClassificationResponse {
    fn from(val: common::ImageClassificationResponse) -> Self {
        Self {
            results: val
                .results
                .into_iter()
                .map(|result| result.into())
                .collect(),
            metadata: val.metadata.unwrap_or_default(),
        }
    }
}

impl From<common::ImageClassificationResult> for ImageLabels {
    fn from(val: common::ImageClassificationResult) -> Self {
        ImageLabels {
            labels: val.labels.into_iter().map(|label| label.into()).collect(),
        }
    }
}
