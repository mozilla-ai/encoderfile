use crate::{
    common::{
        ImageClassificationRequest,
        ImageClassificationResponse,
        ImageClassificationResult,
        ImageLabelScore,
        model_type
    },

    error::ApiError,
    runtime::AppState,
};

use image::{DynamicImage, GenericImageView};

use super::inference::Inference;

// No service impl yet

impl Inference for AppState<model_type::ImageClassification>
{
    type Input = ImageClassificationRequest;
    type Output = ImageClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        // FIXME
        /*
        // convert input image into flattened rbg
        let image = image::load_from_memory(&request.image_info.image_bytes)?;
        let (width, height) = image.dimensions();
        let image = image.to_rgb8();
        let mut flattened_rgb = Vec::with_capacity((width * height * 3) as usize);
        for pixel in image.pixels() {
            flattened_rgb.push(pixel[0] as f32);
            flattened_rgb.push(pixel[1] as f32);
            flattened_rgb.push(pixel[2] as f32);
        }

        let labels_batch = inference::image_classification::image_classification(self.session.lock(), &transform, encodings)?;
        */

        let dummy_labels = vec![
            ImageLabelScore {
                label: "dummy1".to_string(),
                score: 0.9,
            },
            ImageLabelScore {
                label: "dummy2".to_string(),
                score: 0.1,
            },
        ];

        Ok(ImageClassificationResponse {
            results: vec![ImageClassificationResult { labels: dummy_labels }],
            metadata: request.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::common::model_type::ImageClassification;
    use crate::dev_utils;
    use crate::common::ImageClassificationRequest;
    use crate::common::FromReadInput;
    use std::fs::File;
    use super::*;

    #[test]
    fn test_image_classification_request_from_file() {
        let state = dev_utils::get_state::<ImageClassification>("../models/image_classification");
        let mut file = File::open("../test-pictures/w3c_home.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec).expect("Failed to create request from read input");
        let response = state.inference(request).expect("Inference failed");
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].labels.len(), 2);
        assert_eq!(response.results[0].labels[0].label, "dummy1");
        assert_eq!(response.results[0].labels[0].score, 0.9);
        assert_eq!(response.results[0].labels[1].label, "dummy2");
        assert_eq!(response.results[0].labels[1].score, 0.1);
    }
}