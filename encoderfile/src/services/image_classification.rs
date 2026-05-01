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

/*
impl Inference for AppState<model_type::ImageClassification> {
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
    */
