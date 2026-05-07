use crate::{
    common::{
        ImageClassificationRequest,
        ImageClassificationResponse,
        ImageClassificationResult,
        model_type
    },

    error::ApiError,
    runtime::AppState,
};
use image::RgbImage;
use ndarray::{Array4};

use super::inference::Inference;
use crate::inference::image_classification::image_classification;

// No service impl yet

const DEFAULT_FILTER_TYPE: image::imageops::FilterType = image::imageops::FilterType::Nearest;

impl Inference for AppState<model_type::ImageClassification>
{
    type Input = ImageClassificationRequest;
    type Output = ImageClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();

        // convert input image into flattened rbg
        let images: Vec<RgbImage> = (&request.images).into_iter().map(|image_info| {
            let img = image::load_from_memory(&image_info.image_bytes).expect("Failed to load image from bytes");
            img
                .resize_exact(
                    self.per_model_input_state.width.unwrap(),
                    self.per_model_input_state.height.unwrap(),
                    DEFAULT_FILTER_TYPE
                )
                .to_rgb8()
        }).collect();
        let images_array = Array4::from_shape_vec(
            (
                request.images.len().clone() as usize,
                self.per_model_input_state.num_channels as usize,
                self.per_model_input_state.height.unwrap() as usize,
                self.per_model_input_state.width.unwrap() as usize
            ),
            images
                .into_iter()
                .flat_map(|img| img.into_raw().into_iter().map(|pixel| pixel as f32))
                .collect()
        ).expect("Failed to convert images to ndarray");

        let label_map = self.per_task_state.id2label.clone().unwrap();
        let mut entries: Vec<_> = label_map.iter().collect();
        entries.sort_by(|x, y| x.0.cmp(&y.0));
        let classes: Vec<String> = entries.into_iter().map(|(_, label)| label.clone()).collect();

        let labels_batch = image_classification(
            self.session.lock(),
            images_array,
            // COMMENT having optional fields complicates things later on, but otoh
            // it allows models with variations of these fields
            classes)?;

        Ok(ImageClassificationResponse {
            results: labels_batch.iter().map(|labels| ImageClassificationResult { labels: labels.clone() }).collect(),
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
    use std::sync::Once;
    use super::*;

    fn init_tracing() {
        static TRACING: Once = Once::new();

        TRACING.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug,ort=warn")),
                )
                .with_test_writer()
                .try_init();
        });
    }

    #[test]
    fn test_image_classification_request_from_file() {
        init_tracing();

        let state = dev_utils::get_state::<ImageClassification>("../models/image_classification");
        let mut file = File::open("../test-pictures/w3c_home.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec).expect("Failed to create request from read input");
        let response = state.inference(request).expect("Inference failed");
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].labels.len(), 2);
        assert_eq!(response.results[0].labels[0].label, "normal");
        assert_eq!(response.results[0].labels[0].score, 1.5378942);
        assert_eq!(response.results[0].labels[1].label, "nsfw");
        assert_eq!(response.results[0].labels[1].score, -1.6556994);
    }
}