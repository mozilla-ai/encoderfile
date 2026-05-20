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
use ndarray::{Array4, s};

use super::inference::Inference;
use crate::inference::image_classification::image_classification;

// No service impl yet

const DEFAULT_FILTER_TYPE: image::imageops::FilterType = image::imageops::FilterType::Triangle;

impl Inference for AppState<model_type::ImageClassification>
{
    type Input = ImageClassificationRequest;
    type Output = ImageClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        let request = request.into();
        if request.images.is_empty() {
            return Err(ApiError::InputError("Cannot classify empty image list"));
        }
        let rescale_factor = self.model_input_state.preprocessing.rescale_factor.ok_or(ApiError::InternalError("missing rescale factor"))?;
        let image_mean = self.model_input_state.preprocessing.image_mean.as_ref().ok_or(ApiError::InternalError("missing image mean"))?;
        let image_std = self.model_input_state.preprocessing.image_std.as_ref().ok_or(ApiError::InternalError("missing image std"))?;
        // bilinear resampling

        // convert input image into flattened rbg
        let images: Vec<RgbImage> = request.images.iter().map(|image_info| {
            let img = image::load_from_memory(&image_info.image_bytes).expect("Failed to load image from bytes");
            img
                .resize_exact(
                    self.model_input_state.preprocessing.size.as_ref().unwrap().width.unwrap(),
                    self.model_input_state.preprocessing.size.as_ref().unwrap().height.unwrap(),
                    DEFAULT_FILTER_TYPE
                )
                .to_rgb8()
        }).collect();
        let batch_size = request.images.len();
        let num_channels = self.model_input_state.config.num_channels as usize;
        let height = self.model_input_state.preprocessing.size.as_ref().unwrap().height.unwrap() as usize;
        let width = self.model_input_state.preprocessing.size.as_ref().unwrap().width.unwrap() as usize;

        if num_channels != 3 {
            return Err(ApiError::InputError("Image classification currently expects 3 RGB channels"));
        }

        let mut images_array = Array4::<f32>::zeros((batch_size, num_channels, height, width));
        for (image_idx, img) in images.into_iter().enumerate() {
            let raw = img.into_raw();

            // The image crate stores RGB bytes in HWC order; rewrite into NCHW.
            for y in 0..height {
                for x in 0..width {
                    let pixel_offset = (y * width + x) * num_channels;
                    for c in 0..num_channels {
                        images_array[[image_idx, c, y, x]] = raw[pixel_offset + c] as f32;
                    }
                }
            }
        }
        // TODO make parallel
        for c in 0..num_channels {
            let mean = image_mean[c];
            let std = image_std[c];
            images_array.slice_mut(s![.., c, .., ..]).mapv_inplace(|x| ((x * rescale_factor) - mean) / std);
        }

        let label_map = self.task_state.id2label.clone().unwrap();
        let mut entries: Vec<_> = label_map.iter().collect();
        entries.sort_by(|x, y| x.0.cmp(y.0));
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
        let mut file = File::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec).expect("Failed to create request from read input");
        let response = state.inference(request).expect("Inference failed");
        println!("Inference response: {:?}", response);
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].labels.len(), 9);
        assert!(response.results[0].labels.iter().enumerate().max_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap()).unwrap().1.label == "Downward-Dog"); // top label should be "yoga mat"
    }

    #[test]
    fn test_image_classification_empty() {
        init_tracing();

        let state = dev_utils::get_state::<ImageClassification>("../models/image_classification");
        let request = ImageClassificationRequest {
            images: vec![],
            metadata: Default::default(),
        };
        let response = state.inference(request);
        assert!(response.is_err());
    }
}