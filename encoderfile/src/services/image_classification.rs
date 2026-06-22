use crate::{
    common::{
        ImageClassificationRequest, ImageClassificationResponse, ImageClassificationResult,
        model_type,
    },
    error::ApiError,
    runtime::AppState,
    transforms::{DEFAULT_LIBS, Image, ImageClassificationTransform, Preprocessor},
};
use ndarray::{ArrayD, Axis, Ix4, Zip};

use super::inference::Inference;
use crate::inference::image_classification::image_classification;

// No service impl yet

impl Inference for AppState<model_type::ImageClassification> {
    type Input = ImageClassificationRequest;
    type Output = ImageClassificationResponse;

    fn inference(&self, request: impl Into<Self::Input>) -> Result<Self::Output, ApiError> {
        // let transform = ImageClassificationTransform::new(self.lua_libs.clone(), self.transform_str())?;

        let request = request.into();
        if request.images.is_empty() {
            return Err(ApiError::InputError("Cannot classify empty image list"));
        }

        let postprocess_code = r##"
        function Preprocess(img)
            return img:resize(224,224):to_array(3)
        end
        "##
        .to_string();

        let engine =
            ImageClassificationTransform::new(DEFAULT_LIBS.to_vec(), Some(postprocess_code))
                .expect("Failed to create engine");

        let num_channels = self.model_input_state.config.num_channels as usize;
        let rescale_factor = self
            .model_input_state
            .preprocessing
            .rescale_factor
            .ok_or(ApiError::InternalError("missing rescale factor"))?;
        let image_mean = self
            .model_input_state
            .preprocessing
            .image_mean
            .as_ref()
            .ok_or(ApiError::InternalError("missing image mean"))?;
        let image_std = self
            .model_input_state
            .preprocessing
            .image_std
            .as_ref()
            .ok_or(ApiError::InternalError("missing image std"))?;

        let images: Vec<ArrayD<f32>> = request
            .images
            .iter()
            .map(|image_info| {
                let img = image::load_from_memory(&image_info.image_bytes)
                    .expect("Failed to load image from bytes");
                let mut res = engine
                    .preprocess((Image(img), self.model_input_state.clone()))
                    .expect("Failed")
                    .into_inner();
                let mean_arr =
                    ndarray::Array::from_shape_vec((num_channels, 1, 1), image_mean.to_vec())
                        .expect("mean shape mismatch");
                let std_arr =
                    ndarray::Array::from_shape_vec((num_channels, 1, 1), image_std.to_vec())
                        .expect("std shape mismatch");
                Zip::from(&mut res)
                    .and_broadcast(&mean_arr)
                    .and_broadcast(&std_arr)
                    .for_each(|x, &m, &s| *x = (*x * rescale_factor - m) / s);
                res
            })
            .collect();

        let images_array = ndarray::stack(
            Axis(0),
            &images.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap()
        .into_dimensionality::<Ix4>()
        .unwrap();

        // TODO overlap preprocessing and inference, but for now just do it sequentially
        // Since we are adding gpu providers now, preprocessing could run in cpu while inference
        // is running. Using some sort of task queue will pave the way for more efficient batch
        // processing. However, it will not be implemented right now.

        let label_map = self.task_state.id2label.clone().unwrap();
        let mut entries: Vec<_> = label_map.iter().collect();
        entries.sort_by(|x, y| x.0.cmp(y.0));
        let classes: Vec<String> = entries
            .into_iter()
            .map(|(_, label)| label.clone())
            .collect();

        let labels_batch = image_classification(self.session.lock(), images_array, classes)?;

        Ok(ImageClassificationResponse {
            results: labels_batch
                .iter()
                .map(|labels| ImageClassificationResult {
                    labels: labels.clone(),
                })
                .collect(),
            metadata: request.metadata,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::FromReadInput;
    use crate::common::ImageClassificationRequest;
    use crate::common::model_type::ImageClassification;
    use crate::dev_utils;
    use std::fs::File;
    use std::sync::{Arc, Once};

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
        let mut file =
            File::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec)
            .expect("Failed to create request from read input");
        let response = state.inference(request).expect("Inference failed");
        println!("Inference response: {:?}", response);
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].labels.len(), 9);
        assert!(
            response.results[0]
                .labels
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.score.partial_cmp(&b.1.score).unwrap())
                .unwrap()
                .1
                .label
                == "Downward-Dog"
        ); // top label should be "yoga mat"
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

    #[test]
    fn test_image_classification_missing_rescale_factor() {
        init_tracing();

        let mut state =
            dev_utils::get_state::<ImageClassification>("../models/image_classification");
        Arc::get_mut(&mut state)
            .expect("state should not be shared")
            .model_input_state
            .preprocessing
            .rescale_factor = None;

        let mut file =
            File::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec)
            .expect("Failed to create request from read input");

        let response = state.inference(request);
        assert!(matches!(
            response,
            Err(ApiError::InternalError("missing rescale factor"))
        ));
    }

    #[test]
    fn test_image_classification_missing_image_mean() {
        init_tracing();

        let mut state =
            dev_utils::get_state::<ImageClassification>("../models/image_classification");
        Arc::get_mut(&mut state)
            .expect("state should not be shared")
            .model_input_state
            .preprocessing
            .image_mean = None;

        let mut file =
            File::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec)
            .expect("Failed to create request from read input");

        let response = state.inference(request);
        assert!(matches!(
            response,
            Err(ApiError::InternalError("missing image mean"))
        ));
    }

    #[test]
    fn test_image_classification_missing_image_std() {
        init_tracing();

        let mut state =
            dev_utils::get_state::<ImageClassification>("../models/image_classification");
        Arc::get_mut(&mut state)
            .expect("state should not be shared")
            .model_input_state
            .preprocessing
            .image_std = None;

        let mut file =
            File::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");
        let file_vec = vec![&mut file];
        let request = ImageClassificationRequest::from_read_input(file_vec)
            .expect("Failed to create request from read input");

        let response = state.inference(request);
        assert!(matches!(
            response,
            Err(ApiError::InternalError("missing image std"))
        ));
    }
}
