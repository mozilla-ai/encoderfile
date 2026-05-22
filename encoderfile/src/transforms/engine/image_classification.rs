use crate::{common::model_type, error::ApiError, runtime::ImageInputState};

use super::{super::image::Image, super::tensor::Tensor, Postprocessor, Preprocessor, Transform};
use ndarray::{Array2, Ix2};

impl Postprocessor for Transform<model_type::ImageClassification> {
    type Input = Array2<f32>;
    type Output = Array2<f32>;

    fn postprocess(&self, data: Self::Input) -> Result<Self::Output, ApiError> {
        let func = match self.postprocessor() {
            Some(p) => p,
            None => return Ok(data),
        };

        let expected_shape = data.shape().to_owned();

        let tensor = Tensor(data.into_dyn());

        let result = func
            .call::<Tensor>(tensor)
            .map_err(|e| ApiError::LuaError(e.to_string()))?
            .into_inner()
            .into_dimensionality::<Ix2>().map_err(|e| {
                tracing::error!("Failed to cast array into Ix2: {e}. Check your lua transform to make sure it returns a tensor of shape [batch_size, num_classes]");
                ApiError::LuaError("Error postprocessing image classifications".to_string())
            })?;

        let result_shape = result.shape();

        if expected_shape.as_slice() != result_shape {
            tracing::error!(
                "Transform error: expected tensor of shape {:?}, got tensor of shape {:?}",
                expected_shape.as_slice(),
                result_shape
            );

            return Err(ApiError::LuaError(
                "Error postprocessing image classifications".to_string(),
            ));
        }

        Ok(result)
    }
}

impl Preprocessor for Transform<model_type::ImageClassification> {
    type Input = (Image, ImageInputState);
    type Output = Tensor;

    fn preprocess(&self, (image, config): Self::Input) -> Result<Self::Output, ApiError> {
        let func = match self.preprocessor() {
            Some(p) => p,
            None => {
                return Err(ApiError::InternalError(
                    "No preprocessor defined for this model",
                ));
            }
        };

        self.lua
            .globals()
            .set("input_config", config)
            .map_err(|e| ApiError::LuaError(e.to_string()))?;

        func.call::<Tensor>(image)
            .map_err(|e| ApiError::LuaError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::DEFAULT_LIBS;

    #[test]
    fn test_image_cls_no_transform() {
        let engine = Transform::<model_type::ImageClassification>::new(
            DEFAULT_LIBS.to_vec(),
            Some("".to_string()),
        )
        .expect("Failed to create Transform");

        let arr = ndarray::Array2::<f32>::from_elem((32, 16), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_image_cls_identity_transform() {
        let engine = Transform::<model_type::ImageClassification>::new(
            DEFAULT_LIBS.to_vec(),
            Some(
                r##"
        function Postprocess(arr)
            return arr
        end
        "##
                .to_string(),
            ),
        )
        .expect("Failed to create engine");

        let arr = ndarray::Array2::<f32>::from_elem((16, 32), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_image_cls_transform_bad_fn() {
        let engine = Transform::<model_type::ImageClassification>::new(
            DEFAULT_LIBS.to_vec(),
            Some(
                r##"
        function Postprocess(arr)
            return 1
        end
        "##
                .to_string(),
            ),
        )
        .expect("Failed to create engine");

        let arr = ndarray::Array2::<f32>::from_elem((16, 32), 2.0);

        let result = engine.postprocess(arr.clone());

        assert!(result.is_err())
    }

    #[test]
    fn test_bad_dimensionality_transform_postprocessing() {
        let engine = Transform::<model_type::ImageClassification>::new(
            DEFAULT_LIBS.to_vec(),
            Some(
                r##"
        function Postprocess(x)
            return x:sum_axis(1)
        end
        "##
                .to_string(),
            ),
        )
        .unwrap();

        let arr = ndarray::Array2::<f32>::from_elem((3, 3), 2.0);
        let result = engine.postprocess(arr.clone());

        assert!(result.is_err());

        if let Err(e) = result {
            match e {
                ApiError::LuaError(s) => {
                    assert!(s.contains("Error postprocessing image classifications"))
                }
                _ => panic!("Didn't return lua error"),
            }
        }
    }

    #[test]
    fn test_image_preprocess() {
        let engine = Transform::<model_type::ImageClassification>::new(
            DEFAULT_LIBS.to_vec(),
            Some(
                r##"
        function Preprocess(img)
            return img:resize(input_config.size_height, input_config.size_width):to_array(input_config.num_channels)
        end
        "##
                .to_string(),
            ),
        )
        .expect("Failed to create engine");

        let img = image::open("../test-pictures/yoga02.jpg").expect("Failed to open test image");

        let config = ImageInputState {
            config: crate::runtime::ImageConfig {
                num_channels: 3,
                image_size: Some(224),
            },
            preprocessing: crate::runtime::ImagePreprocessing {
                rescale_factor: None,
                image_mean: None,
                image_std: None,
                do_normalize: None,
                do_rescale: None,
                do_resize: None,
                image_processor_type: None,
                size: Some(crate::runtime::ImageSize {
                    height: Some(224),
                    width: Some(224),
                    shortest_edge: None,
                }),
            },
        };
        let result = engine.preprocess((Image(img), config)).expect("Failed");

        assert!(result.into_inner().shape() == [3, 224, 224]);
    }
}
