use crate::error::ApiError;

use super::{super::tensor::Tensor, Postprocessor, SequenceClassificationTransform};
use ndarray::{Array2, Ix2};

impl Postprocessor for SequenceClassificationTransform {
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
                ApiError::LuaError("Error postprocessing sequence classifications".to_string())
            })?;

        let result_shape = result.shape();

        if expected_shape.as_slice() != result_shape {
            tracing::error!(
                "Transform error: expected tensor of shape {:?}, got tensor of shape {:?}",
                expected_shape.as_slice(),
                result_shape
            );

            return Err(ApiError::LuaError(
                "Error postprocessing sequence classifications".to_string(),
            ));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_cls_no_transform() {
        let engine =
            SequenceClassificationTransform::new(Some("")).expect("Failed to create Transform");

        let arr = ndarray::Array2::<f32>::from_elem((16, 2), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_seq_cls_transform() {
        let engine = SequenceClassificationTransform::new(Some(
            r##"
        function Postprocess(arr)
            return arr
        end
        "##,
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array2::<f32>::from_elem((16, 2), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_seq_cls_transform_bad_fn() {
        let engine = SequenceClassificationTransform::new(Some(
            r##"
        function Postprocess(arr)
            return 1
        end
        "##,
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array2::<f32>::from_elem((16, 2), 2.0);

        let result = engine.postprocess(arr.clone());

        assert!(result.is_err())
    }

    #[test]
    fn test_bad_dimensionality_transform_postprocessing() {
        let engine = SequenceClassificationTransform::new(Some(
            r##"
        function Postprocess(x)
            return x:sum_axis(1)
        end
        "##,
        ))
        .unwrap();

        let arr = ndarray::Array2::<f32>::from_elem((2, 2), 2.0);
        let result = engine.postprocess(arr.clone());

        assert!(result.is_err());

        if let Err(e) = result {
            match e {
                ApiError::LuaError(s) => {
                    assert!(s.contains("Error postprocessing sequence classifications"))
                }
                _ => panic!("Didn't return lua error"),
            }
        }
    }
}
