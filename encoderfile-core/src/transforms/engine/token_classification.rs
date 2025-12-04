use crate::error::ApiError;

use super::{super::tensor::Tensor, Postprocessor, TokenClassificationTransform};
use ndarray::{Array3, Ix3};

impl Postprocessor for TokenClassificationTransform {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

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
            .into_dimensionality::<Ix3>().map_err(|e| {
                tracing::error!("Failed to cast array into Ix3: {e}. Check your lua transform to make sure it returns a tensor of shape [batch_size, seq_len, num_classes]");
                ApiError::InternalError("Error postprocessing token classifications")
            })?;

        let result_shape = result.shape();

        if expected_shape.as_slice() != result_shape {
            tracing::error!(
                "Transform error: expected tensor of shape {:?}, got tensor of shape {:?}",
                expected_shape.as_slice(),
                result_shape
            );

            return Err(ApiError::InternalError(
                "Error postprocessing token classifications",
            ));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_cls_no_transform() {
        let engine =
            TokenClassificationTransform::new(Some("")).expect("Failed to create Transform");

        let arr = ndarray::Array3::<f32>::from_elem((32, 16, 2), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_token_cls_identity_transform() {
        let engine = TokenClassificationTransform::new(Some(r##"
        function Postprocess(arr)
            return arr
        end
        "##)).expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 2), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_token_cls_transform_bad_fn() {
        let engine = TokenClassificationTransform::new(Some(r##"
        function Postprocess(arr)
            return 1
        end
        "##)).expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 2), 2.0);

        let result = engine.postprocess(arr.clone());

        assert!(result.is_err())
    }
}
