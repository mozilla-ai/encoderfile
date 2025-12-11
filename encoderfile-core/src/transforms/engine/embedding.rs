use crate::{common::model_type, error::ApiError};

use super::{super::tensor::Tensor, Postprocessor, Transform};
use ndarray::{Array3, Ix3};

impl Postprocessor for Transform<model_type::Embedding> {
    type Input = Array3<f32>;
    type Output = Array3<f32>;

    fn postprocess(&self, data: Self::Input) -> Result<Self::Output, ApiError> {
        let func = match self.postprocessor() {
            Some(p) => p,
            None => return Ok(data),
        };

        let batch_size = data.shape()[0];
        let seq_len = data.shape()[1];

        let tensor = Tensor(data.into_dyn());

        let result = func
            .call::<Tensor>(tensor)
            .map_err(|e| ApiError::LuaError(e.to_string()))?
            .into_inner()
            .into_dimensionality::<Ix3>().map_err(|e| {
                tracing::error!("Transform error: Failed to cast array into Ix3: {e}. Check your lua transform to make sure it returns a tensor of shape [batch_size, seq_len, *]");
                ApiError::LuaError("Error postprocessing embeddings".to_string())
            })?;

        let result_shape = result.shape();

        if batch_size != result_shape[0] || seq_len != result_shape[1] {
            tracing::error!(
                "Transform error: expected tensor of shape [{}, {}, *], got tensor of shape {:?}",
                batch_size,
                seq_len,
                result_shape
            );

            return Err(ApiError::LuaError(
                "Error postprocessing embeddings".to_string(),
            ));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_no_transform() {
        let engine = Transform::<model_type::Embedding>::new(Some("".to_string()))
            .expect("Failed to create Transform");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_embedding_identity_transform() {
        let engine = Transform::<model_type::Embedding>::new(Some(
            r##"
        function Postprocess(arr)
            return arr
        end
        "##
            .to_string(),
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);

        let result = engine.postprocess(arr.clone()).expect("Failed");

        assert_eq!(arr, result);
    }

    #[test]
    fn test_embedding_transform_bad_fn() {
        let engine = Transform::<model_type::Embedding>::new(Some(
            r##"
        function Postprocess(arr)
            return 1
        end
        "##
            .to_string(),
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);

        let result = engine.postprocess(arr.clone());

        assert!(result.is_err())
    }

    #[test]
    fn test_bad_dimensionality_transform_postprocessing() {
        let engine = Transform::<model_type::Embedding>::new(Some(
            r##"
        function Postprocess(x)
            return x:sum_axis(1)
        end
        "##
            .to_string(),
        ))
        .unwrap();

        let arr = ndarray::Array3::<f32>::from_elem((3, 3, 3), 2.0);
        let result = engine.postprocess(arr.clone());

        assert!(result.is_err());

        if let Err(e) = result {
            match e {
                ApiError::LuaError(s) => {
                    assert!(s.contains("Error postprocessing embeddings"))
                }
                _ => panic!("Didn't return lua error"),
            }
        }
    }
}
