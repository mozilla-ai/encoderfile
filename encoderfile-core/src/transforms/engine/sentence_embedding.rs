use crate::{common::model_type, error::ApiError};

use super::{super::tensor::Tensor, Postprocessor, Transform};
use ndarray::{Array2, Array3, Ix2};

impl Postprocessor for Transform<model_type::SentenceEmbedding> {
    type Input = (Array3<f32>, Array2<f32>);
    type Output = Array2<f32>;

    fn postprocess(&self, (data, mask): Self::Input) -> Result<Self::Output, ApiError> {
        let func = match &self.postprocessor {
            Some(p) => p,
            None => {
                let Tensor(mean_pooled) = Tensor(data.into_dyn())
                    .mean_pool(Tensor(mask.into_dyn()))
                    .map_err(|e| {
                        tracing::error!(
                            "Failed to mean pool. This should not happen. More details: {:?}",
                            e
                        );
                        ApiError::InternalError("Failed to postprocess embeddings")
                    })?;

                return mean_pooled.into_dimensionality::<Ix2>()
                    .map_err(|e| {
                        tracing::error!("Failed to cast mean pool results into Ix2. This should not happen. More details: {:?}", e);
                        ApiError::InternalError("Failed to postprocess embeddings")
                    });
            }
        };

        let batch_size = data.shape()[0];

        let tensor = Tensor(data.into_dyn());

        let result = func
            .call::<Tensor>((tensor, Tensor(mask.into_dyn())))
            .map_err(|e| ApiError::LuaError(e.to_string()))?
            .into_inner()
            .into_dimensionality::<Ix2>().map_err(|e| {
                tracing::error!("Failed to cast array into Ix2: {e}. Check your lua transform to make sure it returns a tensor of shape [batch_size, *]");
                ApiError::LuaError("Error postprocessing embeddings".to_string())
            })?;

        let result_shape = result.shape();

        if batch_size != result_shape[0] {
            tracing::error!(
                "Transform error: expected tensor of shape [{}, *], got tensor of shape {:?}",
                batch_size,
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
    use ndarray::Axis;

    #[test]
    fn test_no_pooling() {
        let engine = Transform::<model_type::SentenceEmbedding>::new(Some(""))
            .expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);
        let mask = ndarray::Array2::<f32>::from_elem((16, 32), 1.0);

        let result = engine
            .postprocess((arr.clone(), mask))
            .expect("Failed to compute pool");

        assert_eq!(result.shape(), [16, 128]);

        // if all elements are the same and all mask = 1, should return mean axis array
        assert_eq!(arr.mean_axis(Axis(1)), Some(result));
    }

    #[test]
    fn test_successful_pool() {
        let engine = Transform::<model_type::SentenceEmbedding>::new(Some(
            r##"
        function Postprocess(arr, mask)
            -- sum along second axis (lol)
            return arr:sum_axis(2)
        end
        "##,
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);
        let mask = ndarray::Array2::<f32>::from_elem((16, 32), 1.0);

        let result = engine
            .postprocess((arr, mask))
            .expect("Failed to compute pool");

        assert_eq!(result.shape(), [16, 128])
    }

    #[test]
    fn test_bad_dim_pool() {
        let engine = Transform::<model_type::SentenceEmbedding>::new(Some(
            r##"
        function Postprocess(arr, mask)
            return arr
        end
        "##,
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);
        let mask = ndarray::Array2::<f32>::from_elem((16, 32), 1.0);

        let result = engine.postprocess((arr, mask));

        assert!(result.is_err());
    }

    #[test]
    fn test_sentence_embedding_transform_bad_fn() {
        let engine = Transform::<model_type::SentenceEmbedding>::new(Some(
            r##"
        function Postprocess(arr, mask)
            return 1
        end
        "##,
        ))
        .expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);
        let mask = ndarray::Array2::<f32>::from_elem((16, 32), 1.0);

        let result = engine.postprocess((arr.clone(), mask));

        assert!(result.is_err())
    }

    #[test]
    fn test_bad_dimensionality_transform_postprocessing() {
        let engine = Transform::<model_type::SentenceEmbedding>::new(Some(
            r##"
        function Postprocess(arr, mask)
            return arr
        end
        "##,
        ))
        .unwrap();

        let arr = ndarray::Array3::<f32>::from_elem((3, 3, 3), 2.0);
        let mask = ndarray::Array2::<f32>::from_elem((3, 3), 1.0);
        let result = engine.postprocess((arr.clone(), mask));

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
