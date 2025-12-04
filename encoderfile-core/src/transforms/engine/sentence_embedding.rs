use crate::error::ApiError;

use super::{super::tensor::Tensor, Postprocessor, SentenceEmbeddingTransform};
use ndarray::{Array2, Array3, Axis, Ix2};

impl Postprocessor for SentenceEmbeddingTransform {
    type Input = (Array3<f32>, Array2<f32>);
    type Output = Array2<f32>;

    fn postprocess(&self, (data, mask): Self::Input) -> Result<Self::Output, ApiError> {
        let func = match &self.postprocessor {
            Some(p) => p,
            None => {
                let batch = data.len_of(Axis(0));
                let hidden = data.len_of(Axis(2));

                let mut out = Array2::<f32>::zeros((batch, hidden));

                for b in 0..batch {
                    let emb = data.slice(ndarray::s![b, .., ..]); // [seq_len, hidden]
                    let m = mask.slice(ndarray::s![b, ..]); // [seq_len]

                    // expand mask to [seq_len, hidden]
                    let m2 = m.insert_axis(Axis(1));

                    let weighted = &emb * &m2; // zero out padded tokens
                    let sum = weighted.sum_axis(Axis(0)); // sum over seq_len
                    let count = m.sum(); // number of real tokens

                    out.slice_mut(ndarray::s![b, ..]).assign(&(sum / count));
                }

                return Ok(out);
            }
        };

        let batch_size = data.shape()[0];

        let tensor = Tensor(data.into_dyn());

        let result = func
            .call::<Tensor>(tensor)
            .map_err(|e| ApiError::LuaError(e.to_string()))?
            .into_inner()
            .into_dimensionality::<Ix2>().map_err(|e| {
                tracing::error!("Failed to cast array into Ix2: {e}. Check your lua transform to make sure it returns a tensor of shape [batch_size, *]");
                ApiError::InternalError("Error postprocessing embeddings")
            })?;

        let result_shape = result.shape();

        if batch_size != result_shape[0] {
            tracing::error!(
                "Transform error: expected tensor of shape [{}, *], got tensor of shape {:?}",
                batch_size,
                result_shape
            );

            return Err(ApiError::InternalError("Error postprocessing embeddings"));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_pooling() {
        let engine = SentenceEmbeddingTransform::new(Some("")).expect("Failed to create engine");

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
        let engine = SentenceEmbeddingTransform::new(Some(
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
        let engine = SentenceEmbeddingTransform::new(Some(
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
        let engine = SentenceEmbeddingTransform::new(Some(r##"
        function Postprocess(arr, mask)
            return 1
        end
        "##)).expect("Failed to create engine");

        let arr = ndarray::Array3::<f32>::from_elem((16, 32, 128), 2.0);
        let mask = ndarray::Array2::<f32>::from_elem((16, 32), 1.0);

        let result = engine.postprocess((arr.clone(), mask));

        assert!(result.is_err())
    }
}
