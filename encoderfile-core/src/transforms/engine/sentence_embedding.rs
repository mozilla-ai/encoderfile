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
